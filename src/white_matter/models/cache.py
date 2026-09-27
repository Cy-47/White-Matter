"""Hugging Face cache storage with explicit logical positions and KV ownership."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import cast

import torch
from transformers.cache_utils import Cache, DynamicLayer, StaticLayer


class DecoderCache(Cache):
    """Native HF storage per KV owner; feedback channels fold into the head axis.

    prefix_slots reserves dummy tokens without changing the logical token count.
    Static storage supports unpadded single documents. Dynamic storage additionally
    supports document boundaries and padding.
    """

    def __init__(self, prefix_slots: tuple[int, ...], max_cache_len: int | None = None) -> None:
        if max_cache_len is not None and max_cache_len < 1:
            raise ValueError("max_cache_len must be positive")
        self.prefix_slots = prefix_slots
        self.capacity = max_cache_len
        super().__init__(
            layers=[
                DynamicLayer() if max_cache_len is None else StaticLayer(max_cache_len + extra)
                for extra in prefix_slots
            ]
        )
        self.seen_tokens = 0
        self.position: torch.Tensor | None = None
        self.document_ids: torch.Tensor | None = None
        self.last_document_id: torch.Tensor | None = None
        self.next_document_id: torch.Tensor | None = None

    @property
    def is_compileable(self) -> bool:
        # Explicit decode capture keeps Python metadata outside graph replay.
        return False

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        # RoPE/norm may promote keys; cache the dtype actually consumed by attention.
        return super().update(key_states.to(value_states.dtype), value_states, layer_idx, *args, **kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seen_tokens

    @contextmanager
    def _prefill_batch(self, start: int, stop: int, *, batch_size: int):
        """Prefill consecutive rows, sharing KV and committing the last chunk.

        Start at row zero and cover the full batch in order. If a chunk fails,
        reset the cache before retrying: shared writes are not transactional.
        """
        initialized = all(layer.is_initialized for layer in self.layers)
        if initialized:
            if self.position is None:
                self.position = torch.zeros((batch_size, 1), dtype=torch.long, device=self.layers[0].keys.device)
            chunk = copy.copy(self)
            chunk.position = self.position[start:stop]
            chunk.layers = [copy.copy(layer) for layer in self.layers]
            for layer in chunk.layers:
                layer.keys, layer.values = layer.keys[start:stop], layer.values[start:stop]
                layer.max_batch_size = stop - start
                layer.cumulative_length = layer.cumulative_length.clone()
        else:
            chunk = type(self)(self.prefix_slots, self.capacity)
        yield chunk
        if not initialized:
            # Infer owner dimensions from the first real chunk, as HF does.
            for target, source in zip(self.layers, chunk.layers, strict=True):
                target.lazy_initialization(
                    *(t[:1].expand(batch_size, -1, -1, -1) for t in (source.keys, source.values))
                )
                target.keys[start:stop].copy_(source.keys)
                target.values[start:stop].copy_(source.values)
            self.position = chunk.position.new_zeros((batch_size, 1))
            self.position[start:stop].copy_(chunk.position)
        if stop == batch_size:
            self.seen_tokens = chunk.seen_tokens
            for target, source in zip(self.layers, chunk.layers, strict=True):
                target.cumulative_length.copy_(source.cumulative_length)

    def _snapshot_position(self) -> tuple:
        """Save static-cache positions for trials that leave the prefix unchanged."""
        if (
            self.capacity is None
            or self.position is None
            or self.document_ids is not None
            or not all(layer.is_initialized for layer in self.layers)
        ):
            raise ValueError("position snapshots require an initialized single-document static cache")
        return self.seen_tokens, self.position.clone(), tuple(layer.cumulative_length.clone() for layer in self.layers)

    def _restore_position(self, snapshot: tuple) -> None:
        """Restore counters in place, without restoring KV or supporting rollback."""
        seen, position, lengths = snapshot
        self.position.copy_(position)
        for layer, length in zip(self.layers, lengths, strict=True):
            layer.cumulative_length.copy_(length)
        self.seen_tokens = seen

    @torch.no_grad()
    def repeat_prefix(self, batch_size: int, capacity: int) -> DecoderCache:
        """Copy one committed single-document prefix into independent static rows."""
        if (
            self.position is None
            or self.position.shape[0] != 1
            or self.document_ids is not None
            or not self.seen_tokens
        ):
            raise ValueError("repeat_prefix requires a single-row, single-document prefix")
        if batch_size < 1 or capacity < self.seen_tokens:
            raise ValueError("batch_size must be positive and capacity must fit the prefix")
        cache = type(self)(self.prefix_slots, capacity)
        for index, (layer, extra) in enumerate(zip(self.layers, self.prefix_slots, strict=True)):
            end = self.seen_tokens + extra
            cache.update(*(t[..., :end, :].expand(batch_size, -1, -1, -1) for t in (layer.keys, layer.values)), index)
        cache.position = self.position.expand(batch_size, -1).clone()
        cache.seen_tokens = self.seen_tokens
        return cache

    def prepare(self, x, input_ids, attention_mask, document_ids, separator, position_ids):
        batch, length = x.shape[:2]
        if position_ids is not None and (
            position_ids.shape not in {(1, length), (batch, length)}
            or position_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("position_ids must be integer (1,T) or (B,T)")
        if attention_mask is not None and attention_mask.shape not in {
            (batch, length),
            (batch, self.seen_tokens + length),
        }:
            raise ValueError("attention_mask must cover the new tokens or full prefix")
        if self.capacity is not None:
            if self.seen_tokens + length > self.capacity:
                raise ValueError("input exceeds cache capacity")
            if separator is not None or document_ids is not None:
                raise ValueError("static cache requires a single document: set document_separator_token_id=None")
            if attention_mask is not None:
                if not bool(attention_mask.all()):
                    raise ValueError("static cache does not support padding; use dynamic cache")
                attention_mask = None
        if self.position is None:
            self.position = torch.zeros((batch, 1), dtype=torch.long, device=x.device)
        if self.position.shape[0] != batch or self.position.device != x.device:
            raise ValueError("cache batch/device differs from inputs; reorder or reset the cache")
        if (
            separator is None
            and document_ids is None
            and self.document_ids is None
            and (attention_mask is None or bool(attention_mask.all()))
        ):
            positions = self.position + torch.arange(length, device=x.device)
            documents, valid = None, None
        else:
            documents, positions, valid = self.append_document_metadata(
                x, input_ids, attention_mask, document_ids, separator
            )
        if position_ids is not None:
            positions = position_ids.to(x.device).expand(batch, -1)
        return documents, positions, valid

    def advance(self, positions: torch.Tensor, valid: torch.Tensor | None = None) -> None:
        if valid is None:
            self.position.copy_(positions[:, -1:] + 1)
        else:
            # Padding consumes storage but does not advance the logical position.
            indices = torch.arange(1, positions.shape[1] + 1, device=positions.device)
            end = torch.where(valid, indices, 0).amax(1, keepdim=True)
            self.position.copy_(torch.cat((self.position, positions + 1), dim=1).gather(1, end))
        self.seen_tokens += positions.shape[1]

    def lengths(self, layer_idx: int, batch_size: int) -> torch.Tensor:
        layer = self.layers[layer_idx]
        if self.capacity is not None:
            return layer.cumulative_length.expand(batch_size).to(
                dtype=torch.int32, memory_format=torch.contiguous_format
            )
        return torch.full((batch_size,), layer.keys.shape[-2], dtype=torch.int32, device=layer.keys.device)

    def append_document_metadata(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        document_ids: torch.Tensor | None,
        separator_token_id: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append document metadata and return positions for the new input slots."""
        batch, length = x.shape[:2]
        seen = self.get_seq_length()
        valid = (
            torch.ones((batch, length), device=x.device, dtype=torch.bool)
            if attention_mask is None
            else attention_mask[:, -length:].to(device=x.device, dtype=torch.bool)
        )
        if self.document_ids is None:
            self.last_document_id = torch.full((batch, 1), 0 if seen else -1, device=x.device, dtype=torch.long)
            self.next_document_id = torch.zeros_like(self.last_document_id)
            self.document_ids = torch.zeros((batch, seen), device=x.device, dtype=torch.long)
        elif self.document_ids.shape[0] != batch or self.document_ids.device != x.device:
            raise ValueError("cache batch/device differs from the inputs; reorder or reset the cache")
        assert self.last_document_id is not None
        assert self.next_document_id is not None
        separators = (
            torch.zeros_like(valid)
            if input_ids is None or separator_token_id is None
            else (input_ids == separator_token_id) & valid
        ).long()
        if document_ids is None:
            documents = self.next_document_id + separators.cumsum(1) - separators
        else:
            if document_ids.shape != (batch, length) or document_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("cached document_ids must be an integer (B,T) tensor with persistent document IDs")
            documents = document_ids.to(x.device)
            if bool((documents < 0).any()):
                raise ValueError("document IDs must be nonnegative")

        # Ignore padding when finding the preceding document and advancing RoPE.
        indices = torch.arange(1, length + 1, device=x.device).expand(batch, length)
        last_valid = torch.where(valid, indices, 0).cummax(1).values
        previous = torch.cat((torch.zeros_like(last_valid[:, :1]), last_valid[:, :-1]), dim=1)
        history = torch.cat((self.last_document_id, documents), dim=1)
        starts = valid & (documents != history.gather(1, previous))
        steps = valid.long().cumsum(1) - 1
        start_steps = torch.where(starts, steps, -self.position)
        positions = (steps - start_steps.cummax(1).values).masked_fill(~valid, 0)
        end = last_valid[:, -1:]
        self.last_document_id = history.gather(1, end)
        self.next_document_id = torch.cat((self.next_document_id, documents + separators), dim=1).gather(1, end)
        new_document_ids = documents.masked_fill(~valid, -2)
        self.document_ids = torch.cat((self.document_ids, new_document_ids), dim=1)
        return documents, positions, valid

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        super().reorder_cache(cast(torch.LongTensor, beam_idx))
        for name in ("position", "document_ids", "last_document_id", "next_document_id"):
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.index_select(0, beam_idx.to(tensor.device)))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self.reorder_cache(indices)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.position is not None:
            indices = torch.arange(self.position.shape[0], device=self.position.device).repeat_interleave(repeats)
            self.reorder_cache(indices)

    def reset(self) -> None:
        if self.capacity is None:
            self.layers = [DynamicLayer() for _ in self.layers]
            self.position = None
        else:
            super().reset()
            if self.position is not None:
                self.position.zero_()
        self.seen_tokens = 0
        self.document_ids = self.last_document_id = self.next_document_id = None

    def crop(self, max_length: int) -> None:
        raise NotImplementedError("Cache rollback is not supported; use ordinary sampling or beam search")

"""Autoregressive feedback execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from white_matter._typing import compiler_disable, eager_loop
from white_matter.modules.documents import document_start_mask

if TYPE_CHECKING:
    from white_matter.blocks.white_matter import WhiteMatterBlock

from .checkpointing import _OffloadedPackedARCheckpoint


@compiler_disable
def _offloaded_checkpoint(*args: Any) -> torch.Tensor:
    # CPU offload and explicit reverse-mode scheduling retain their validated
    # packed-AR backend. Tensor steps compile independently inside this driver.
    return cast(torch.Tensor, _OffloadedPackedARCheckpoint.apply(*args))


def forward_autoregressive(
    self: WhiteMatterBlock,
    x: torch.Tensor,
    *,
    document_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    checkpoint_chunk_size: int = 0,
    backward_batch_size: int = 0,
    split_state_vjp: bool = True,
) -> torch.Tensor:
    """Exact AR with optional checkpointing and packed-document resets.

    Supports unpadded/right-padded sequences. Packed checkpoints offload one
    KV endpoint and preserve full BPTT through explicit state/reset inputs.
    """
    if attention_mask is not None:
        if attention_mask.shape != x.shape[:2]:
            raise ValueError(
                "attention_mask must match x's (B,T) dimensions, got "
                f"{tuple(attention_mask.shape)} vs {tuple(x.shape[:2])}"
            )
        valid = attention_mask.to(dtype=torch.bool)
        # Real tokens followed by right padding are causally harmless to the
        # real prefix.  Left/interior padding would require a second logical
        # reset convention and is rejected explicitly.
        if bool(((~valid[:, :-1]) & valid[:, 1:]).any()):
            raise NotImplementedError("forward_autoregressive supports only unpadded or right-padded inputs")

    checkpoint_chunk_size = int(checkpoint_chunk_size)
    if checkpoint_chunk_size < 0:
        raise ValueError(f"checkpoint_chunk_size must be non-negative, got {checkpoint_chunk_size}")
    if document_ids is None:
        if not checkpoint_chunk_size or not torch.is_grad_enabled():
            return self.forward_recurrent(x)[0]
        document_ids = torch.zeros(x.shape[:2], device=x.device, dtype=torch.long)

    B, T = x.shape[:2]
    device = x.device
    document_ids = document_ids.to(device=device)
    is_start = document_start_mask(document_ids)
    token_index = torch.arange(T, device=device).view(1, T).expand(B, T)
    start_markers = torch.where(is_start, token_index, torch.zeros_like(token_index))
    # A document starting at token s reads its dummy token from slot s: the preceding
    # token appends a dummy token instead of its own K/V when reset_after is true.
    valid_start = start_markers.cummax(dim=1).values
    q_pos = token_index - valid_start + int(self.use_dummy_token)
    reset_after = torch.zeros((B, T), device=device, dtype=torch.bool)
    reset_after[:, :-1] = is_start[:, 1:]

    keys, values = self._initial_kv(B)
    K_state, V_state = keys.transpose(0, 1), values.transpose(0, 1)
    # Reset channels are explicit checkpoint inputs.  Clone them so checkpoint
    # boundaries do not receive aliased state/reset arguments while both
    # gradient paths still accumulate into ``dummy_token``.
    K_dummy = K_state.clone()
    V_dummy = V_state.clone()
    if checkpoint_chunk_size > 0 and torch.is_grad_enabled():
        parameters = tuple(parameter for parameter in self.parameters() if parameter.requires_grad)
        return _offloaded_checkpoint(
            self,
            checkpoint_chunk_size,
            backward_batch_size,
            split_state_vjp,
            x,
            K_state,
            V_state,
            K_dummy,
            V_dummy,
            q_pos,
            valid_start,
            reset_after,
            *parameters,
        )

    return _forward_packed_chunk(self, x, K_state, V_state, K_dummy, V_dummy, q_pos, valid_start, reset_after)[0]


@eager_loop
def _forward_packed_chunk(
    self: WhiteMatterBlock,
    x_chunk: torch.Tensor,
    K_prefix: torch.Tensor,
    V_prefix: torch.Tensor,
    K_dummy: torch.Tensor,
    V_dummy: torch.Tensor,
    q_pos_chunk: torch.Tensor,
    valid_start_chunk: torch.Tensor,
    reset_after_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run explicit packed state; reuse fixed-capacity storage when no tape is needed."""
    inplace = not torch.is_grad_enabled()
    initial_length = K_prefix.shape[3]
    if inplace:
        shape = (*K_prefix.shape[:3], initial_length + x_chunk.shape[1], K_prefix.shape[-1])
        keys, values = K_prefix.new_empty(shape), V_prefix.new_empty(shape)
        keys[..., :initial_length, :].copy_(K_prefix)
        values[..., :initial_length, :].copy_(V_prefix)
        output = torch.empty_like(x_chunk)
    else:
        outs: list[torch.Tensor] = []
    for local_index in range(x_chunk.shape[1]):
        N = initial_length + local_index
        if inplace:
            K_prefix, V_prefix = keys[..., :N, :], values[..., :N, :]
        slots = torch.arange(N, device=x_chunk.device).view(1, N)
        # Storage grows across documents; this mask resets the visible history.
        valid_mask = slots >= valid_start_chunk[:, local_index : local_index + 1]
        out, K_append, V_append = self._packed_autoregressive_step(
            x_chunk[:, local_index : local_index + 1],
            K_prefix,
            V_prefix,
            K_dummy,
            V_dummy,
            q_pos_chunk[:, local_index],
            valid_mask,
            reset_after_chunk[:, local_index],
        )
        if inplace:
            keys[..., N : N + 1, :].copy_(K_append)
            values[..., N : N + 1, :].copy_(V_append)
            output[:, local_index : local_index + 1].copy_(out)
        else:
            # Backward readers retain their prefixes; ordinary saved views cannot
            # share mutable storage without invalidating autograd versions.
            K_prefix = torch.cat((K_prefix, K_append), dim=3)
            V_prefix = torch.cat((V_prefix, V_append), dim=3)
            outs.append(out)
    return (output, keys, values) if inplace else (torch.cat(outs, dim=1), K_prefix, V_prefix)

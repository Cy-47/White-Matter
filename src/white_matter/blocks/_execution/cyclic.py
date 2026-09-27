"""Cyclic feedback execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from white_matter._typing import dynamo_disable, dynamo_disable_nonrecursive
from white_matter.modules.checkpointing import checkpoint_pointwise
from white_matter.modules.precision import model_autocast_context
from white_matter.ops import CyclicAttentionMetadata

from ..decoder_layer import run_feedback_layers
from . import resolve_passes

if TYPE_CHECKING:
    from white_matter.blocks.white_matter import WhiteMatterBlock


@dynamo_disable
def _prepare_cyclic_groups(
    self: WhiteMatterBlock,
    T: int,
    cyclic_groups: int,
    q_pos_emb: tuple[torch.Tensor, torch.Tensor],
    k_pos_emb: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    *,
    cache_rope: bool = True,
) -> tuple[
    list[torch.Tensor],
    list[tuple[torch.Tensor, torch.Tensor]],
    list[tuple[torch.Tensor, torch.Tensor]],
    torch.Tensor,
]:
    """Cache the static schedule; reuse RoPE slices only for fixed positions."""
    schedule_key = (T, cyclic_groups, str(device))
    schedule = self._cyclic_schedule_cache.get(schedule_key)
    if schedule is None:
        # Groups are strided: with two groups, visit [0,2,4,...], then [1,3,5,...].
        query_groups = [torch.arange(c, T, cyclic_groups, device=device) for c in range(cyclic_groups)]
        restore_order = torch.argsort(torch.cat(query_groups))
        schedule = (query_groups, restore_order)
        self._cyclic_schedule_cache[schedule_key] = schedule
    query_groups, restore_order = schedule

    rope_key = (*schedule_key, q_pos_emb[0].dtype)
    # Document-reset positions depend on the batch, so their RoPE slices cannot be reused.
    cached_rope = self._cyclic_rope_cache.get(rope_key) if cache_rope else None
    if cached_rope is None:
        q_cos, q_sin = q_pos_emb
        k_cos, k_sin = k_pos_emb
        query_group_rope = [(q_cos[:, p], q_sin[:, p]) for p in query_groups]
        key_slots = [torch.cat((p.new_zeros(1), p + 1)) for p in query_groups]
        key_group_rope = [(k_cos.index_select(1, p), k_sin.index_select(1, p)) for p in key_slots]
        cached_rope = (query_group_rope, key_group_rope)
        if cache_rope:
            self._cyclic_rope_cache[rope_key] = cached_rope
    return query_groups, *cached_rope, restore_order


def _compact_channel_major(
    K: torch.Tensor,
    V: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B,k,H,T+1,d)`` -> contiguous consumed ``(k,B,H,T,d)``."""
    return (
        K[..., :-1, :].permute(1, 0, 2, 3, 4).contiguous(),
        V[..., :-1, :].permute(1, 0, 2, 3, 4).contiguous(),
    )


def _rebuild_state(
    base: torch.Tensor,
    positions: tuple[torch.Tensor, ...],
    updates: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    for slots, update in zip(positions, updates, strict=True):
        base = base.index_copy(3, slots, update)
    return base


def _training_group(
    block: WhiteMatterBlock,
    x: torch.Tensor,
    base_keys: torch.Tensor,
    base_values: torch.Tensor,
    update_positions: tuple[torch.Tensor, ...],
    key_updates: tuple[torch.Tensor, ...],
    value_updates: tuple[torch.Tensor, ...],
    dummy_token: torch.Tensor,
    positions: torch.Tensor,
    query_rope: tuple[torch.Tensor, torch.Tensor],
    key_rope: tuple[torch.Tensor, torch.Tensor],
    metadata: CyclicAttentionMetadata | None,
    groups: int,
    offset: int,
    return_output: bool,
    terminal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct one group's readers from the compact update history."""
    keys = _rebuild_state(base_keys, update_positions, key_updates)
    values = _rebuild_state(base_values, update_positions, value_updates)
    hidden, layer_inputs = _group_layers(
        block,
        x,
        keys.unbind(0),
        values.unbind(0),
        positions,
        query_rope,
        groups,
        offset,
        return_output,
        metadata,
    )
    keys, values = block.kv_pool.project_sequence(
        torch.stack(layer_inputs, dim=2),
        key_rope,
        dummy_token=dummy_token,
    )
    keys, values = keys[..., 1:, :].transpose(0, 1), values[..., 1:, :].transpose(0, 1)
    if terminal:
        keys, values = keys[..., :-1, :], values[..., :-1, :]
    return hidden if hidden is not None else x.new_empty(0), keys, values


def _checkpoint_training_group(*args: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return checkpoint_pointwise(_training_group, *args)


_compiled_training_group = torch.compile(
    _checkpoint_training_group,
    fullgraph=False,
    dynamic=False,
    options={"emulate_precision_casts": True},
)


@dynamo_disable_nonrecursive
def _run_training_group(*args: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _compiled_training_group(*args)


def forward_cyclic(
    self: WhiteMatterBlock,
    x: torch.Tensor,
    *,
    num_passes: int | None = None,
    cyclic_groups: int = 8,
    num_gradient_passes: int | None = None,
    attention_mask: torch.Tensor | None = None,
    document_ids: torch.Tensor | None = None,
    output_final_state: bool = False,
    kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    on_pass: Callable[[int, torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Run cyclic passes over strided token groups, refreshing KV after each group.

    Detached passes run without gradients. Before differentiated passes, rebuild
    the pool from the last detached hidden states (or the input embeddings) so
    pool parameters still receive gradients. Document metadata is explicit and
    RoPE resets per document. Padding is stably moved to the end and restored
    before returning. Short sequences are padded to fill the cyclic groups.
    """
    if on_pass is not None and (torch.is_grad_enabled() or attention_mask is not None):
        raise ValueError("pass observation requires unpadded inference")
    n_iter, num_gradient_passes = resolve_passes(self.num_passes, num_passes, num_gradient_passes)
    if type(cyclic_groups) is not int or cyclic_groups < 1:
        raise ValueError("cyclic_groups must be a positive integer")
    if x.ndim != 3 or x.shape[1] < 1:
        raise ValueError("inputs must have a nonempty batch/sequence/hidden layout")
    if kv_cache is not None and (
        torch.is_grad_enabled()
        or attention_mask is not None
        or document_ids is not None
        or x.shape[1] <= cyclic_groups
        or num_gradient_passes
    ):
        raise ValueError(
            "bounded cyclic prefill requires no gradients, unpadded single-document inputs longer than the group count"
        )
    # One inference schedule, whether storage is temporary or caller-owned.
    bounded = kv_cache is not None or (
        x.is_cuda
        and torch.is_autocast_enabled("cuda")
        and not self.training
        and not torch.is_grad_enabled()
        and not num_gradient_passes
    )
    if bounded and torch.compiler.is_compiling():
        return inference_forward(
            self,
            x,
            num_passes=n_iter,
            num_gradient_passes=num_gradient_passes,
            cyclic_groups=cyclic_groups,
            attention_mask=attention_mask,
            document_ids=document_ids,
            output_final_state=output_final_state,
            kv_cache=kv_cache,
            on_pass=on_pass,
        )
    n_passes_no_grad = n_iter - num_gradient_passes
    length = x.shape[1]
    if length <= cyclic_groups:
        # Fill every group while keeping artificial tokens after the real prefix.
        padding = 2 * cyclic_groups - length
        x = F.pad(x, (0, 0, 0, padding))
        if attention_mask is None:
            attention_mask = x.new_ones((x.shape[0], length), dtype=torch.bool)
        attention_mask = F.pad(attention_mask, (0, padding))
        if document_ids is not None:
            document_ids = torch.cat((document_ids, (document_ids[:, -1:] + 1).expand(-1, padding)), dim=1)

    # Stably move real tokens before padding, then restore their original slots.
    restore_perm = None
    if attention_mask is not None:
        pad_first = (~attention_mask.bool()).to(device=x.device, dtype=torch.uint8)
        order = torch.argsort(pad_first, dim=1, stable=True)  # real slots lead
        restore_perm = torch.empty_like(order)
        restore_perm.scatter_(1, order, torch.arange(order.shape[1], device=x.device).expand_as(order))
        x = torch.gather(x, 1, order.unsqueeze(-1).expand_as(x))
        if document_ids is not None:
            document_ids = torch.gather(document_ids, 1, order)

    T = x.shape[1]
    if bounded and kv_cache is None:
        shape = (x.shape[0], self.num_kv_channels, self.kv_pool.num_key_value_heads, T + 1, self.kv_pool.head_dim)
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        keys = torch.empty(shape, device=x.device, dtype=dtype)
        kv_cache = keys, torch.empty_like(keys)
    q_pos_emb, k_pos_emb = self._prepare_rope(x, document_ids)
    # The explicit batched input preserves backward numerics across the compiled pass boundary.
    dummy_token = self.dummy_token.view(1, 1, -1).expand(x.shape[0], 1, -1)
    L = len(self.layers)
    pass_inputs = _prepare_cyclic_groups(
        self, T, cyclic_groups, q_pos_emb, k_pos_emb, x.device, cache_rope=(document_ids is None)
    )
    metadata = None
    if document_ids is not None:
        from .metadata import prepare_feedback_metadata

        metadata = prepare_feedback_metadata(document_ids, pass_inputs[0], T)

    # Cached prefill compiles its bounded operations, not the whole training pass.
    execute_pass = run_pass.__get__(self) if kv_cache is not None else self.cyclic_pass
    layer_hidden_states = None
    for detached, count in ((True, n_passes_no_grad), (False, num_gradient_passes)):
        if not count:
            continue
        with (
            torch.no_grad() if detached else nullcontext(),
            model_autocast_context(x.device) if detached else nullcontext(),
        ):
            if kv_cache is not None:
                K, V = (tensor.detach() for tensor in kv_cache)
                initial = torch.cat((dummy_token.to(x.dtype), x), dim=1)
                for start in range(0, T + 1, 64):
                    stop = min(start + 64, T + 1)
                    slots = torch.arange(start, stop, device=x.device)
                    rope = tuple(t[:, start:stop] for t in k_pos_emb)
                    _write_pool(self, [initial[:, start:stop]] * L, K, V, slots, (rope[0], rope[1]))
                del initial
            elif detached:
                # Detached initialization can broadcast embeddings across layers.
                source = x.unsqueeze(2).expand(-1, -1, L, -1)
            else:
                # Re-project detached layer inputs to reconnect pool parameters.
                # Preserve the differentiated path's contiguous projection layout.
                if not n_passes_no_grad:
                    layer_hidden_states = x.unsqueeze(1).expand(-1, L, -1, -1).contiguous()
                assert layer_hidden_states is not None
                source = layer_hidden_states.transpose(1, 2).contiguous()
            if kv_cache is None:
                K, V = self.kv_pool.project_sequence(source, k_pos_emb, dummy_token=dummy_token)
                if 1 < self.num_kv_channels < self.num_layers:
                    K, V = _compact_channel_major(K, V)
                del source
            for p in range(count):
                last = p == count - 1
                h, K, V, layer_hidden_states = execute_pass(
                    x,
                    K,
                    V,
                    dummy_token,
                    *pass_inputs,
                    metadata=metadata,
                    return_hidden_states=detached and last and num_gradient_passes > 0,
                    consume_state=detached,
                    pool_chunk_size=64 if kv_cache is not None else 0,
                    return_output=on_pass is not None or (last and (not detached or not num_gradient_passes)),
                    output_final_state=output_final_state and last and (not detached or not num_gradient_passes),
                )
                if on_pass is not None:
                    assert h is not None
                    on_pass(p + 1, h[:, :length])

    assert h is not None
    if restore_perm is not None:
        h = torch.gather(h, 1, restore_perm.unsqueeze(-1).expand_as(h))
    if length < T:
        h = h[:, :length]
    if not output_final_state:
        return h, None
    if restore_perm is not None:
        # State must follow the caller's token order, just like the hidden outputs.
        indices = restore_perm[:, None, None, :, None].expand_as(K[..., 1:, :])
        K = torch.cat((K[..., :1, :], K[..., 1:, :].gather(3, indices)), dim=3)
        V = torch.cat((V[..., :1, :], V[..., 1:, :].gather(3, indices)), dim=3)
    if length < T or kv_cache is not None:
        K, V = K[..., : length + 1, :], V[..., : length + 1, :]
    return h, (K, V)


def run_pass(
    self: WhiteMatterBlock,
    x_in: torch.Tensor,
    K_initial: torch.Tensor,
    V_initial: torch.Tensor,
    dummy_token: torch.Tensor,
    query_groups: list[torch.Tensor],
    query_group_rope: list[tuple[torch.Tensor, torch.Tensor]],
    key_group_rope: list[tuple[torch.Tensor, torch.Tensor]],
    restore_order: torch.Tensor,
    metadata: tuple[CyclicAttentionMetadata, ...] | None = None,
    return_hidden_states: bool = False,
    consume_state: bool = False,
    output_final_state: bool = False,
    pool_chunk_size: int = 0,
    return_output: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Shared group order with functional training or bounded cache publication.

    Training keeps its whole-pass compilation and dummy-row projection shape.
    Cached inference compiles only layer sweeps and bounded pool writes;
    pool_chunk_size and return_output configure that bounded branch.
    """
    _, T, _ = x_in.shape
    if K_initial.ndim != 5 or K_initial.shape != V_initial.shape:
        raise ValueError("cyclic state requires matching five-dimensional K/V tensors")
    # Compact states are channel-major without the final unconsumed key;
    # direct callers may still supply the public T+1 layout.
    compact_fixed = 1 < self.num_kv_channels < self.num_layers and K_initial.shape[-2] == T
    inplace = not torch.is_grad_enabled()
    K = K_initial.clone() if inplace and not consume_state else K_initial
    V = V_initial.clone() if inplace and not consume_state else V_initial
    final_kv = None
    if pool_chunk_size:
        assert inplace
        assert not return_hidden_states
        assert not compact_fixed

    if compact_fixed and not inplace and not return_hidden_states and not output_final_state:
        outputs = []
        update_positions: list[torch.Tensor] = []
        key_updates: list[torch.Tensor] = []
        value_updates: list[torch.Tensor] = []
        terminal_group = (T - 1) % len(query_groups)
        for offset, positions in enumerate(query_groups):
            terminal = offset == terminal_group
            hidden, keys, values = _run_training_group(
                self,
                x_in,
                K_initial,
                V_initial,
                tuple(update_positions),
                tuple(key_updates),
                tuple(value_updates),
                dummy_token,
                positions,
                query_group_rope[offset],
                key_group_rope[offset],
                None if metadata is None else metadata[offset],
                len(query_groups),
                offset,
                return_output,
                terminal,
            )
            if return_output:
                outputs.append(hidden)
            update_positions.append((positions + 1)[:-1] if terminal else positions + 1)
            key_updates.append(keys)
            value_updates.append(values)
        K = _rebuild_state(K_initial, tuple(update_positions), tuple(key_updates))
        V = _rebuild_state(V_initial, tuple(update_positions), tuple(value_updates))
        output = torch.cat(outputs, dim=1).index_select(1, restore_order) if return_output else None
        return output, K, V, None

    outputs_per_chunk: list[torch.Tensor] = []
    output = torch.empty_like(x_in) if pool_chunk_size and return_output else None
    layer_inputs_per_chunk: list[torch.Tensor] = []
    execute_layers = _inference_layers if pool_chunk_size else _group_layers
    for c, chunk_positions in enumerate(query_groups):
        if compact_fixed:
            channel_keys, channel_values = K.unbind(0), V.unbind(0)
        elif pool_chunk_size:
            channel_keys, channel_values = K[..., :T, :].unbind(1), V[..., :T, :].unbind(1)
        else:
            channel_keys = tuple(tensor[..., :-1, :].contiguous() for tensor in K.unbind(1))
            channel_values = tuple(tensor[..., :-1, :].contiguous() for tensor in V.unbind(1))
        h, chunk_layer_inputs = execute_layers(
            self,
            x_in,
            channel_keys,
            channel_values,
            chunk_positions,
            query_group_rope[c],
            len(query_groups),
            c,
            return_output,
            None if metadata is None else metadata[c],
        )
        if h is not None:
            if output is not None:
                output.index_copy_(1, chunk_positions, h)
            else:
                outputs_per_chunk.append(h)
        del h
        if pool_chunk_size:
            for start in range(0, chunk_positions.numel(), pool_chunk_size):
                stop = min(start + pool_chunk_size, chunk_positions.numel())
                rope = tuple(t[:, start + 1 : stop + 1] for t in key_group_rope[c])
                _write_pool(
                    self,
                    [state[:, start:stop] for state in chunk_layer_inputs],
                    K,
                    V,
                    chunk_positions[start:stop] + 1,
                    (rope[0], rope[1]),
                )
            del chunk_layer_inputs
            continue
        chunk_layer_states = torch.stack(chunk_layer_inputs, dim=2)
        del chunk_layer_inputs
        if return_hidden_states:
            layer_inputs_per_chunk.append(chunk_layer_states)
        # Keep the dummy row during projection: changing GEMM shape changes BF16 gradients.
        K_src, V_src = self.kv_pool.project_sequence(chunk_layer_states, key_group_rope[c], dummy_token=dummy_token)
        del chunk_layer_states
        K_src, V_src = K_src[..., 1:, :], V_src[..., 1:, :]
        # Publish after the layer sweep: later groups read these refreshed K/V.
        positions = chunk_positions + 1
        if compact_fixed:
            K_src, V_src = K_src.transpose(0, 1), V_src.transpose(0, 1)
            # Compact storage omits the last unread token; decoding needs it.
            if c == (T - 1) % len(query_groups):
                if output_final_state:
                    final_kv = K_src[..., -1:, :], V_src[..., -1:, :]
                positions = positions[:-1]
                K_src, V_src = K_src[..., :-1, :], V_src[..., :-1, :]
        if inplace:
            K.index_copy_(3, positions, K_src)
            V.index_copy_(3, positions, V_src)
        else:
            # Earlier attention calls retain their K/V for backward.
            K, V = K.index_copy(3, positions, K_src), V.index_copy(3, positions, V_src)

    out = (
        output
        if pool_chunk_size or not return_output
        else torch.cat(outputs_per_chunk, dim=1).index_select(1, restore_order)
    )
    if output_final_state and compact_fixed:
        # Export all tokens in batch-major order, including the unread final token.
        assert final_kv is not None
        K = torch.cat((K, final_kv[0]), dim=3).transpose(0, 1)
        V = torch.cat((V, final_kv[1]), dim=3).transpose(0, 1)
    hs = None
    if return_hidden_states:
        # Restore (B,L,T,D) layer inputs for the gradient transition.
        hs = torch.cat(layer_inputs_per_chunk, dim=1)
        hs = hs.permute(0, 2, 1, 3).contiguous().index_select(2, restore_order)
    return out, K, V, hs


# Static prefill keeps host scheduling outside the enclosing model graph.
inference_forward = dynamo_disable(forward_cyclic)


@torch.compile(fullgraph=True, dynamic=True, options={"emulate_precision_casts": True, "max_autotune": True})
def _write_pool(
    block: WhiteMatterBlock,
    states: list[torch.Tensor],
    keys: torch.Tensor,
    values: torch.Tensor,
    slots: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor],
) -> None:
    key, value = block.kv_pool.project_sequence(torch.stack(states, dim=2), rope)
    keys.index_copy_(3, slots, key)
    values.index_copy_(3, slots, value)


def _group_layers(
    block: WhiteMatterBlock,
    x: torch.Tensor,
    keys: Sequence[torch.Tensor],
    values: Sequence[torch.Tensor],
    positions: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor],
    groups: int,
    offset: int,
    return_output: bool,
    metadata: CyclicAttentionMetadata | None = None,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    # Specialize batches while keeping ragged query lengths symbolic.
    if not block.training and not torch.is_grad_enabled():
        torch._dynamo.mark_static(x, 0)
    hidden, states = run_feedback_layers(
        block.layers,
        x.index_select(1, positions),
        keys,
        values,
        rope,
        return_output=return_output,
        query_stride=groups,
        query_offset=offset,
        metadata=metadata,
    )
    # The final layer has no pool output, so non-output passes stop at its input.
    return hidden if return_output else None, states


_inference_layers = torch.compile(
    _group_layers, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True}
)

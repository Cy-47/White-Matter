"""Strict-past attention for parallel prefill and committed KV caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from white_matter._typing import compiler_disable

from .flash_attention import flash_attention_decode


@dataclass(frozen=True)
class StrictCausalMetadata:
    """Reusable varlen gather/scatter schedule, independent of heads and values."""

    query_indices: torch.Tensor
    key_indices: torch.Tensor
    cu_queries: torch.Tensor
    cu_keys: torch.Tensor
    max_queries: int
    max_keys: int
    key_extent: int

    def with_key_extent(self, key_extent: int) -> StrictCausalMetadata:
        """Address the same key slots in storage with a different row width."""
        if key_extent == self.key_extent:
            return self
        rows = self.key_indices // self.key_extent
        slots = self.key_indices % self.key_extent
        return StrictCausalMetadata(
            self.query_indices,
            rows * key_extent + slots,
            self.cu_queries,
            self.cu_keys,
            self.max_queries,
            self.max_keys,
            key_extent,
        )


@compiler_disable
def prepare_strict_causal_metadata(
    document_ids: torch.Tensor,
    query_length: int,
    *,
    query_start: int | torch.Tensor = 0,
    kv_lengths: torch.Tensor | None = None,
    use_dummy_token: bool = False,
) -> StrictCausalMetadata:
    """Pack document runs, ignoring negative document IDs as padding.

    document_ids labels real key slots (including current query token slots),
    excluding the optional leading dummy. Query positions start at query_start.
    Build once and reuse across layers and Jacobi iterations.
    """
    batch, extent = document_ids.shape
    device = document_ids.device
    start = torch.as_tensor(query_start, device=device, dtype=torch.long).expand(batch)
    positions = start[:, None] + torch.arange(query_length, device=device)
    if bool(((positions < 0) | (positions >= extent)).any()):
        raise ValueError("document_ids must label every query position")
    slots = torch.arange(extent, device=device).expand(batch, extent)
    present = document_ids >= 0
    counts = torch.cat((slots.new_zeros(batch, 1), present.cumsum(1)), dim=1)
    previous = torch.cat((slots.new_full((batch, 1), -1), torch.where(present, slots, -1)), dim=1)[:, :extent]
    previous = previous.cummax(1).values
    boundary = present & ((previous < 0) | (document_ids != document_ids.gather(1, previous.clamp_min(0))))
    run_start = torch.where(boundary, counts[:, :-1], 0).cummax(1).values
    # Globally unique run IDs prevent repeated labels from reconnecting documents.
    run = boundary.flatten().cumsum(0).view(batch, extent)
    query_runs = run.gather(1, positions)
    first = run_start.gather(1, positions)
    visible_end = positions if kv_lengths is None else torch.minimum(positions, kv_lengths[:, None])
    visible_end = counts.gather(1, visible_end.clamp(0, extent))
    valid = (document_ids.gather(1, positions) >= 0) & ((visible_end > first) | use_dummy_token)
    query_indices = valid.flatten().nonzero().flatten()
    selected_runs = query_runs.flatten()[query_indices]
    boundaries = torch.ones_like(selected_runs, dtype=torch.bool)
    selected_ends = visible_end.flatten()[query_indices]
    boundaries[1:] = (selected_runs[1:] != selected_runs[:-1]) | (selected_ends[1:] != selected_ends[:-1] + 1)
    q_starts = boundaries.nonzero().flatten()
    cu_queries = torch.cat((q_starts, q_starts.new_full((1,), query_indices.numel())))
    if query_indices.numel() == 0:
        empty = query_indices
        return StrictCausalMetadata(
            empty,
            empty,
            cu_queries.int(),
            cu_queries.int(),
            query_length,
            extent + int(use_dummy_token),
            extent + int(use_dummy_token),
        )
    last_queries = query_indices[cu_queries[1:] - 1]
    first_keys = first.flatten()[last_queries]
    ends = visible_end.flatten()[last_queries]
    lengths = (ends - first_keys).clamp_min(0) + int(use_dummy_token)
    cu_keys = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    groups = torch.repeat_interleave(torch.arange(lengths.numel(), device=device), lengths)
    local = torch.arange(groups.numel(), device=device) - cu_keys[groups]
    rows = last_queries // query_length
    offset = int(use_dummy_token)
    ranks = first_keys[groups] + local - offset
    row_offsets = counts[:, -1].cumsum(0) - counts[:, -1]
    real_slots = present.flatten().nonzero().flatten() % extent
    key_slots = real_slots[(row_offsets[rows[groups]] + ranks).clamp_min(0)]
    key_slots = torch.where(local == 0, -1, key_slots) if use_dummy_token else key_slots
    key_indices = rows[groups] * (extent + offset) + key_slots + offset
    return StrictCausalMetadata(
        query_indices, key_indices, cu_queries.int(), cu_keys.int(), query_length, extent + offset, extent + offset
    )


def strict_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_start: int | torch.Tensor = 0,
    kv_lengths: torch.Tensor | None = None,
    use_dummy_token: bool = False,
    metadata: StrictCausalMetadata | None = None,
    key_mask: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    backend: Literal["auto", "reference", "flash_attention_2"] = "auto",
    num_splits: int = 0,
) -> torch.Tensor:
    """Attend only to earlier real tokens, plus an optional leading dummy.

    Q/K/V use (B,H,T,D); output uses (B,T,H,D). Inputs already contain RoPE.
    query_start and kv_lengths count real tokens, excluding the dummy slot.
    K/V may contain current tokens or unused cache capacity; neither is visible
    outside the strict-past bound. Inputs are never mutated. Empty rows return
    zero. metadata is prepared from document IDs once outside layer/pass loops.
    key_mask, when supplied, is boolean (B,N) and forces the reference backend.
    """
    if backend not in {"auto", "reference", "flash_attention_2"}:
        raise ValueError(f"unsupported strict causal backend: {backend!r}")
    if type(use_dummy_token) is not bool:
        raise ValueError("use_dummy_token must be boolean")
    if any(t.ndim != 4 for t in (query, key, value)) or key.shape != value.shape:
        raise ValueError("Q/K/V require (B,H,T,D) tensors with matching K/V")
    batch, heads, length, dim = query.shape
    if key.shape[0] != batch or key.shape[-1] != dim or heads % key.shape[1]:
        raise ValueError("Q/K/V require matching batch/head_dim and divisible heads")
    if any(t.device != query.device for t in (key, value)):
        raise ValueError("Q/K/V must share a device")
    offset = int(use_dummy_token)
    capacity = key.shape[-2] - offset
    if capacity < 0:
        raise ValueError("dummy-enabled K/V must contain the leading dummy slot")
    if kv_lengths is not None and (kv_lengths.shape != (batch,) or kv_lengths.dtype not in (torch.int32, torch.int64)):
        raise ValueError("kv_lengths must be integer (B,)")
    if isinstance(query_start, torch.Tensor) and (
        query_start.shape != (batch,) or query_start.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("query_start must be an integer or integer (B,) tensor")
    if metadata is not None and key_mask is not None:
        raise ValueError("metadata and key_mask are mutually exclusive")
    if key_mask is not None and (key_mask.shape != (batch, key.shape[-2]) or key_mask.dtype != torch.bool):
        raise ValueError("key_mask must be boolean (B,N)")
    if metadata is not None:
        # Document labels may include unstored query slots or omit unused capacity.
        metadata = metadata.with_key_extent(key.shape[-2])
    scale = dim**-0.5 if softmax_scale is None else softmax_scale
    # Empty slices retain zero gradients without reading uninitialized cache slots.
    if not length or not key.shape[-2]:
        zero = (query[..., :0, :].sum() + key[..., :0, :].sum() + value[..., :0, :].sum()).to(query.dtype)
        return query.new_zeros(batch, length, heads, dim) + zero
    use_flash = backend != "reference" and query.is_cuda and key_mask is None
    if backend == "auto" and use_flash:
        from importlib.util import find_spec

        use_flash = dim <= 256 and find_spec("flash_attn") is not None
    if (
        use_flash
        and metadata is None
        and length == 1
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)))
    ):
        # Keep decode graph-visible; only the pybind call is a registered custom op.
        dtype = query.dtype if query.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        q, k, v = (t.transpose(1, 2).to(dtype) for t in (query, key, value))
        lengths = torch.as_tensor(query_start, device=query.device).expand(batch)
        if kv_lengths is not None:
            lengths = torch.minimum(lengths, kv_lengths)
        lengths = lengths.clamp(0, capacity).to(torch.int32) + offset
        output = flash_attention_decode(q, k, v, lengths, scale, False, num_splits)
        return output.masked_fill((lengths == 0)[:, None, None, None], 0).to(query.dtype)
    if use_flash:
        return _flash_attention(query, key, value, query_start, kv_lengths, use_dummy_token, metadata, scale)
    if backend == "flash_attention_2" and not query.is_cuda:
        raise ValueError("FlashAttention requires CUDA")
    if metadata is not None:
        # Varlen schedule defines independent groups and bottom-right causal bounds.
        q = query.transpose(1, 2).flatten(0, 1)[metadata.query_indices]
        k = key.transpose(1, 2).flatten(0, 1)[metadata.key_indices]
        v = value.transpose(1, 2).flatten(0, 1)[metadata.key_indices]
        zero = (query[..., :0, :].sum() + key[..., :0, :].sum() + value[..., :0, :].sum()).to(query.dtype)
        output = query.new_zeros(batch * length, heads, dim) + zero
        for group in range(metadata.cu_queries.numel() - 1):
            qs, qe = metadata.cu_queries[group : group + 2].tolist()
            ks, ke = metadata.cu_keys[group : group + 2].tolist()
            keep = torch.arange(ke - ks, device=query.device)[None, :] <= (
                ke - ks - (qe - qs) + torch.arange(qe - qs, device=query.device)[:, None]
            )
            result = _sdpa(
                q[qs:qe].transpose(0, 1)[None],
                k[ks:ke].transpose(0, 1)[None],
                v[ks:ke].transpose(0, 1)[None],
                keep,
                scale,
            )
            output = output.index_copy(0, metadata.query_indices[qs:qe], result[0])
        return output.view(batch, length, heads, dim)
    starts = torch.as_tensor(query_start, device=query.device).expand(batch)
    qpos = starts[:, None] + torch.arange(length, device=query.device)
    slots = torch.arange(key.shape[-2], device=query.device) - offset
    keep = slots[None, None, :] < qpos[:, :, None]
    if kv_lengths is not None:
        keep = keep & (slots[None, None, :] < kv_lengths[:, None, None])
    if use_dummy_token:
        keep[..., 0] = True
    if key_mask is not None:
        keep = keep & key_mask[:, None, :]
    # Clear unused storage: even a masked NaN in K/V can contaminate GEMMs.
    visible = keep.any(1)[:, None, :, None]
    return _sdpa(query, key.masked_fill(~visible, 0), value.masked_fill(~visible, 0), keep[:, None], scale)


def _sdpa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, keep: torch.Tensor, scale: float
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query,
        key.to(query.dtype),
        value.to(query.dtype),
        attn_mask=keep,
        dropout_p=0.0,
        scale=scale,
        enable_gqa=True,
    ).transpose(1, 2)


@compiler_disable
def _flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start: int | torch.Tensor,
    kv_lengths: torch.Tensor | None,
    use_dummy_token: bool,
    metadata: StrictCausalMetadata | None,
    scale: float,
) -> torch.Tensor:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    batch, heads, length, dim = query.shape
    dtype = query.dtype if query.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    q, k, v = (t.transpose(1, 2).to(dtype) for t in (query, key, value))
    offset = int(use_dummy_token)
    if metadata is not None:
        output = q.new_zeros(batch * length, heads, dim)
        if metadata.query_indices.numel():
            selected = flash_attn_varlen_func(
                q.flatten(0, 1)[metadata.query_indices],
                k.flatten(0, 1)[metadata.key_indices],
                v.flatten(0, 1)[metadata.key_indices],
                metadata.cu_queries,
                metadata.cu_keys,
                metadata.max_queries,
                metadata.max_keys,
                softmax_scale=scale,
                causal=True,
            )
            output = output.index_copy(0, metadata.query_indices, selected)
        else:
            output = output + (q[:, :0].sum() + k[:, :0].sum() + v[:, :0].sum())
        return output.view(batch, length, heads, dim).to(query.dtype)
    if isinstance(query_start, int) and kv_lengths is None:
        end = query_start + length - 1 + offset
        if end <= key.shape[-2]:
            skip = int(query_start == 0 and not use_dummy_token)
            if length == skip:
                return (q * 0 + (k[:, :0].sum() + v[:, :0].sum())).to(query.dtype)
            result = flash_attn_func(q[:, skip:], k[:, :end], v[:, :end], softmax_scale=scale, causal=True)
            return F.pad(result, (0, 0, 0, 0, skip, 0)).to(query.dtype)
    # Ragged prefixes use the same document packing as a single document per row.
    starts = torch.as_tensor(query_start, device=query.device).expand(batch)
    extent = max(key.shape[-2] - offset, int(starts.max()) + length)
    docs = torch.zeros(batch, extent, dtype=torch.long, device=query.device)
    schedule = prepare_strict_causal_metadata(
        docs,
        length,
        query_start=starts,
        kv_lengths=torch.full((batch,), key.shape[-2] - offset, device=query.device)
        if kv_lengths is None
        else kv_lengths,
        use_dummy_token=use_dummy_token,
    )
    # Packing indices use the document extent, which may include unstored queries.
    schedule = schedule.with_key_extent(key.shape[-2])
    return _flash_attention(query, key, value, query_start, kv_lengths, use_dummy_token, schedule, scale)

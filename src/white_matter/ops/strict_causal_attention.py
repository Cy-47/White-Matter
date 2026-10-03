"""Strict-past attention for parallel prefill and committed KV caches."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from white_matter._typing import compiler_assume_constant_result, compiler_disable

from .flash_attention import (
    TorchFlashDense,
    TorchFlashVarlen,
    can_use_torch_flash,
    flash_attention_decode,
    torch_flash_decode,
)


def _query_starts(query_start: int | torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    if isinstance(query_start, torch.Tensor):
        return query_start.to(device=device).expand(batch)
    # torch.as_tensor(SymInt) specializes each growing cache length under Dynamo.
    return torch.full((batch,), query_start, device=device, dtype=torch.long)


@compiler_assume_constant_result
def _standalone_flash_available() -> bool:
    return find_spec("flash_attn") is not None


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
    segments: tuple[tuple[int, int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.segments and self.cu_queries.numel() > 1:
            queries = self.cu_queries.tolist()
            keys = self.cu_keys.tolist()
            object.__setattr__(
                self, "segments", tuple(zip(queries[:-1], queries[1:], keys[:-1], keys[1:], strict=True))
            )

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
            self.segments,
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
        query_indices,
        key_indices,
        cu_queries.int(),
        cu_keys.int(),
        int(cu_queries.diff().max()),
        int(lengths.max()),
        extent + offset,
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
    backend: Literal["auto", "reference", "torch_flash", "flash_attention_2"] = "auto",
    num_splits: int = 0,
    static_cache: bool = False,
) -> torch.Tensor:
    """Attend only to earlier real tokens, plus an optional leading dummy.

    Q/K/V use (B,H,T,D); output uses (B,T,H,D). Inputs already contain RoPE.
    query_start and kv_lengths count real tokens, excluding the dummy slot.
    K/V may contain current tokens or unused cache capacity; neither is visible
    outside the strict-past bound. Inputs are never mutated. Empty rows return
    zero. metadata is prepared from document IDs once outside layer/pass loops.
    auto uses efficient SDPA for differentiable single-query masked/ragged reads.
    Other CUDA reads prefer standalone FlashAttention, otherwise requiring native
    PyTorch FlashAttention. CPU auto uses the reference. torch_flash requires native FA;
    flash_attention_2 explicitly selects the optional external package. num_splits
    applies only to external cached decoding. key_mask is boolean (B,N) and
    supports accelerated single-query reads. static_cache permits lifting decode offsets only
    when the caller guarantees fixed batch size and KV storage capacity.
    """
    if backend not in {"auto", "reference", "torch_flash", "flash_attention_2"}:
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
    masked_training = (
        backend == "auto"
        and query.is_cuda
        and length == 1
        and metadata is None
        and (key_mask is not None or kv_lengths is not None or isinstance(query_start, torch.Tensor))
        and torch.is_grad_enabled()
        and any(t.requires_grad for t in (query, key, value))
    )
    if masked_training:
        # Masked SDPA keeps ragged lengths on-device during forward/backward capture.
        backend = "reference"
    if backend == "auto":
        backend = (
            ("flash_attention_2" if _standalone_flash_available() else "torch_flash") if query.is_cuda else "reference"
        )
    use_flash = backend != "reference"
    if use_flash:
        if not query.is_cuda:
            raise ValueError("FlashAttention requires CUDA")
        if key_mask is not None:
            if length != 1:
                raise ValueError("FlashAttention key_mask requires a single query; use document metadata for prefill")
            # Pack visible committed slots before FA reads the cache. Each row
            # can have a different document boundary or padding pattern.
            starts = _query_starts(query_start, batch, query.device)
            bounds = starts if kv_lengths is None else torch.minimum(starts, kv_lengths)
            slots = torch.arange(key.shape[-2], device=query.device)
            keep = key_mask & (slots[None, :] < bounds[:, None] + offset)
            lengths = keep.sum(-1, dtype=torch.int32)
            order = (~keep).to(torch.uint8).argsort(dim=-1, stable=True)
            indices = order[:, None, :, None].expand_as(key)
            return strict_causal_attention(
                query,
                key.gather(2, indices),
                value.gather(2, indices),
                query_start=lengths,
                kv_lengths=lengths,
                softmax_scale=scale,
                backend=backend,
                num_splits=num_splits,
                static_cache=static_cache,
            )
    if use_flash:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return _accelerated_attention(
                query,
                key,
                value,
                query_start,
                kv_lengths,
                use_dummy_token,
                metadata,
                scale,
                backend,
                num_splits,
                static_cache,
            )
    if metadata is not None:
        # Varlen schedule defines independent groups and bottom-right causal bounds.
        q = query.transpose(1, 2).flatten(0, 1)[metadata.query_indices]
        k = key.transpose(1, 2).flatten(0, 1)[metadata.key_indices]
        v = value.transpose(1, 2).flatten(0, 1)[metadata.key_indices]
        zero = (query[..., :0, :].sum() + key[..., :0, :].sum() + value[..., :0, :].sum()).to(query.dtype)
        output = query.new_zeros(batch * length, heads, dim) + zero
        for qs, qe, ks, ke in metadata.segments:
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
    starts = _query_starts(query_start, batch, query.device)
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
    key, value = key.masked_fill(~visible, 0), value.masked_fill(~visible, 0)
    if masked_training:
        # Fold grouped query heads into the query axis to avoid copying K/V.
        grouped_query = query.reshape(batch, key.shape[1], heads // key.shape[1], dim)
        # Efficient SDPA requires an aligned additive-mask stride.
        bias = torch.zeros_like(keep, dtype=query.dtype).masked_fill(~keep, float("-inf"))
        padded = F.pad(bias, (0, (-keep.shape[-1]) % 8), value=float("-inf"))
        keep = padded[..., : keep.shape[-1]]
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            output = _sdpa(grouped_query, key, value, keep[:, None], scale)
        return output.transpose(1, 2).reshape(batch, 1, heads, dim)
    return _sdpa(query, key, value, keep[:, None], scale)


def _accelerated_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start: int | torch.Tensor,
    kv_lengths: torch.Tensor | None,
    use_dummy_token: bool,
    metadata: StrictCausalMetadata | None,
    scale: float,
    backend: str,
    num_splits: int,
    static_cache: bool,
) -> torch.Tensor:
    if (
        metadata is None
        and query.shape[-2] == 1
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)))
    ):
        # Keep lengths on-device; no document packing is needed for a single query.
        dtype = query.dtype if query.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        q, k, v = (t.transpose(1, 2).to(dtype) for t in (query, key, value))
        lengths = _query_starts(query_start, query.shape[0], query.device)
        if kv_lengths is not None:
            lengths = torch.minimum(lengths, kv_lengths)
        lengths = lengths.clamp(0, key.shape[-2] - int(use_dummy_token)).to(torch.int32) + int(use_dummy_token)
        if backend == "flash_attention_2":
            output = flash_attention_decode(q, k, v, lengths, scale, False, num_splits)
        elif can_use_torch_flash(q, k, v):
            output = torch_flash_decode(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), lengths, scale, static_cache=static_cache
            )
        else:
            raise RuntimeError("PyTorch FlashAttention does not support these inputs or this GPU/build")
        return output.to(query.dtype)
    return _flash_attention(query, key, value, query_start, kv_lengths, use_dummy_token, metadata, scale, backend)


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


def _flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start: int | torch.Tensor,
    kv_lengths: torch.Tensor | None,
    use_dummy_token: bool,
    metadata: StrictCausalMetadata | None,
    scale: float,
    backend: str,
) -> torch.Tensor:
    external = backend == "flash_attention_2"
    if external:
        from flash_attn import flash_attn_func, flash_attn_varlen_func

    batch, heads, length, dim = query.shape
    dtype = query.dtype if query.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    q, k, v = (t.transpose(1, 2).to(dtype) for t in (query, key, value))
    offset = int(use_dummy_token)
    if not external and not can_use_torch_flash(q, k, v):
        raise RuntimeError("PyTorch FlashAttention does not support these inputs or this GPU/build")
    if metadata is not None:
        output = q.new_zeros(batch * length, heads, dim)
        if metadata.query_indices.numel():
            packed_q = q.flatten(0, 1).index_select(0, metadata.query_indices)
            packed_k = k.flatten(0, 1).index_select(0, metadata.key_indices)
            packed_v = v.flatten(0, 1).index_select(0, metadata.key_indices)
            if external:
                selected = flash_attn_varlen_func(
                    packed_q,
                    packed_k,
                    packed_v,
                    metadata.cu_queries,
                    metadata.cu_keys,
                    metadata.max_queries,
                    metadata.max_keys,
                    softmax_scale=scale,
                    causal=True,
                )
            else:
                if dim % 8:
                    packed_q, packed_k, packed_v = (F.pad(t, (0, 8 - dim % 8)) for t in (packed_q, packed_k, packed_v))
                selected = TorchFlashVarlen.apply(
                    packed_q,
                    packed_k,
                    packed_v,
                    metadata.cu_queries,
                    metadata.cu_keys,
                    metadata.max_queries,
                    metadata.max_keys,
                    scale,
                )
            output = output.index_copy(0, metadata.query_indices, cast(torch.Tensor, selected)[..., :dim])
        else:
            output = output + (q[:, :0].sum() + k[:, :0].sum() + v[:, :0].sum())
        return output.view(batch, length, heads, dim).to(query.dtype)
    if isinstance(query_start, int) and kv_lengths is None:
        end = query_start + length - 1 + offset
        if end <= key.shape[-2]:
            skip = int(query_start == 0 and not use_dummy_token)
            if length == skip:
                return (q * 0 + (k[:, :0].sum() + v[:, :0].sum())).to(query.dtype)
            if external:
                result = flash_attn_func(q[:, skip:], k[:, :end], v[:, :end], softmax_scale=scale, causal=True)
            else:
                result = _torch_flash_attention(q[:, skip:], k[:, :end], v[:, :end], scale)
            return F.pad(result, (0, 0, 0, 0, skip, 0)).to(query.dtype)
    # Ragged prefixes use the same document packing as a single document per row.
    starts = _query_starts(query_start, batch, query.device)
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
    return _flash_attention(query, key, value, query_start, kv_lengths, use_dummy_token, schedule, scale, backend)


def _torch_flash_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
    """Native FlashAttention's causal flag uses lower-right alignment."""
    dim = query.shape[-1]
    q, k, v = (t.transpose(1, 2) for t in (query, key, value))
    if dim % 8:
        q, k, v = (F.pad(t, (0, 8 - dim % 8)) for t in (q, k, v))
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return cast(torch.Tensor, TorchFlashDense.apply(q, k, v, scale, True))[..., :dim]
    # This is the operator used by CausalBias, without constructing a Tensor
    # subclass inside the compiled region. Eligibility was checked by the caller.
    result = torch.ops.aten._scaled_dot_product_flash_attention(
        q, k, v, 0.0, is_causal=True, return_debug_mask=False, scale=scale
    )[0]
    return result[..., :dim].transpose(1, 2)

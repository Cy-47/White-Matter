"""Compile-visible native and external FlashAttention kernels."""

from typing import Any

import torch
import torch.nn.functional as F

from white_matter._typing import compiler_assume_constant_result

# Dynamo understands SDPAParams metadata, but its pybind constructor needs an
# explicit graph registration before it can be used inside a nested region.
_SDPAParams = torch.compiler.allow_in_graph(torch.backends.cuda.SDPAParams)


@torch.library.custom_op("white_matter::flash_attention_decode", mutates_args=())
def flash_attention_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    causal: bool,
    num_splits: int = 0,
) -> torch.Tensor:
    from flash_attn import flash_attn_with_kvcache

    # FA's decode pybind entry has no native torch.compile registration.
    return flash_attn_with_kvcache(
        query,
        key,
        value,
        cache_seqlens=lengths,
        softmax_scale=scale,
        causal=causal,
        num_splits=num_splits,
    ).contiguous()


@flash_attention_decode.register_fake
def _fake_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    causal: bool,
    num_splits: int = 0,
) -> torch.Tensor:
    return query.new_empty(query.shape)


def torch_flash_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    *,
    static_cache: bool = False,
) -> torch.Tensor:
    """Single-query native FlashAttention on (B,H,N,D) caches, with GPU lengths."""
    if static_cache and torch.compiler.is_compiling():
        torch._dynamo.mark_static(query, 0)
        torch._dynamo.mark_static(key, 1)
        torch._dynamo.mark_static(key, 2)
    batch, heads, _, dim = query.shape
    original_dim = dim
    if dim % 8:
        query, key, value = (F.pad(t, (0, 8 - dim % 8)) for t in (query, key, value))
        dim = query.shape[-1]
    kv_heads, capacity = key.shape[1:3]
    # Fold contiguous KV heads into the batch to avoid transposing/copying the cache.
    fold_heads = batch > 1 and all(t.stride(0) == kv_heads * t.stride(1) for t in (key, value))
    if fold_heads:
        q = query.reshape(batch * kv_heads, heads // kv_heads, dim)
        k, v = (t.reshape(batch * kv_heads * capacity, 1, dim) for t in (key, value))
        lengths = lengths[:, None].expand(batch, kv_heads).reshape(-1).contiguous()
        sequences = batch * kv_heads
    else:
        q, k, v = (t.transpose(1, 2).flatten(0, 1) for t in (query, key, value))
        sequences = batch
    cu_q, cu_k = (
        _static_decode_offsets(query, key, fold_heads)
        if static_cache
        else _decode_offsets(sequences, capacity, query.device)
    )
    # The low-level operator preserves seqused_k without the public varlen
    # wrapper's unused RNG allocation. This path is inference-only.
    result = torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        1,
        capacity,
        0.0,
        False,
        False,
        scale=scale,
        seqused_k=lengths.contiguous(),
    )[0]
    return result.reshape(batch, 1, heads, dim)[..., :original_dim]


def can_use_torch_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    params = _SDPAParams(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), None, 0.0, False, True)
    return torch.backends.cuda.can_use_flash_attention(params)


def _decode_offsets(sequences: int, capacity: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build offsets in the graph so growing cache capacities stay symbolic."""
    cu_q = torch.arange(sequences + 1, device=device, dtype=torch.int32)
    return cu_q, cu_q * capacity


@compiler_assume_constant_result
def _static_decode_offsets(
    query: torch.Tensor, key: torch.Tensor, fold_heads: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = query.shape[0] * (key.shape[1] if fold_heads else 1)
    return _decode_offsets(sequences, key.shape[2], query.device)


class TorchFlashVarlen(torch.autograd.Function):
    """Trace native packed forward/backward directly, with no dropout RNG work."""

    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        scale: float,
    ) -> torch.Tensor:
        output, lse, rng, unused, _ = torch.ops.aten._flash_attention_forward(
            query, key, value, cu_q, cu_k, max_q, max_k, 0.0, True, False, scale=scale
        )
        ctx.save_for_backward(query, key, value, output, lse, cu_q, cu_k, rng, unused)
        ctx.max_q, ctx.max_k, ctx.scale = max_q, max_k, scale
        return output

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        query, key, value, output, lse, cu_q, cu_k, rng, unused = ctx.saved_tensors
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            grad,
            query,
            key,
            value,
            output,
            lse,
            cu_q,
            cu_k,
            ctx.max_q,
            ctx.max_k,
            0.0,
            True,
            rng,
            unused,
            scale=ctx.scale,
        )
        return dq, dk, dv, None, None, None, None, None


@torch.library.custom_op("white_matter::torch_flash_dense_forward", mutates_args=())
def _torch_flash_dense_forward(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float, causal: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, length, dim = query.shape
    output = query.new_empty(batch, length, heads, dim)
    lse = torch.ops.aten._flash_attention_forward_no_dropout_inplace(
        output,
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        None,
        None,
        length,
        key.shape[2],
        0.0,
        causal,
        False,
        scale=scale,
    )
    return output, lse


@_torch_flash_dense_forward.register_fake
def _fake_dense_forward(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float, causal: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, length, dim = query.shape
    return query.new_empty(batch, length, heads, dim), query.new_empty(batch, heads, length, dtype=torch.float32)


class TorchFlashDense(torch.autograd.Function):
    """Write contiguous (B,T,H,D) output to avoid copying it in native backward."""

    @staticmethod
    def forward(
        ctx: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float, causal: bool
    ) -> torch.Tensor:
        output, lse = _torch_flash_dense_forward(query, key, value, scale, causal)
        ctx.save_for_backward(query, key, value, output, lse)
        ctx.scale, ctx.causal = scale, causal
        return output

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        query, key, value, output, lse = ctx.saved_tensors
        # Dropout is zero; native backward does not read RNG state.
        rng = torch.empty(2, dtype=torch.uint64, device=query.device)
        unused = torch.empty(0, dtype=torch.uint64, device=query.device)
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            grad,
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            output,
            lse,
            None,
            None,
            query.shape[2],
            key.shape[2],
            0.0,
            ctx.causal,
            rng,
            unused,
            scale=ctx.scale,
        )
        return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2), None, None

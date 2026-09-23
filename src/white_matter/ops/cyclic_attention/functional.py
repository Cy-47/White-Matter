"""Public cyclic attention API; backend selection never changes the schedule."""

from typing import Literal

import torch

from .metadata import CyclicAttentionMetadata
from .reference import cyclic_attention_reference


def cyclic_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_stride: int = 1,
    query_offset: int = 0,
    metadata: CyclicAttentionMetadata | None = None,
    backend: Literal["reference", "tilelang"] = "reference",
) -> torch.Tensor:
    """Attend from query i to key slots j <= query_offset + query_stride * i.

    Tensors use (batch, heads, sequence, head_dim); K/V share heads and shape.
    Scale is 1/sqrt(head_dim), with no dropout. The reference preserves query
    dtype; TileLang requires CUDA BF16 and supports head dimensions 64/96/128.
    Q/K/V must share dtype/device. Document metadata must satisfy the contract
    of prepare_cyclic_attention_metadata, including its leading dummy slot.
    Returned storage layout is backend-specific. Inputs are never mutated.
    """
    if backend not in ("reference", "tilelang"):
        raise ValueError(f"unknown cyclic attention backend: {backend!r}")
    if type(query_stride) is not int or query_stride < 1:
        raise ValueError("query_stride must be a positive integer")
    if type(query_offset) is not int or not 0 <= query_offset < query_stride:
        raise ValueError("query_offset must be in [0, query_stride)")
    if any(x.ndim != 4 for x in (query, key, value)):
        raise ValueError("Q/K/V must have shape (batch, heads, sequence, head_dim)")
    if min(*query.shape, *key.shape) < 1 or key.shape != value.shape:
        raise ValueError("Q/K/V must be nonempty, with matching K/V shapes")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1] or query.shape[1] % key.shape[1]:
        raise ValueError("Q/K/V require matching batch/head_dim and divisible query/KV heads")
    if query_offset + query_stride * (query.shape[-2] - 1) >= key.shape[-2]:
        raise ValueError("query slots must be within the key sequence")
    if any(x.device != query.device or x.dtype != query.dtype for x in (key, value)):
        raise ValueError("Q/K/V must share dtype and device")
    if not query.is_floating_point():
        raise ValueError("Q/K/V must be floating-point tensors")
    if metadata is not None:
        if len(metadata) != 4:
            raise ValueError("document metadata requires four tensors")
        for tensor, length in zip(metadata, (query.shape[-2], key.shape[-2]) * 2, strict=True):
            if tensor.shape != (query.shape[0], length) or tensor.device != query.device:
                raise ValueError("document metadata shape/device does not match Q/K/V")
            if tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError("document metadata must be int32 or int64")
    if backend == "reference":
        return cyclic_attention_reference(
            query,
            key,
            value,
            query_stride=query_stride,
            query_offset=query_offset,
            metadata=metadata,
        )
    if query.device.type != "cuda" or query.dtype != torch.bfloat16 or query.shape[-1] not in (64, 96, 128):
        raise ValueError("TileLang cyclic attention requires CUDA BF16 with head_dim 64, 96, or 128")
    from ._tilelang.registration import cyclic_attn_doc_fwd, cyclic_attn_fwd

    if metadata is None:
        return cyclic_attn_fwd(query, key, value, query_stride, query_offset, 64)[0]
    return cyclic_attn_doc_fwd(query, key, value, *metadata, query_stride, query_offset, 64)[0]

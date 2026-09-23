"""Portable SDPA math reference consuming production document metadata."""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .metadata import CyclicAttentionMetadata


def cyclic_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_stride: int,
    query_offset: int,
    metadata: CyclicAttentionMetadata | None = None,
) -> torch.Tensor:
    q_len, k_len = query.shape[-2], key.shape[-2]
    q_slots = query_offset + query_stride * torch.arange(q_len, device=query.device)
    k_slots = torch.arange(k_len, device=query.device)
    # In WM's dummy-shifted layout, j <= t exposes the dummy token and real tokens before t.
    keep = q_slots.unsqueeze(-1) >= k_slots.unsqueeze(0)
    if metadata is not None:
        q_seg, k_seg, q_start, k_end = metadata
        dummy = k_seg.unsqueeze(1) == -1
        same = (q_seg.unsqueeze(2) == k_seg.unsqueeze(1)) | dummy
        after_start = (k_slots.view(1, 1, k_len) >= q_start.unsqueeze(2)) | dummy
        before_end = torch.arange(q_len, device=query.device).view(1, q_len, 1) < k_end.unsqueeze(1)
        keep = (keep & same & after_start & before_end).unsqueeze(1)
    # Pin the math backend so this reference is independent of fused-kernel dispatch.
    with sdpa_kernel([SDPBackend.MATH]):
        return F.scaled_dot_product_attention(
            query,
            key.to(query.dtype),
            value.to(query.dtype),
            attn_mask=keep,
            is_causal=False,
            scale=query.shape[-1] ** -0.5,
            dropout_p=0.0,
            enable_gqa=query.shape[1] != key.shape[1],
        )

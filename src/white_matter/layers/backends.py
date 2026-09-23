# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ordinary attention dispatch for feedback readers, independent of HF."""

import torch
import torch.nn.functional as F


def pack_kv_cache(
    key: torch.Tensor, value: torch.Tensor, keep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move visible slots first in (B,...,N,d) caches without host synchronization."""
    lengths = keep.sum(-1, dtype=torch.int32)
    order = (~keep).to(torch.uint8).argsort(dim=-1, stable=True)
    indices = order.view(keep.shape[0], *([1] * (key.ndim - 3)), keep.shape[1], 1).expand_as(key)
    return key.gather(-2, indices), value.gather(-2, indices), lengths


def _repeat_kv(states: torch.Tensor, repeats: int) -> torch.Tensor:
    batch, heads, length, dim = states.shape
    if repeats == 1:
        return states
    return states[:, :, None].expand(batch, heads, repeats, length, dim).reshape(batch, heads * repeats, length, dim)


def attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    scaling: float,
    implementation: str,
    is_causal: bool = True,
    cache_seqlens: torch.Tensor | None = None,
    num_splits: int = 0,
) -> torch.Tensor:
    repeats = query.shape[1] // key.shape[1]
    if implementation == "flash_attention_2":
        from flash_attn import flash_attn_func
        from white_matter.ops.flash_attention import flash_attention_decode

        if attention_mask is not None:
            raise ValueError("FlashAttention reader does not support an additive attention mask")
        dtype = query.dtype if query.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        query, key, value = (tensor.transpose(1, 2).to(dtype) for tensor in (query, key, value))
        if cache_seqlens is not None:
            # All supplied keys are already visible; WM appends this token only after its layer sweep.
            return flash_attention_decode(query, key, value, cache_seqlens, scaling, is_causal, num_splits)
        return flash_attn_func(
            query, key, value,
            dropout_p=0.0,
            softmax_scale=scaling,
            causal=is_causal,
        )
    if cache_seqlens is not None:
        slots = torch.arange(key.shape[-2], device=key.device)
        visible = slots[None, :] < cache_seqlens[:, None]
        key = key.masked_fill(~visible[:, None, :, None], 0)
        value = value.masked_fill(~visible[:, None, :, None], 0)
        keep = visible[:, None, None, :]
        if is_causal and query.shape[-2] > 1:
            query_slots = cache_seqlens[:, None] - query.shape[-2] + torch.arange(query.shape[-2], device=query.device)
            keep = keep & (slots[None, None, None, :] <= query_slots[:, None, :, None])
        attention_mask = query.new_zeros(keep.shape).masked_fill(~keep, float("-inf"))
    if implementation == "sdpa":
        if attention_mask is not None:
            key, value = _repeat_kv(key, repeats), _repeat_kv(value, repeats)
        result = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=scaling,
            is_causal=query.shape[2] > 1 and attention_mask is None and is_causal,
            dropout_p=0.0,
            enable_gqa=attention_mask is None,
        )
        return result.transpose(1, 2).contiguous()
    if implementation == "eager":
        key, value = _repeat_kv(key, repeats), _repeat_kv(value, repeats)
        scores = query @ key.transpose(2, 3) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        elif is_causal and query.shape[2] > 1:
            keep = torch.ones(query.shape[2], key.shape[2], dtype=torch.bool, device=query.device).tril()
            scores = scores.masked_fill(~keep, float("-inf"))
        weights = scores.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
        return (weights @ value).transpose(1, 2).contiguous()
    raise ValueError(f"unsupported attention implementation: {implementation!r}")

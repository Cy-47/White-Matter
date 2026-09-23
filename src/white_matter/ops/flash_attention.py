"""Compile-visible, read-only FlashAttention KV-cache access."""

import torch


@torch.library.custom_op("white_matter::flash_attention_decode", mutates_args=())
def flash_attention_decode(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    lengths: torch.Tensor, scale: float, causal: bool, num_splits: int = 0,
) -> torch.Tensor:
    from flash_attn import flash_attn_with_kvcache

    # FA's decode pybind entry has no native torch.compile registration.
    return flash_attn_with_kvcache(
        query, key, value, cache_seqlens=lengths, softmax_scale=scale, causal=causal, num_splits=num_splits,
    ).contiguous()


@flash_attention_decode.register_fake
def _fake_decode(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    lengths: torch.Tensor, scale: float, causal: bool, num_splits: int = 0,
) -> torch.Tensor:
    return query.new_empty(query.shape)

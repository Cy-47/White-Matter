"""Preserve causal FLOP accounting across attention implementations."""

import sys

import pytest
import torch

from benchmarks._flop_counter import attention_formula, make_counter


def test_sdpa_counter_does_not_import_external_flash(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    make_counter("sdpa")


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kernel", ["flash", "efficient"])
@pytest.mark.parametrize(
    ("queries", "keys", "flash_pairs", "efficient_pairs"), [(3, 5, 12, 6), (5, 3, 6, 12), (3, 3, 6, 6)]
)
def test_causal_pair_counts_match_kernel_alignment(
    backward, causal, kernel, queries, keys, flash_pairs, efficient_pairs
):
    name = f"_scaled_dot_product_{kernel}_attention" + ("_backward" if backward else "")
    formula = attention_formula(getattr(torch.ops.aten, name), layout="sdpa", backward=backward)
    q = torch.empty(2, 4, queries, 8)
    k = torch.empty(2, 4, keys, 8)
    pairs = (flash_pairs if kernel == "flash" else efficient_pairs) if causal else queries * keys
    assert formula(query=q, key=k, is_causal=causal) == (10 if backward else 4) * 2 * 4 * 8 * pairs


@pytest.mark.parametrize("backward", [False, True])
def test_efficient_attention_counts_only_visible_cache_slots(backward):
    name = "_scaled_dot_product_efficient_attention" + ("_backward" if backward else "")
    formula = attention_formula(getattr(torch.ops.aten, name), layout="sdpa", backward=backward)
    q = torch.empty(2, 4, 1, 8)
    k = torch.empty(2, 4, 5, 8)
    bias = torch.tensor([0, 0, 0, float("-inf"), float("-inf")]).view(1, 1, 1, 5)
    assert formula(query=q, key=k, attn_bias=bias, is_causal=False) == (10 if backward else 4) * 2 * 4 * 8 * 3


def test_native_cache_flops_ignore_unused_capacity():
    from benchmarks._flop_counter import attention_formula

    formula = attention_formula(torch.ops.aten._flash_attention_forward, layout="native_varlen")
    q, k = torch.empty(3, 4, 64), torch.empty(96, 2, 64)
    cu_q, cu_k = torch.arange(4, dtype=torch.int32), torch.arange(4, dtype=torch.int32) * 32
    count = formula(q, k, k, cu_q, cu_k, 1, 32, 0.0, False, False, seqused_k=torch.tensor([0, 7, 16]))
    assert count == 4 * 4 * 64 * (7 + 16)

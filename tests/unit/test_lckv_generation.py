"""LCKV cached generation matches a converged strictly causal reference."""

from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from white_matter.blocks._execution import jacobi
from white_matter.blocks.lckv import LCKVBlock
from white_matter.models import register_models


@pytest.mark.parametrize("surrounding", [0, 1])
@torch.no_grad()
def test_cached_lckv_matches_converged_jacobi(surrounding):
    register_models()
    torch.manual_seed(67)
    config = AutoConfig.for_model(
        "lckv",
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3 + 2 * surrounding,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        num_pre_layers=surrounding,
        num_post_layers=surrounding,
        num_passes=6,
        eos_token_id=66,
        document_separator_token_id=None,
        residual_dtype="fp32",
    )
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config).eval()
    ids = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])
    expected = model(ids, use_cache=False).logits
    cache = model.allocate_inference_cache(ids.shape[1])
    outputs = []
    for start, end in ((0, 2), (2, 4), (4, 5)):
        result = model(ids[:, start:end], past_key_values=cache, use_cache=True)
        outputs.append(result.logits)
        assert cache.get_seq_length() == end
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=2e-5, atol=3e-6)
    assert len(cache.layers) == 1 + 2 * surrounding


@torch.no_grad()
def test_lckv_generate_uses_cache():
    register_models()
    config = AutoConfig.for_model(
        "lckv",
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        num_passes=6,
        eos_token_id=66,
        document_separator_token_id=None,
    )
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config).eval()
    ids = torch.tensor([[2, 3, 4]])
    actual = model.generate(ids, max_new_tokens=2, do_sample=False, use_cache=True)
    expected = model.generate(ids, max_new_tokens=2, do_sample=False, use_cache=False)
    torch.testing.assert_close(actual, expected)


@torch.no_grad()
def test_lckv_dynamic_cache_respects_document_boundaries():
    register_models()
    torch.manual_seed(71)
    config = AutoConfig.for_model(
        "lckv",
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        num_pre_layers=1,
        num_post_layers=1,
        num_passes=8,
        eos_token_id=66,
        document_separator_token_id=66,
        residual_dtype="fp32",
    )
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config).eval()
    ids = torch.tensor([[2, 3, 66, 4, 5, 6], [7, 66, 8, 9, 10, 11]])
    expected = model(ids, use_cache=False).logits
    cache = model.allocate_inference_cache()
    actual = torch.cat(
        [
            model(ids[:, :3], past_key_values=cache, use_cache=True).logits,
            model(ids[:, 3:], past_key_values=cache, use_cache=True).logits,
        ],
        dim=1,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)


@torch.no_grad()
def test_fixed_source_prefill_matches_full_stack_reference():
    register_models()
    torch.manual_seed(83)
    config = AutoConfig.for_model(
        "lckv",
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        num_pre_layers=1,
        num_post_layers=1,
        num_passes=4,
        eos_token_id=66,
        document_separator_token_id=None,
        residual_dtype="fp32",
    )
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config).eval()
    ids = torch.randint(1, 66, (2, 9))
    optimized_cache = model.allocate_inference_cache(12)
    optimized = model(ids, past_key_values=optimized_cache, use_cache=True, logits_to_keep=1).logits
    reference_cache = model.allocate_inference_cache(12)
    pool = model.model.decoder.block.kv_pool
    with patch.object(LCKVBlock, "forward", jacobi.forward_jacobi):
        pool.train()  # The reference also uses the original full-source projection.
        try:
            reference = model(ids, past_key_values=reference_cache, use_cache=True, logits_to_keep=1).logits
        finally:
            pool.eval()
    torch.testing.assert_close(optimized, reference, rtol=0, atol=0)
    for actual, expected in zip(optimized_cache.layers, reference_cache.layers, strict=True):
        for name in ("keys", "values"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)

    # Document-aware Jacobi without a static cache takes the same source path.
    documents = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 2, 2], [0, 0, 1, 1, 1, 2, 2, 2, 2]])
    optimized = model(ids, use_cache=False, document_ids=documents).logits
    with patch.object(LCKVBlock, "forward", jacobi.forward_jacobi):
        pool.train()
        try:
            reference = model(ids, use_cache=False, document_ids=documents).logits
        finally:
            pool.eval()
    torch.testing.assert_close(optimized, reference, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_compiled_bf16_fixed_source_prefill_matches_full_stack_reference():
    from white_matter.modules import KVPool

    register_models()
    torch.manual_seed(89)
    config = AutoConfig.for_model(
        "lckv",
        vocab_size=131,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=64,
        num_pre_layers=1,
        num_post_layers=1,
        num_passes=4,
        eos_token_id=130,
        document_separator_token_id=None,
        residual_dtype="bf16",
    )
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_config(config)
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.to(dtype=torch.bfloat16)
        elif isinstance(module, KVPool):
            module.k_proj_weight.data = module.k_proj_weight.data.to(torch.bfloat16)
            module.v_proj_weight.data = module.v_proj_weight.data.to(torch.bfloat16)
    model = model.cuda().eval()
    ids = torch.randint(1, 130, (2, 16), device="cuda")
    reference_cache = model.allocate_inference_cache(20)
    pool = model.model.decoder.block.kv_pool
    with patch.object(LCKVBlock, "forward", jacobi.forward_jacobi):
        pool.train()
        try:
            reference = model(ids, past_key_values=reference_cache, use_cache=True, logits_to_keep=1).logits
        finally:
            pool.eval()
    model.compile(options={"emulate_precision_casts": True, "reorder_for_locality": False})
    optimized_cache = model.allocate_inference_cache(20)
    optimized = model(ids, past_key_values=optimized_cache, use_cache=True, logits_to_keep=1).logits
    torch.testing.assert_close(optimized, reference, rtol=2e-2, atol=5e-3)
    for actual, expected in zip(optimized_cache.layers, reference_cache.layers, strict=True):
        for name in ("keys", "values"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=2e-2, atol=5e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_lckv_flash_cached_matches_full_reference():
    register_models()
    torch.manual_seed(71)
    config = AutoConfig.for_model(
        "lckv",
        vocab_size=131,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=64,
        num_pre_layers=1,
        num_post_layers=1,
        num_passes=8,
        eos_token_id=130,
        document_separator_token_id=None,
        residual_dtype="bf16",
    )
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_config(config).cuda().eval()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]], device="cuda")
    expected = model(ids, use_cache=False).logits
    cache = model.allocate_inference_cache(ids.shape[1])
    prefill = model(ids[:, :4], past_key_values=cache, use_cache=True).logits
    decode = torch.cat(
        [model(ids[:, i : i + 1], past_key_values=cache, use_cache=True).logits for i in range(4, ids.shape[1])], dim=1
    )
    torch.testing.assert_close(torch.cat((prefill, decode), dim=1), expected, rtol=3e-2, atol=1e-2)

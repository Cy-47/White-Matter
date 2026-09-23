"""LCKV cached generation matches a converged strictly causal reference."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from white_matter.models import register_models


@pytest.mark.parametrize("surrounding", [0, 1])
@torch.no_grad()
def test_cached_lckv_matches_converged_jacobi(surrounding):
    register_models()
    torch.manual_seed(67)
    config = AutoConfig.for_model(
        "lckv", vocab_size=67, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3 + 2 * surrounding, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, max_position_embeddings=64,
        num_pre_layers=surrounding, num_post_layers=surrounding,
        num_passes=6, eos_token_id=66, document_separator_token_id=None,
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
        "lckv", vocab_size=67, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=64, num_passes=6,
        eos_token_id=66, document_separator_token_id=None,
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
        "lckv", vocab_size=67, hidden_size=32, intermediate_size=64,
        num_hidden_layers=5, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=64, num_pre_layers=1,
        num_post_layers=1, num_passes=8, eos_token_id=66,
        document_separator_token_id=66, residual_dtype="fp32",
    )
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config).eval()
    ids = torch.tensor([[2, 3, 66, 4, 5, 6], [7, 66, 8, 9, 10, 11]])
    expected = model(ids, use_cache=False).logits
    cache = model.allocate_inference_cache()
    actual = torch.cat([
        model(ids[:, :3], past_key_values=cache, use_cache=True).logits,
        model(ids[:, 3:], past_key_values=cache, use_cache=True).logits,
    ], dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_lckv_flash_cached_matches_full_reference():
    register_models()
    torch.manual_seed(71)
    config = AutoConfig.for_model(
        "lckv", vocab_size=131, hidden_size=128, intermediate_size=192,
        num_hidden_layers=5, num_attention_heads=2, num_key_value_heads=1,
        head_dim=64, max_position_embeddings=64, num_pre_layers=1,
        num_post_layers=1, num_passes=8, eos_token_id=130,
        document_separator_token_id=None, residual_dtype="bf16",
    )
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_config(config).cuda().eval()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]], device="cuda")
    expected = model(ids, use_cache=False).logits
    cache = model.allocate_inference_cache(ids.shape[1])
    prefill = model(ids[:, :4], past_key_values=cache, use_cache=True).logits
    decode = torch.cat([
        model(ids[:, i:i + 1], past_key_values=cache, use_cache=True).logits
        for i in range(4, ids.shape[1])
    ], dim=1)
    torch.testing.assert_close(torch.cat((prefill, decode), dim=1), expected, rtol=3e-2, atol=1e-2)


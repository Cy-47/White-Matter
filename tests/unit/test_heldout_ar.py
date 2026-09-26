"""Exact AR scoring uses the complete model and an independent cache per batch."""

import copy

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from evals.heldout_ar import forward_hidden, score_hidden
from white_matter.models import register_models


@pytest.mark.parametrize("family", ["white_matter", "lckv"])
@pytest.mark.parametrize("separator", [None, 30])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[
    pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
])])
@torch.inference_mode()
def test_ar_evaluation_matches_complete_model(family, separator, device):
    register_models()
    torch.manual_seed(71)
    options = {"num_kv_channels": 1} if family == "white_matter" else {}
    config = AutoConfig.for_model(
        family, vocab_size=31, eos_token_id=30, document_separator_token_id=separator,
        hidden_size=128 if device == "cuda" else 16,
        intermediate_size=256 if device == "cuda" else 32, num_hidden_layers=4,
        num_pre_layers=1, num_post_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=64 if device == "cuda" else 8,
        num_passes=1, residual_dtype="bf16", **options,
    )
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config).to(device).eval()
    reference = copy.deepcopy(model)
    reference.config.prefill_mode = "autoregressive"
    ids = torch.tensor([[1, 2, 30, 3, 4], [5, 30, 6, 7, 8]], device=device)
    original_config = model.config.to_dict()
    expected = reference.model(ids, use_cache=True).last_hidden_state
    for _ in range(2):
        torch.testing.assert_close(forward_hidden(model, ids, "ar"), expected)
    assert model.config.to_dict() == original_config
    torch.testing.assert_close(
        forward_hidden(model, ids, "configured"), model.model(ids, use_cache=False).last_hidden_state,
    )


def test_score_hidden_accepts_inference_outputs_with_trainable_weights():
    torch.manual_seed(72)
    with torch.inference_mode():
        hidden = torch.randn(1, 5, 8)
    weight = torch.nn.Parameter(torch.randn(11, 8))
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        logits = torch.nn.functional.linear(hidden[:, :-1], weight)
        expected = torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten(), reduction="sum")
    assert score_hidden(hidden, ids, weight, chunk_size=2) == pytest.approx(float(expected), abs=1e-5)
    assert weight.grad is None


def test_ar_evaluation_restores_prefill_policy_on_failure(monkeypatch):
    from white_matter.models.white_matter import WhiteMatterConfig, WhiteMatterForCausalLM

    with torch.device("meta"):
        model = WhiteMatterForCausalLM(WhiteMatterConfig(prefill_mode="jacobi"))

    def fail(*args, **kwargs):
        raise RuntimeError("failed forward")

    monkeypatch.setattr(model.model, "forward", fail)
    with pytest.raises(RuntimeError, match="failed forward"):
        forward_hidden(model, torch.tensor([[1, 2]]), "ar")
    assert model.config.prefill_mode == "jacobi"

"""Automatic backend selection and strict-causal model routing on CUDA."""

import copy
import importlib

import pytest
import torch

from tests.unit.test_model_compilation import tiny_model
from white_matter.compilation import execution_policy
from white_matter.ops import strict_causal_attention

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dummy", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_masked_training_compiles_without_host_schedule(monkeypatch, dtype, dummy, masked):
    module = importlib.import_module("white_matter.ops.strict_causal_attention")

    def reject_schedule(*args, **kwargs):
        raise AssertionError("Single-query training must keep lengths on the GPU")

    monkeypatch.setattr(module, "prepare_strict_causal_metadata", reject_schedule)
    q = torch.randn(3, 4, 1, 64, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(3, 2, 11, 64, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    mask = torch.tensor([[False] * 11, [True, False] * 5 + [True], [True] * 11], device="cuda")
    if masked:
        k.masked_fill_(~mask[:, None, :, None], float("nan"))
        v.masked_fill_(~mask[:, None, :, None], float("nan"))
    k.requires_grad_()
    v.requires_grad_()
    lengths = torch.tensor([0, 6, 9], device="cuda", dtype=torch.int32)
    kwargs = {
        "query_start": lengths + 1,
        "kv_lengths": lengths,
        "key_mask": mask if masked else None,
        "use_dummy_token": dummy,
    }
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        expected = strict_causal_attention(q, k, v, backend="reference", **kwargs)
    try:
        compiled = torch.compile(strict_causal_attention, fullgraph=True)
        with execution_policy():
            actual = compiled(q, k, v, **kwargs)
        probe = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (q, k, v), probe)
        expected_grads = torch.autograd.grad(expected, (q, k, v), probe)
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
        torch.testing.assert_close(actual_grads, expected_grads, rtol=0.03, atol=0.03)
        if masked or not dummy:
            assert actual[0].count_nonzero() == 0
        assert all(torch.isfinite(g).all() for g in actual_grads)
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("family", ["white_matter", "lckv", "feedback_transformer"])
@torch.inference_mode()
def test_fp32_execution_preserves_strict_attention_precision(monkeypatch, family):
    from evals.execution import execution

    module = importlib.import_module("white_matter.ops.strict_causal_attention")
    model = tiny_model(family, execution_mode="autoregressive", prefill_mode="autoregressive").cuda().eval()
    original = module._sdpa
    observed = []

    def check_precision(q, k, v, *args):
        observed.append(q.dtype)
        assert q.dtype == k.dtype == v.dtype == torch.float32
        return original(q, k, v, *args)

    def reject_flash(*args, **kwargs):
        raise AssertionError("FP32 evaluation must use reference strict attention")

    monkeypatch.setattr(module, "_sdpa", check_precision)
    monkeypatch.setattr(module, "_accelerated_attention", reject_flash)
    settings = [layer.self_attn.attention_implementation for layer in model.model.decoder.block.layers]
    with execution(model, mode="autoregressive", precision="fp32"):
        result = model(torch.tensor([[1, 2, 3, 4]], device="cuda"), use_cache=True)
        result = model(torch.tensor([[5]], device="cuda"), past_key_values=result.past_key_values, use_cache=True)
    assert observed
    assert torch.isfinite(result.logits).all()
    assert settings == [layer.self_attn.attention_implementation for layer in model.model.decoder.block.layers]


@pytest.mark.parametrize("standalone", [False, True])
@torch.inference_mode()
def test_masked_cache_auto_preserves_visibility(monkeypatch, standalone):
    module = importlib.import_module("white_matter.ops.strict_causal_attention")
    monkeypatch.setattr(module, "_standalone_flash_available", lambda: standalone)
    q = torch.randn(3, 4, 1, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, 2, 8, 64, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    keep = torch.tensor([[False] * 8, [True, False] * 4, [True] * 8], device="cuda")
    lengths = torch.tensor([0, 6, 7], device="cuda", dtype=torch.int32)
    k.masked_fill_(~keep[:, None, :, None], float("nan"))
    v.masked_fill_(~keep[:, None, :, None], float("nan"))
    expected = strict_causal_attention(q, k, v, query_start=lengths, key_mask=keep, backend="reference")
    try:
        compiled = torch.compile(strict_causal_attention, fullgraph=True)
        with execution_policy():
            actual = compiled(q, k, v, query_start=lengths, key_mask=keep)
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("family", ["lckv", "feedback_transformer"])
@torch.inference_mode()
def test_cached_strict_models_use_shared_api(monkeypatch, family, standalone):
    module = importlib.import_module("white_matter.ops.strict_causal_attention")
    monkeypatch.setattr(module, "_standalone_flash_available", lambda: standalone)
    model = tiny_model(family, prefill_mode="autoregressive").cuda().eval()
    reference = copy.deepcopy(model)
    for layer in reference.model.decoder.block.layers:
        layer.self_attn.attention_implementation = "eager"

    # These models have only strict-causal feedback layers.
    def reject_ordinary(*args, **kwargs):
        raise AssertionError("Strict-causal readers must use the shared API")

    monkeypatch.setattr("white_matter.layers.white_matter.attention_forward", reject_ordinary)
    try:
        model.compile(options={"emulate_precision_casts": True})
        actual_cache = expected_cache = None
        with execution_policy():
            for ids in ([[1, 2, 3, 4]], [[5]], [[6]]):
                ids = torch.tensor(ids, device="cuda")
                expected = reference(ids, use_cache=True, past_key_values=expected_cache)
                actual = model(ids, use_cache=True, past_key_values=actual_cache)
                torch.testing.assert_close(actual.logits, expected.logits, atol=0.03, rtol=0.03)
                actual_cache, expected_cache = actual.past_key_values, expected.past_key_values
    finally:
        torch.compiler.reset()

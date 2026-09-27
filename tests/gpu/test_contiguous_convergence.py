"""Compare real compiled FlashAttention execution with explicit causal masks."""

import pytest
import torch
from transformers import AutoModelForCausalLM

from studies.prefill_convergence.contiguous import forward_contiguous
from studies.prefill_convergence.protocol import RECIPE
from training.recipes import load_recipe
from white_matter.modules.precision import model_autocast_context

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


@pytest.mark.parametrize(
    ("length", "chunks", "batch"), [(129, 16, 2), (16, 1, 2), (16, 16, 2), (2048, 64, 2), (2048, 32, 64)]
)
@torch.inference_mode()
def test_compiled_flash_matches_reference(length, chunks, batch):
    torch.manual_seed(52)
    config = load_recipe(RECIPE).model
    config.vocab_size = 257
    config.eos_token_id = 256
    config.document_separator_token_id = None
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_config(config).cuda().eval()
    block = model.model.decoder.block
    x = torch.randn(batch, length, config.hidden_size, device="cuda")
    expected = forward_contiguous(block, x, num_passes=3, chunks=chunks, output_final_state=True)
    for _ in range(2):
        actual = forward_contiguous(
            block, x, num_passes=3, chunks=chunks, backend="flash_attention_2", compiled=True, output_final_state=True
        )
        # Both paths use BF16 GEMMs. FA versus SDPA reduction differences can
        # compound across iterations; require small absolute and RMS errors.
        for ref, got in zip((expected[0], *expected[1]), (actual[0], *actual[1]), strict=True):
            error = (got.float() - ref.float()).square().mean().sqrt()
            scale = ref.float().square().mean().sqrt()
            assert float(error / scale.clamp_min(1e-6)) < 0.015
            torch.testing.assert_close(got, ref, rtol=0.04, atol=0.06)
    with model_autocast_context("cuda"):
        if chunks == 1:
            jacobi = block.forward_jacobi(x, num_passes=3, num_gradient_passes=0)
            torch.testing.assert_close(actual[0], jacobi, rtol=0.04, atol=0.06)


@pytest.mark.parametrize("offset", [0, 31, 63])
@torch.inference_mode()
def test_cyclic64_attention_matches_reference(offset):
    from white_matter.ops import cyclic_attention

    torch.manual_seed(53)
    query = torch.randn(2, 6, 32, 96, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(2, 3, 2048, 96, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    expected = cyclic_attention(query, key, value, query_stride=64, query_offset=offset)
    actual = torch.compile(cyclic_attention, fullgraph=True)(
        query, key, value, query_stride=64, query_offset=offset, backend="tilelang"
    )
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.02)


@torch.inference_mode()
def test_timing_contiguous_executes_every_layer_each_pass(monkeypatch):
    from studies.prefill_convergence import benchmark, contiguous
    from tests.unit.test_experiment_protocols import tiny_model

    model = tiny_model("cuda")
    block = model.model.decoder.block
    for layer in block.layers:
        layer.self_attn.attention_implementation = "flash_attention_2"
    # Keep layer hooks visible while exercising the actual benchmark dispatch.
    monkeypatch.setattr(benchmark, "compile_feedback", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(contiguous, "_compiled_chunk", contiguous._chunk)
    calls = [0] * len(block.layers)

    def count(index):
        def hook(module, args, output):
            calls[index] += 1

        return hook

    handles = [layer.register_forward_hook(count(i)) for i, layer in enumerate(block.layers)]
    try:
        benchmark.measure(
            model,
            torch.ones(2, 16, dtype=torch.long, device="cuda"),
            {"contiguous2": 3},
            warmups=0,
            repetitions=1,
            include_ar=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [6] * len(block.layers)

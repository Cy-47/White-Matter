"""Bounded inference agrees with the detached training computation."""

import pytest
import torch

from white_matter.blocks import FeedbackDecoderLayer, WhiteMatterBlock
from white_matter.blocks._execution import cyclic
from white_matter.layers import WhiteMatterAttention
from white_matter.modules import GatedMLP, KVPool, RotaryEmbedding
from white_matter.modules.precision import model_autocast_context


@pytest.fixture(autouse=True)
def reset_compiler():
    # Each case builds a distinct model; don't accumulate its specializations.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.parametrize("channels", [1, 2, 4])
@pytest.mark.parametrize("passes", [1, 3])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
            ],
        ),
    ],
)
@torch.no_grad()
def test_cyclic_inference_matches_general_execution(monkeypatch, channels, passes, device):
    torch.manual_seed(81)
    dim = 64 if device == "cuda" else 8
    width = 2 * dim
    block = (
        WhiteMatterBlock(
            [
                FeedbackDecoderLayer(width, WhiteMatterAttention(width, 2, dim), GatedMLP(width, 2 * width))
                for _ in range(4)
            ],
            KVPool(width, 1, dim, 5, channels),
            RotaryEmbedding(dim),
            num_passes=passes,
        )
        .to(device)
        .eval()
    )
    length, groups = (256, 4) if device == "cuda" else (67, 3)
    x = torch.randn(2, length, width, device=device)
    # Keep the independent training schedule, with every pass explicitly detached.
    with torch.enable_grad(), model_autocast_context(device):
        expected, expected_kv = block(x, num_gradient_passes=0, cyclic_groups=groups, output_final_state=True)
    # Match canonical production KV, not the reference driver's exported strides.
    shape = (*expected_kv[0].shape[:-2], length + 8, dim)
    keys = torch.full(shape, float("nan"), device=device, dtype=torch.bfloat16 if device == "cuda" else x.dtype)
    values = torch.full_like(keys, float("nan"))
    run = torch.compile(cyclic.forward_cyclic, fullgraph=True) if device == "cuda" else cyclic.forward_cyclic
    actual, state = run(
        block,
        x,
        kv_cache=(keys, values),
        num_passes=passes,
        num_gradient_passes=0,
        cyclic_groups=groups,
        output_final_state=True,
    )
    assert all(torch.isnan(t[..., length + int(block.use_dummy_token) :, :]).all() for t in (keys, values))
    assert all(t.shape == expected_kv[0].shape for t in state)
    assert state[0].data_ptr() == keys.data_ptr()
    assert state[1].data_ptr() == values.data_ptr()
    keys, values = state
    if device == "cpu":
        torch.testing.assert_close((actual, keys, values), (expected, *expected_kv), rtol=2e-5, atol=3e-6)
    else:
        # Compare eager and compiled BF16 execution; eager tiling itself agrees.
        for result, reference in zip((actual, keys, values), (expected, *expected_kv), strict=True):
            delta, reference = result.float() - reference.float(), reference.float()
            assert delta.norm() <= 0.01 * reference.norm() + 1e-6
            assert delta.abs().max() <= 0.02 * reference.abs().max() + 1e-6

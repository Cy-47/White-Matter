"""The learned dummy parameter must preserve explicit-concatenation training gradients."""

import copy

import pytest
import torch

from training.compile import compile_feedback
from training.forward import TrainingForward
from training.precision import configure_precision
from white_matter import WhiteMatterConfig, WhiteMatterForCausalLM
from white_matter.modules.precision import model_autocast_context


def _project_with_explicit_dummy(block):
    """Reference: concatenate the parameter explicitly before calling the pool."""
    project = block.kv_pool.project_sequence

    def reference(stacked, position_embeddings, *, dummy_token=None):
        dummy = block.dummy_token.view(1, 1, -1).expand(stacked.shape[0], 1, -1)
        dummy = dummy.to(stacked.dtype).unsqueeze(2).expand(-1, -1, stacked.shape[2], -1)
        return project(torch.cat((dummy, stacked), dim=1), position_embeddings)

    block.kv_pool.project_sequence = reference


@pytest.mark.parametrize(
    ("channels", "gradient_passes", "device", "residual_dtype", "compiled"),
    [(k, passes, "cpu", "fp32", False) for k in (1, 2, 4) for passes in (1, 3)]
    + [
        pytest.param(k, passes, "cuda", dtype, False, marks=pytest.mark.gpu)
        for k in (1, 2, 4)
        for passes in (1, 3)
        for dtype in ("fp32", "bf16")
    ]
    + [pytest.param(2, 1, "cuda", dtype, True, marks=pytest.mark.gpu) for dtype in ("fp32", "bf16")],
)
def test_dummy_projection_preserves_training(channels, gradient_passes, device, residual_dtype, compiled):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    configure_precision(device)
    torch.manual_seed(97)
    config = WhiteMatterConfig(
        vocab_size=101,
        eos_token_id=100,
        document_separator_token_id=100,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=6,
        num_pre_layers=1,
        num_post_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_kv_channels=channels,
        cyclic_groups=4,
        num_passes=3,
        router_layer_stride=2,
        residual_dtype=residual_dtype,
    )
    config._attn_implementation = "flash_attention_2" if device == "cuda" else "sdpa"
    model = WhiteMatterForCausalLM(config).to(device).train()
    # A nonzero dummy catches wrong values/positions as well as detached gradients.
    with torch.no_grad():
        model.model.decoder.block.dummy_token.normal_(std=0.1)
    reference = copy.deepcopy(model)
    _project_with_explicit_dummy(reference.model.decoder.block)
    batches = [torch.randint(0, 100, (2, 17), device=device) for _ in range(2)]
    for step, ids in enumerate(batches):
        ids[0, [2 + step, 9]] = 100
        ids[1, [5, 12 + step]] = 100

    def run(current):
        if compiled:
            compile_feedback(current, mode="default")
        forward = TrainingForward(current)
        if compiled:
            forward = torch.compile(forward, fullgraph=False)
        records = []
        for ids in batches:
            current.zero_grad(set_to_none=True)
            inputs = current.get_input_embeddings()(ids)
            inputs.retain_grad()
            with model_autocast_context(device):
                hidden = forward(inputs, 3, gradient_passes, token_ids=ids, compute_ce=False)
                loss = forward(inputs, 3, gradient_passes, token_ids=ids)
            loss.backward()
            gradients = {name: p.grad.detach().clone() for name, p in current.named_parameters()}
            assert gradients["model.decoder.block.dummy_token"].norm() > 0
            records.append((hidden.detach(), loss.detach(), inputs.grad, gradients))
        return records

    expected, actual = run(reference), run(model)
    # GPU reductions use BF16; keep the existing production gradient tolerances.
    torch.testing.assert_close(
        actual,
        expected,
        rtol=5e-2 if device == "cuda" else 2e-5,
        atol=2e-4 if device == "cuda" else 2e-6,
    )

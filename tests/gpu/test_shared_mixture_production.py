"""Production cyclic training against independent routing and attention arithmetic."""

import copy
from types import MethodType

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from studies.shared_mixture.model import SharedMixtureConfig, register_model
from tests.numerics import assert_close, assert_gradient_maps_close
from tests.unit.test_shared_mixture_study import _manual_projection
from training.compile import compile_feedback, compile_training_forward
from training.forward import TrainingForward
from training.precision import attention_kernel_context, configure_precision
from white_matter.modules.precision import model_autocast_context

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


def _run(model, ids, *, compiled):
    model.zero_grad(set_to_none=True)
    inputs = model.get_input_embeddings()(ids)
    inputs.retain_grad()
    runner = compile_training_forward(TrainingForward(model)) if compiled else TrainingForward(model)
    with attention_kernel_context("cuda"), model_autocast_context("cuda"):
        hidden = runner(inputs, 3, 2, token_ids=ids, compute_ce=False)
        loss = runner(inputs, 3, 2, token_ids=ids)
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            gradients[name] = parameter.grad.detach().clone()
    return hidden.detach(), loss.detach(), inputs.grad.detach(), gradients


def test_full_rank_shared_mixture_production_training_matches_independent_reference():
    """Cover k=L, g=8, 1+2 passes, packed masks and checkpoint recomputation."""
    pytest.importorskip("tilelang")
    register_model()
    configure_precision("cuda")
    torch.manual_seed(627)
    config = AutoConfig.for_model(
        SharedMixtureConfig.model_type,
        vocab_size=257,
        hidden_size=192,
        intermediate_size=384,
        num_hidden_layers=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=96,
        max_position_embeddings=256,
        eos_token_id=256,
        document_separator_token_id=256,
        num_kv_channels=16,
        num_passes=3,
        cyclic_groups=8,
        router_layer_stride=2,
        residual_dtype="fp32",
    )
    config._attn_implementation = "flash_attention_2"
    optimized = AutoModelForCausalLM.from_config(config).cuda().train()
    reference = copy.deepcopy(optimized)
    # A nonzero router matrix exercises token-dependent mixtures; the default
    # zero matrix would only verify the initialization bias.
    with torch.no_grad():
        optimized.model.decoder.block.kv_pool.mixer.router.linear.weight.normal_(std=0.003)
    reference.load_state_dict(optimized.state_dict())
    for layer in reference.model.decoder.block.layers:
        layer.self_attn._force_cyclic_reference = True
    pool = reference.model.decoder.block.kv_pool
    pool._project = MethodType(_manual_projection, pool)
    compile_feedback(optimized, mode="default")

    ids = torch.randint(0, 256, (2, 128), device="cuda")
    ids[0, [7, 23, 71]] = 256
    ids[1, [13, 39, 87]] = 256
    expected = _run(reference, ids, compiled=False)
    actual = _run(optimized, ids, compiled=True)
    assert_close(actual[0], expected[0], rtol=5e-2, atol=2e-2, aggregate_rtol=3e-2, cosine=0.9995)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-3, atol=2e-3)
    assert_close(actual[2], expected[2], rtol=1e-1, atol=2e-3, aggregate_rtol=3e-2, cosine=0.999)
    assert_gradient_maps_close(
        actual[3],
        expected[3],
        rtol=1e-1,
        atol=2e-3,
        aggregate_rtol=3e-2,
        cosine=0.999,
    )

"""Production-feature gradient gates for the paper ablation implementations."""

import copy

import pytest
import torch
from transformers import AutoModelForCausalLM

from studies.rank_20k.depth_causal import DepthCausalConfig, register_model as register_depth_causal
from tests.numerics import assert_close, assert_gradient_maps_close
from training.compile import compile_feedback, compile_training_forward
from training.forward import TrainingForward
from training.precision import attention_kernel_context, configure_precision
from white_matter.models import register_models
from white_matter.models.white_matter.configuration_white_matter import WhiteMatterConfig
from white_matter.modules.precision import model_autocast_context


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


def _configuration(**changes):
    values = dict(
        vocab_size=257, hidden_size=192, intermediate_size=384,
        num_hidden_layers=16, num_attention_heads=2, num_key_value_heads=1,
        head_dim=96, max_position_embeddings=256, rope_theta=1_000_000.0,
        eos_token_id=256, document_separator_token_id=256,
        num_kv_channels=16, num_passes=3, cyclic_groups=8,
        router_layer_stride=2, router_prior="shifted_identity:0.25",
        residual_dtype="fp32",
    )
    values.update(changes)
    config = WhiteMatterConfig(**values)
    config._attn_implementation = "flash_attention_2"
    return config


def _ids(length=128):
    ids = torch.randint(0, 256, (2, length), device="cuda")
    ids[0, [7, 23, 71]] = 256
    ids[1, [13, 39, 87]] = 256
    return ids


def _run(model, ids, *, compiled: bool, passes: int, gradient_passes: int):
    model.zero_grad(set_to_none=True)
    inputs = model.get_input_embeddings()(ids)
    inputs.retain_grad()
    runner = compile_training_forward(TrainingForward(model)) if compiled else TrainingForward(model)
    with attention_kernel_context("cuda"), model_autocast_context("cuda"):
        hidden = runner(inputs, passes, gradient_passes, token_ids=ids, compute_ce=False)
        loss = runner(inputs, passes, gradient_passes, token_ids=ids, compute_ce=True)
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            gradients[name] = parameter.grad.detach().clone()
    return hidden.detach(), loss.detach(), inputs.grad.detach(), gradients


def _compare(actual, expected, *, subset=False):
    assert_close(actual[0], expected[0], rtol=5e-2, atol=2e-2, aggregate_rtol=3e-2, cosine=0.9995)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-3, atol=2e-3)
    assert_close(actual[2], expected[2], rtol=1e-1, atol=2e-3, aggregate_rtol=3e-2, cosine=0.999)
    expected_grads = {name: expected[3][name] for name in actual[3]} if subset else expected[3]
    assert_gradient_maps_close(
        actual[3], expected_grads, rtol=1e-1, atol=2e-3, aggregate_rtol=3e-2, cosine=0.999,
    )


def test_static_k16_matches_zero_weight_dynamic_reference_with_packed_documents():
    pytest.importorskip("tilelang")
    register_models()
    configure_precision("cuda")
    torch.manual_seed(1419)
    static = AutoModelForCausalLM.from_config(_configuration(router_dynamic=False)).cuda().train()
    dynamic = AutoModelForCausalLM.from_config(_configuration()).cuda().train()
    dynamic.load_state_dict(copy.deepcopy(static.state_dict()))
    for layer in dynamic.model.decoder.block.layers:
        layer.self_attn._force_cyclic_reference = True
    compile_feedback(static, mode="default")
    ids = _ids()
    expected = _run(dynamic, ids, compiled=False, passes=3, gradient_passes=2)
    actual = _run(static, ids, compiled=True, passes=3, gradient_passes=2)
    _compare(actual, expected, subset=True)


def test_jacobi_checkpoint_matches_uncheckpointed_full_training_gradients():
    pytest.importorskip("flash_attn")
    register_models()
    configure_precision("cuda")
    torch.manual_seed(1502)
    common = dict(num_kv_channels=8, router_prior="cyclic:0.25", execution_mode="jacobi", cyclic_groups=1)
    optimized = AutoModelForCausalLM.from_config(_configuration(**common, checkpoint_jacobi_passes=True)).cuda().train()
    reference = AutoModelForCausalLM.from_config(_configuration(**common)).cuda().train()
    reference.load_state_dict(copy.deepcopy(optimized.state_dict()))
    for layer in reference.model.decoder.block.layers:
        layer.self_attn._force_jacobi_reference = True
    compile_feedback(optimized, mode="default")
    ids = _ids(length=128)
    expected = _run(reference, ids, compiled=False, passes=3, gradient_passes=2)
    from unittest.mock import patch
    from white_matter.layers import white_matter as attention_module

    original = attention_module.cyclic_attention

    def reject_tilelang(*args, **kwargs):
        if kwargs.get("backend") == "tilelang":
            raise AssertionError("Jacobi CUDA execution must use FlashAttention")
        return original(*args, **kwargs)

    with patch.object(attention_module, "cyclic_attention", side_effect=reject_tilelang):
        actual = _run(optimized, ids, compiled=True, passes=3, gradient_passes=2)
    _compare(actual, expected)


def test_depth_causal_compiled_training_matches_eager_on_packed_documents():
    register_depth_causal()
    configure_precision("cuda")
    torch.manual_seed(1741)
    config = DepthCausalConfig(
        vocab_size=257, hidden_size=192, intermediate_size=384,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=96, max_position_embeddings=256, rope_theta=1_000_000.0,
        eos_token_id=256, document_separator_token_id=256,
        num_kv_channels=4, num_passes=1, router_layer_stride=2,
    )
    config._attn_implementation = "flash_attention_2"
    optimized = AutoModelForCausalLM.from_config(config).cuda().train()
    reference = copy.deepcopy(optimized)
    ids = _ids(length=128)
    expected = _run(reference, ids, compiled=False, passes=1, gradient_passes=1)
    actual = _run(optimized, ids, compiled=True, passes=1, gradient_passes=1)
    _compare(actual, expected)

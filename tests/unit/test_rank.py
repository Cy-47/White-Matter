"""Architecture and recipe checks for the sequential Figure 7b suite."""

import copy
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from studies.rank.depth_causal import register_model as register_depth_causal
from studies.rank.evaluate import validate_checkpoint
from studies.rank.protocol import validate_recipe
from training.forward import TrainingForward
from training.optim import partition_optimizer_parameters
from training.recipes import load_recipe
from white_matter.models import register_models
from white_matter.models.white_matter.configuration_white_matter import WhiteMatterConfig
from white_matter.modules.routing import Router, _init_logits

ROOT = Path("studies/rank/recipes")
EXPECTED_COUNTS = {
    "k1": 125_439_616,
    "k2": 125_866_752,
    "k4": 126_721_024,
    "k8": 128_429_568,
    "k12": 130_138_112,
    "k16": 131_846_656,
    "k1_static": 125_308_544,
    "k16_static": 129_749_504,
    "k16_depth_causal": 131_846_656,
}


def test_top_prior_and_static_router_preserve_bias_gradients():
    expected = torch.zeros(1, 16)
    expected[0, -1] = 0.25
    torch.testing.assert_close(_init_logits(1, 16, "top:0.25"), expected)
    dynamic = Router(16, 8, 1, router_prior="top:0.25", layer_stride=2)
    static = Router(16, 8, 1, router_prior="top:0.25", layer_stride=2, dynamic=False)
    static.load_state_dict(dynamic.state_dict())
    assert static.linear.weight.requires_grad is False
    assert static.linear.bias.requires_grad is True
    source = torch.randn(2, 3, 16, 8, requires_grad=True)
    other = source.detach().clone().requires_grad_()
    got_dynamic = dynamic(source)
    got_static = static(other)
    torch.testing.assert_close(got_dynamic, got_static)
    probe = torch.randn_like(got_dynamic)
    dynamic_grads = torch.autograd.grad(
        (got_dynamic * probe).sum(), (source, dynamic.linear.weight, dynamic.linear.bias)
    )
    static_bias_grad = torch.autograd.grad((got_static * probe).sum(), static.linear.bias)
    torch.testing.assert_close(dynamic_grads[0], torch.zeros_like(dynamic_grads[0]))
    torch.testing.assert_close(dynamic_grads[2], static_bias_grad[0])
    assert torch.count_nonzero(dynamic_grads[1]) > 0


def test_all_sequential_rank_recipes_match_paper_parameter_counts():
    register_models()
    register_depth_causal()
    recipes = {path.stem: load_recipe(path) for path in ROOT.glob("*.yaml")}
    assert set(recipes) == {*EXPECTED_COUNTS, "vanilla"}
    reference = recipes["k16"]
    for arm, recipe in recipes.items():
        assert (recipe.steps, recipe.global_batch_size, recipe.gradient_accumulation_steps, recipe.seed) == (
            20_000,
            8,
            1,
            1337,
        )
        assert recipe.data == reference.data
        assert recipe.optimizer == reference.optimizer
        if arm not in {"vanilla", "k16_depth_causal"}:
            assert (recipe.no_gradient_passes, recipe.gradient_passes) == (1, 2)
            assert (recipe.model.num_passes, recipe.model.cyclic_groups, recipe.model.router_layer_stride) == (3, 8, 2)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(recipe.model)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if arm in EXPECTED_COUNTS:
            assert count == EXPECTED_COUNTS[arm]
        recipe.model.training_step = 20_000
        recipe.model.training_sequence_length = 2048
        recipe.model.recipe_name = recipe.name
        assert validate_checkpoint(recipe.model) == (arm, 1 if arm in {"vanilla", "k16_depth_causal"} else 3)


def _small_config(**extra):
    return WhiteMatterConfig(
        vocab_size=101,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=10_000.0,
        eos_token_id=100,
        document_separator_token_id=100,
        num_kv_channels=2,
        num_passes=3,
        cyclic_groups=2,
        router_layer_stride=1,
        router_prior="shifted_identity:0.25",
        **extra,
    )


def test_static_model_bias_is_optimizer_owned_and_dynamic_weight_is_absent():
    register_models()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(_small_config(router_dynamic=False))
    router = model.model.decoder.block.kv_pool.mixer.k_router
    assert router.linear.weight.requires_grad is False
    assert router.linear.bias.requires_grad is True
    decay, no_decay, muon = partition_optimizer_parameters(model)
    assert any(parameter is router.linear.bias for parameter in no_decay)
    assert all(parameter is not router.linear.weight for parameter in (*decay, *no_decay, *muon))


@pytest.mark.parametrize("dummy", [False, True])
def test_jacobi_model_trains_on_packed_documents_and_prefill_layout(dummy):
    register_models()
    model = AutoModelForCausalLM.from_config(
        _small_config(execution_mode="jacobi", prefill_mode="jacobi", use_dummy_token=dummy)
    )
    ids = torch.tensor([[1, 2, 100, 3, 4]])
    loss = model(ids, labels=ids).loss
    loss.backward()
    router = model.model.decoder.block.kv_pool.mixer.k_router.linear
    assert router.weight.grad is not None
    assert torch.isfinite(router.weight.grad).all()
    model.eval()
    with torch.inference_mode():
        prefix = model(ids[:, :3], use_cache=True)
        assert prefix.past_key_values.get_seq_length() == 3
        assert prefix.past_key_values.layers[0].keys.shape[-2] == 3 + int(dummy)
        continuation = model(ids[:, 3:], past_key_values=prefix.past_key_values)
        assert continuation.logits.shape == (1, 2, 101)


def test_jacobi_checkpoint_recomputation_matches_full_reference_gradients():
    register_models()
    reference = AutoModelForCausalLM.from_config(_small_config(execution_mode="jacobi"))
    checkpointed = AutoModelForCausalLM.from_config(
        _small_config(execution_mode="jacobi", checkpoint_jacobi_passes=True)
    )
    checkpointed.load_state_dict(copy.deepcopy(reference.state_dict()))
    ids = torch.tensor([[1, 100, 2, 3, 100, 4], [5, 6, 100, 7, 8, 100]])

    def run(model):
        model.zero_grad(set_to_none=True)
        inputs = model.get_input_embeddings()(ids)
        inputs.retain_grad()
        runner = TrainingForward(model)
        output = runner(inputs, 3, 2, token_ids=ids, compute_ce=False)
        loss = runner(inputs, 3, 2, token_ids=ids, compute_ce=True)
        loss.backward()
        gradients = {
            name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        assert all(gradient is not None for gradient in gradients.values())
        return output.detach(), loss.detach(), inputs.grad.detach(), gradients

    expected = run(reference)
    actual = run(checkpointed)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=1e-5, atol=1e-6)
    assert actual[3].keys() == expected[3].keys()
    for name in expected[3]:
        torch.testing.assert_close(actual[3][name], expected[3][name], rtol=1e-5, atol=1e-6, msg=name)


def test_rank_checkpoint_rejects_partial_training():
    recipe = load_recipe(ROOT / "k1.yaml")
    recipe.model.training_step = 2_000
    recipe.model.training_sequence_length = 2048
    recipe.model.recipe_name = recipe.name
    with pytest.raises(ValueError, match="final"):
        validate_checkpoint(recipe.model)


def test_rank_recipe_rejects_mislabeled_channel_count_and_optimizer():
    path = ROOT / "k1.yaml"
    recipe = load_recipe(path)
    recipe.model.num_kv_channels = 16
    with pytest.raises(ValueError, match="num_kv_channels"):
        validate_recipe(recipe, path)
    recipe = load_recipe(path)
    from dataclasses import replace

    changed = replace(recipe, optimizer=replace(recipe.optimizer, learning_rate=0.0004))
    with pytest.raises(ValueError, match="optimizer"):
        validate_recipe(changed, path)
    with pytest.raises(ValueError, match="training backend"):
        validate_recipe(replace(recipe, loss_backend="cce"), path)

"""Independent arithmetic and loading checks for the shared-mixture ablation."""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM

from studies.shared_mixture.evaluate_heldout import evaluate_three_pass, validate_checkpoint
from studies.shared_mixture.model import SharedMixtureConfig, SharedMixtureKVPool, register_model
from studies.protocol import PAPER_TEST_TARGETS, validate_paper_cache
from training.checkpoint import training_source_sha256
from training.optim import partition_optimizer_parameters
from training.recipes import load_recipe
from white_matter.modules import RotaryEmbedding


def _manual_projection(pool, stacked):
    B, T, L, D = stacked.shape
    source = F.rms_norm(stacked, (D,), None, pool.rms_norm_eps)
    source = source * pool.pre_mix_weight[None, None].to(source.dtype)
    selected = source[:, :, pool.mixer.router.source_start :: pool.mixer.router.layer_stride]
    router_input = selected.reshape(B, T, -1)
    weights = F.linear(
        router_input, pool.mixer.router.linear.weight, pool.mixer.router.linear.bias,
    ).reshape(B, T, L)
    mixed = (source * weights[..., None]).sum(dim=2)
    mixed = F.rms_norm(mixed, (D,), None, pool.mix_norm_eps)
    keys = torch.stack([
        F.linear(mixed * pool.post_mix["k_gain"][channel], pool.k_proj_weight[channel])
        for channel in range(pool.num_kv_channels)
    ], dim=1)
    values = torch.stack([
        F.linear(mixed * pool.post_mix["v_gain"][channel], pool.v_proj_weight[channel])
        for channel in range(pool.num_kv_channels)
    ], dim=1)
    return tuple(
        tensor.reshape(B, pool.num_kv_channels, T, pool.num_key_value_heads, pool.head_dim)
        .permute(0, 1, 3, 2, 4).contiguous()
        for tensor in (keys, values)
    )


def test_one_mixture_feeds_independent_kv_pairs_and_all_gradients():
    torch.manual_seed(420)
    pool = SharedMixtureKVPool(32, 2, 8, 16, 16, router_layer_stride=2).double()
    with torch.no_grad():
        pool.pre_mix_weight.add_(torch.randn_like(pool.pre_mix_weight) * 0.04)
        pool.mixer.router.linear.weight.normal_(std=0.01)
        for parameter in pool.post_mix.values():
            parameter.add_(torch.randn_like(parameter) * 0.07)
    assert pool.mixer.router.linear.out_features == 16  # one row across sixteen layers
    assert not hasattr(pool, "pre_mix_k_weight")
    assert not hasattr(pool, "pre_mix_v_weight")
    assert pool.k_proj_weight.shape[0] == pool.v_proj_weight.shape[0] == 16
    source = torch.randn(2, 5, 16, 32, dtype=torch.float64, requires_grad=True)
    rope = RotaryEmbedding(8)(source, torch.arange(5).expand(2, -1))
    actual = pool.project_sequence(source, rope)
    expected_key, expected_value = _manual_projection(pool, source)
    normed_key = F.rms_norm(expected_key, (pool.head_dim,), None, pool.rms_norm_eps)
    normed_key = normed_key * pool.k_norm_weight[None, :, None, None, :]
    cos, sin = (part[:, None, None] for part in rope)
    rotated_half = torch.cat((-normed_key[..., 4:], normed_key[..., :4]), dim=-1)
    expected = (normed_key * cos + rotated_half * sin, expected_value)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)
    probes = [torch.randn_like(tensor) for tensor in actual]
    actual_loss = sum((tensor * probe).sum() for tensor, probe in zip(actual, probes, strict=True))
    expected_loss = sum((tensor * probe).sum() for tensor, probe in zip(expected, probes, strict=True))
    parameters = dict(pool.named_parameters())
    actual_grads = torch.autograd.grad(actual_loss, (source, *parameters.values()), retain_graph=True)
    expected_grads = torch.autograd.grad(expected_loss, (source, *parameters.values()))
    for name, got, want in zip(("source", *parameters), actual_grads, expected_grads, strict=True):
        assert got is not None, name
        torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-9, msg=name)
    # Independent projections must actually yield different stored pairs.
    assert not torch.allclose(actual[0][:, 0], actual[0][:, 1])
    assert not torch.allclose(actual[1][:, 0], actual[1][:, 1])


def test_study_recipe_is_trainable_and_matched_to_control():
    register_model()
    control = load_recipe("studies/rank/recipes/k16.yaml")
    shared = load_recipe("studies/shared_mixture/recipes/shared_k16.yaml")
    assert shared.model.model_type == SharedMixtureConfig.model_type
    assert (shared.steps, shared.global_batch_size, shared.data.sequence_length) == (20_000, 8, 2048)
    assert (shared.no_gradient_passes, shared.gradient_passes) == (1, 2)
    assert shared.steps == control.steps and shared.global_batch_size == control.global_batch_size
    assert shared.seed == control.seed and shared.optimizer == control.optimizer
    assert shared.model.num_kv_channels == control.model.num_kv_channels == 16
    assert shared.model.router_prior == "cyclic:0.25"
    assert control.model.router_prior == "shifted_identity:0.25"
    assert shared.data == control.data
    assert shared.gradient_accumulation_steps == control.gradient_accumulation_steps == 1
    assert (shared.model.num_passes, shared.model.cyclic_groups, shared.model.router_layer_stride) == (3, 8, 2)
    assert (control.model.num_passes, control.model.cyclic_groups, control.model.router_layer_stride) == (3, 8, 2)
    assert training_source_sha256(model_config=shared.model) != training_source_sha256(model_config=control.model)


def test_paper_shape_parameter_counts():
    register_model()
    expected = {"control": 131_846_656, "shared": 129_806_352}
    for arm, count in expected.items():
        path = ("studies/rank/recipes/k16.yaml" if arm == "control"
                else "studies/shared_mixture/recipes/shared_k16.yaml")
        config = load_recipe(path).model
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == count
        assert model.get_input_embeddings().weight.numel() == 77_791_232


def test_study_model_round_trips_with_registration(tmp_path):
    register_model()
    cfg = AutoConfig.for_model(
        SharedMixtureConfig.model_type,
        vocab_size=101, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, rope_theta=10_000.0,
        eos_token_id=100, document_separator_token_id=100,
        num_kv_channels=2, num_passes=2, cyclic_groups=2, prefill_mode="cyclic",
    )
    model = AutoModelForCausalLM.from_config(cfg).eval()
    _, no_decay, muon = partition_optimizer_parameters(model)
    shared_gain = model.model.decoder.block.kv_pool.pre_mix_weight
    assert any(parameter is shared_gain for parameter in no_decay)
    assert all(parameter is not shared_gain for parameter in muon)
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert type(loaded) is type(model)
    assert loaded.config.model_type == SharedMixtureConfig.model_type
    assert loaded.lm_head.weight is loaded.get_input_embeddings().weight
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], weight, rtol=0, atol=0)
    ids = torch.tensor([[1, 2, 100, 3, 4]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids).logits, model(ids).logits)
        original = model(ids[:, :3], use_cache=True)
        reloaded = loaded(ids[:, :3], use_cache=True)
        torch.testing.assert_close(original.logits, reloaded.logits)
        torch.testing.assert_close(
            model(ids[:, 3:], past_key_values=original.past_key_values).logits,
            loaded(ids[:, 3:], past_key_values=reloaded.past_key_values).logits,
        )


def test_study_rejects_non_full_rank_or_incompatible_prior():
    base = dict(
        vocab_size=101, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, eos_token_id=100,
    )
    with pytest.raises(ValueError, match="one KV pair"):
        SharedMixtureConfig(**base, num_kv_channels=2)
    with pytest.raises(ValueError, match="equal-source"):
        SharedMixtureConfig(**base, num_kv_channels=4, router_prior="shifted_identity:0.25")


def test_three_pass_heldout_scores_only_the_final_state():
    register_model()
    cfg = SharedMixtureConfig(
        vocab_size=101, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, eos_token_id=100, document_separator_token_id=100,
        num_passes=3, cyclic_groups=2, prefill_mode="cyclic",
    )
    model = AutoModelForCausalLM.from_config(cfg).eval()
    ids = torch.tensor([[1, 2, 100, 3, 4], [5, 100, 6, 7, 8]])
    loader = DataLoader([{"input_ids": row} for row in ids], batch_size=2)
    loss_sum, targets = evaluate_three_pass(model, loader)
    assert targets == 8
    with torch.inference_mode():
        logits = model(ids, num_passes=3).logits[:, :-1]
        expected = F.cross_entropy(logits.float().reshape(-1, 101), ids[:, 1:].reshape(-1))
        assert loss_sum / targets == pytest.approx(float(expected), abs=1e-5)


def test_paper_protocol_rejects_other_cache_and_checkpoint(tmp_path):
    assert PAPER_TEST_TARGETS == 10_235_000
    with pytest.raises(FileNotFoundError):
        validate_paper_cache(tmp_path)
    with pytest.raises(ValueError, match="training_step"):
        validate_checkpoint(SharedMixtureConfig(num_kv_channels=16))
    register_model()
    for filename, name in (
        ("studies/rank/recipes/k16.yaml", "rank_k16"),
        ("studies/shared_mixture/recipes/shared_k16.yaml", "shared_mixture_k16"),
    ):
        cfg = load_recipe(filename).model
        cfg.training_step = 20_000
        cfg.training_sequence_length = 2_048
        cfg.recipe_name = name
        validate_checkpoint(cfg)

"""Closed Figure 7a matrix and fixed-pass scoring checks."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from studies.paper_20k import evaluate_fixed_passes
from studies.schedules_20k.evaluate import validate_checkpoint
from studies.schedules_20k.matrix import GRAD, MODES, NO_GRAD, SEEDS, recipe_path, validate_recipe
from training.recipes import load_recipe
from white_matter.models import register_models
from white_matter.models.white_matter.configuration_white_matter import WhiteMatterConfig


def test_schedule_matrix_contains_exactly_48_valid_paper_recipes():
    register_models()
    paths = {path.resolve() for path in Path("studies/schedules_20k/recipes").glob("seed*/*.yaml")}
    assert len(paths) == len(SEEDS) * len(NO_GRAD) * len(GRAD) * len(MODES) == 48
    for seed in SEEDS:
        for no_grad in NO_GRAD:
            for grad in GRAD:
                for mode in MODES:
                    arm = f"ng{no_grad}_g{grad}_{mode}"
                    path = recipe_path(seed, arm)
                    assert path in paths
                    recipe = load_recipe(path)
                    assert validate_recipe(recipe, path) == (seed, arm)
                    recipe.model.training_step = 20_000
                    recipe.model.training_sequence_length = 2048
                    recipe.model.recipe_name = recipe.name
                    assert validate_checkpoint(recipe.model) == (seed, arm)


def test_schedule_recipe_rejects_changed_model_size():
    path = recipe_path(1337, "ng1_g1_tp")
    recipe = load_recipe(path)
    recipe.model.hidden_size = 1792
    with pytest.raises(ValueError, match="hidden_size"):
        validate_recipe(recipe, path)


def test_fixed_pass_scorer_matches_direct_lm_ce_for_cyclic_and_jacobi():
    register_models()
    ids = torch.tensor([[1, 2, 100, 3, 4], [5, 100, 6, 7, 8]])
    loader = DataLoader([{"input_ids": row} for row in ids], batch_size=2)
    config = WhiteMatterConfig(
        vocab_size=101, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, rope_theta=10_000.0,
        eos_token_id=100, document_separator_token_id=100,
        num_kv_channels=2, num_passes=2, cyclic_groups=2,
        router_layer_stride=1, router_prior="shifted_identity:0.25",
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    for mode in ("cyclic", "jacobi"):
        model.config.execution_mode = mode
        for passes in (1, 2):
            loss_sum, targets = evaluate_fixed_passes(model, loader, num_passes=passes)
            assert targets == 8
            with torch.inference_mode():
                logits = model(ids, num_passes=passes).logits[:, :-1]
                expected = F.cross_entropy(logits.float().reshape(-1, 101), ids[:, 1:].reshape(-1))
            assert loss_sum / targets == pytest.approx(float(expected), abs=1e-5)

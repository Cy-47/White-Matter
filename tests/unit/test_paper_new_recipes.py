"""The new paper comparison changes only its two declared model settings."""

import copy
from pathlib import Path

import pytest
import yaml

from evals.paper import PAPER_MODELS, validate_checkpoint
from training.recipes import load_recipe

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / "recipes/paper_new"


def test_paper_new_recipes_preserve_matched_training_controls():
    assert {p.stem for p in RECIPES.glob("*.yaml")} == set(PAPER_MODELS)
    for name in PAPER_MODELS:
        path = RECIPES / f"{name}.yaml"
        expected = yaml.safe_load((ROOT / "recipes/paper" / path.name).read_text())
        if expected["model"]["model_type"] == "white_matter":
            expected["model"].update(use_dummy_token=False, include_top_output=True)
        assert yaml.safe_load(path.read_text()) == expected
        recipe = load_recipe(path)
        config = copy.deepcopy(recipe.model)
        config.training_step = recipe.steps
        config.training_sequence_length = recipe.data.sequence_length
        validate_checkpoint(config, recipe)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("use_dummy_token", True),
        ("include_top_output", False),
        ("num_kv_channels", 16),
        ("training_step", 1),
        ("training_sequence_length", 1024),
    ],
)
def test_quality_collection_rejects_incompatible_checkpoints(key, value):
    recipe = load_recipe(RECIPES / "white_matter_k8.yaml")
    config = copy.deepcopy(recipe.model)
    config.training_step = recipe.steps
    config.training_sequence_length = recipe.data.sequence_length
    setattr(config, key, value)
    with pytest.raises(ValueError, match=key):
        validate_checkpoint(config, recipe)

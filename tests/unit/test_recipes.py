from pathlib import Path
from dataclasses import replace

import pytest

from training.recipes import load_recipe, per_rank_batch_size

PAPER_RECIPES = Path(__file__).parents[2] / "recipes" / "paper"
ANALYSIS_RECIPES = Path(__file__).parents[2] / "recipes" / "analysis"


def test_all_paper_recipes_are_strict_and_self_contained() -> None:
    recipes = {path.name: load_recipe(path) for path in PAPER_RECIPES.glob("*.yaml")}
    assert set(recipes) == {
        "white_matter_k8.yaml",
        "white_matter_k16.yaml",
        "vanilla_16l.yaml",
        "vanilla_24l.yaml",
        "lckv_w4.yaml",
        "lckv_w7.yaml",
        "lckv_w13_1p3b.yaml",
        "fusedkv.yaml",
        "white_matter_1p3b.yaml",
        "vanilla_1p3b.yaml",
    }
    assert recipes["white_matter_k8.yaml"].no_gradient_passes == 1
    assert recipes["white_matter_k8.yaml"].gradient_passes == 2
    assert recipes["white_matter_k16.yaml"].global_batch_size == 128
    assert recipes["white_matter_k16.yaml"].gradient_accumulation_steps == 2
    assert all(recipe.model.document_separator_token_id == recipe.data.eos_token_id for recipe in recipes.values())


def test_unknown_recipe_option_is_rejected(tmp_path: Path) -> None:
    text = (PAPER_RECIPES / "white_matter_k8.yaml").read_text()
    path = tmp_path / "bad.yaml"
    path.write_text(text + "research_only_switch: true\n")
    with pytest.raises(ValueError, match="unknown recipe keys"):
        load_recipe(path)


def test_unknown_model_option_is_rejected(tmp_path: Path) -> None:
    text = (PAPER_RECIPES / "white_matter_k8.yaml").read_text()
    path = tmp_path / "bad_model.yaml"
    path.write_text(text.replace("  vocab_size:", "  router_form: invalid_value\n  vocab_size:"))
    with pytest.raises(ValueError, match="unknown model keys.*router_form"):
        load_recipe(path)


def test_exact_ar_control_has_one_fixed_training_backend() -> None:
    recipe = load_recipe(ANALYSIS_RECIPES / "exact_ar_4l.yaml")
    assert recipe.model.execution_mode == "autoregressive"
    assert per_rank_batch_size(recipe, 1) == 96


def test_cli_recipe_path_is_relative_to_working_directory(monkeypatch):
    from training.train import parse_args

    root = PAPER_RECIPES.parents[1]
    monkeypatch.chdir(root)
    args = parse_args(["--recipe", "recipes/paper/white_matter_k8.yaml", "--data-dir", "cache", "--output-dir", "run"])
    assert args.recipe == (PAPER_RECIPES / "white_matter_k8.yaml").resolve()
    assert load_recipe(args.recipe).name


def test_global_batch_size_keeps_its_effective_batch_meaning():
    recipe = load_recipe(PAPER_RECIPES / "white_matter_k16.yaml")
    assert per_rank_batch_size(recipe, 8) == 8
    assert per_rank_batch_size(recipe, 8) * 8 * recipe.gradient_accumulation_steps == recipe.global_batch_size


def test_optimization_options_are_validated_and_fingerprinted():
    recipe = load_recipe(PAPER_RECIPES / "white_matter_k8.yaml")
    assert replace(recipe, loss_backend="cce").sha256 != recipe.sha256
    assert replace(recipe, optimizer=replace(recipe.optimizer, distributed_muon=True)).sha256 != recipe.sha256
    with pytest.raises(ValueError, match="loss_backend"):
        replace(recipe, loss_backend="unknown")
    with pytest.raises(ValueError, match="autoregressive"):
        replace(recipe, ar_cuda_graph=True)
    with pytest.raises(ValueError, match="boolean"):
        replace(recipe.optimizer, distributed_muon="yes")

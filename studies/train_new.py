"""Train the new paper's analysis models on the continuous-source cache."""

from pathlib import Path

import numpy as np

from training.data import load_cache_metadata
from training.recipes import load_recipe

ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = ROOT / "recipes/paper_new"


def load_analysis_recipe(path):
    from studies.rank.depth_causal import register_model as register_depth
    from studies.shared_mixture.model import register_model as register_shared

    register_depth()
    register_shared()
    relative = Path(path).resolve().relative_to(RECIPE_DIR)
    study, *parts = relative.parts
    if study not in {"rank", "schedules", "shared_mixture"}:
        raise ValueError("expected a rank, schedules, or shared-mixture recipe")
    original = load_recipe(ROOT / "studies" / study / "recipes" / Path(*parts))
    if original.model.model_type != "vanilla":
        original.model.use_dummy_token = False
        original.model.include_top_output = original.model.model_type != "white_matter_depth_causal"
    recipe = load_recipe(path)
    if recipe.sha256 != original.sha256:
        raise ValueError("analysis recipe differs from the matched no-dummy/L+1 protocol")
    return recipe


def validate_cache(path):
    metadata = load_cache_metadata(path)
    if metadata.get("build", {}).get("ordering_protocol") != "continuous-source-v1":
        raise ValueError("new analysis runs require continuous-source-v1 cache order")
    if metadata["splits"] != {"n_train": 9_765_625, "n_val": 2000, "n_test": 5000}:
        raise ValueError("new analysis cache split sizes differ from the protocol")
    tokens = np.load(Path(path) / "tokenized.npy", mmap_mode="r")
    if tokens.shape != (9_772_625, 2048) or tokens.dtype != np.int32:
        raise ValueError("expected int32 cache with 9,772,625 rows of 2048 tokens")


def main():
    from training.engine import train
    from training.train import parse_args

    args = parse_args()
    recipe = load_analysis_recipe(args.recipe)
    validate_cache(args.data_dir)
    train(recipe, args)


if __name__ == "__main__":
    main()

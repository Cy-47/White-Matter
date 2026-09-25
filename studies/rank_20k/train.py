"""Train one sequential-data Figure 7b recipe on the paper cache."""

from pathlib import Path

from studies.paper_20k import PAPER_SMALL_MODEL, validate_paper_cache, validate_paper_training_recipe


RECIPE_DIR = Path(__file__).resolve().parent / "recipes"
ARMS = frozenset({
    "k1", "k2", "k4", "k8", "k12", "k16", "k1_static", "k16_static", "vanilla", "k16_depth_causal",
})


def validate_recipe(recipe, path: str | Path) -> str:
    path = Path(path).resolve()
    if path.parent != RECIPE_DIR or path.stem not in ARMS:
        raise ValueError(f"Figure 7b recipe must be one of {sorted(ARMS)} in {RECIPE_DIR}")
    if recipe.name != f"rank20k_{path.stem}":
        raise ValueError("Figure 7b recipe name does not match its arm")
    if (recipe.steps, recipe.global_batch_size, recipe.gradient_accumulation_steps, recipe.seed) != (20_000, 8, 1, 1337):
        raise ValueError("Figure 7b training schedule differs from the paper")
    if recipe.data.sequence_length != 2048:
        raise ValueError("Figure 7b requires 2048-token cache rows")
    if path.stem == "vanilla":
        from white_matter.models.vanilla import VanillaConfig

        expected_model = VanillaConfig(**PAPER_SMALL_MODEL)
        expected_passes = (None, None)
    elif path.stem == "k16_depth_causal":
        from studies.rank_20k.depth_causal import DepthCausalConfig

        expected_model = DepthCausalConfig(
            **PAPER_SMALL_MODEL, num_kv_channels=16, num_passes=1,
            router_prior="identity:0.25",
        )
        expected_passes = (0, 1)
    else:
        from white_matter.models.white_matter import WhiteMatterConfig

        channels = int(path.stem.split("_")[0][1:])
        prior = "top:0.25" if channels == 1 else ("shifted_identity:0.25" if channels == 16 else "cyclic:0.25")
        expected_model = WhiteMatterConfig(
            **PAPER_SMALL_MODEL, num_kv_channels=channels, num_passes=3,
            prefill_mode="cyclic", router_prior=prior,
            router_dynamic=not path.stem.endswith("_static"),
        )
        expected_passes = (1, 2)
    validate_paper_training_recipe(recipe, expected_model)
    if (recipe.no_gradient_passes, recipe.gradient_passes) != expected_passes:
        raise ValueError("Figure 7b gradient pass counts differ from the arm")
    return path.stem


def main() -> None:
    from studies.rank_20k.depth_causal import register_model
    from training.recipes import load_recipe
    from training.train import parse_args
    from training.engine import train

    args = parse_args()
    register_model()
    recipe = load_recipe(args.recipe)
    validate_recipe(recipe, args.recipe)
    validate_paper_cache(args.data_dir)
    train(recipe, args)


if __name__ == "__main__":
    main()

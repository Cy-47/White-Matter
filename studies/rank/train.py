"""Train one sequential-data Figure 7b recipe on the paper cache."""

from studies.protocol import validate_paper_cache
from studies.rank.protocol import validate_recipe


def main() -> None:
    from studies.rank.depth_causal import register_model
    from training.engine import train
    from training.recipes import load_recipe
    from training.train import parse_args

    args = parse_args()
    register_model()
    recipe = load_recipe(args.recipe)
    validate_recipe(recipe, args.recipe)
    validate_paper_cache(args.data_dir)
    train(recipe, args)


if __name__ == "__main__":
    main()

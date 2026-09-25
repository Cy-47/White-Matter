"""Train one Figure 7a schedule cell with sequential cache rows."""

from studies.paper_20k import validate_paper_cache
from studies.schedules_20k.matrix import validate_recipe


def main() -> None:
    from training.recipes import load_recipe
    from training.train import parse_args
    from training.engine import train

    args = parse_args()
    recipe = load_recipe(args.recipe)
    validate_recipe(recipe, args.recipe)
    validate_paper_cache(args.data_dir)
    train(recipe, args)


if __name__ == "__main__":
    main()

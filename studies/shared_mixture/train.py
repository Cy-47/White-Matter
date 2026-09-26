"""Train a registered shared-mixture study recipe."""

from studies.shared_mixture.model import register_model
from studies.protocol import validate_paper_cache


def main() -> None:
    register_model()
    from training.recipes import load_recipe
    from training.train import parse_args
    from training.engine import train

    args = parse_args()
    validate_paper_cache(args.data_dir)
    train(load_recipe(args.recipe), args)


if __name__ == "__main__":
    main()

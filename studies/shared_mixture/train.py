"""Train a registered shared-mixture study recipe."""

from studies.protocol import validate_paper_cache
from studies.shared_mixture.model import register_model


def main() -> None:
    register_model()
    from training.engine import train
    from training.recipes import load_recipe
    from training.train import parse_args

    args = parse_args()
    validate_paper_cache(args.data_dir)
    train(load_recipe(args.recipe), args)


if __name__ == "__main__":
    main()

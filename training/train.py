"""Command-line entry point for one strict WhiteMatter training recipe."""

import argparse
from pathlib import Path

from training.recipes import load_recipe


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train one WhiteMatter recipe.")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.log_every <= 0 or args.save_every < 0:
        parser.error("--log-every must be positive; --save-every must be nonnegative")
    args.recipe = args.recipe.expanduser().resolve()
    return args


def main():
    args = parse_args()
    recipe = load_recipe(args.recipe)
    from training.engine import train

    train(recipe, args)


if __name__ == "__main__":
    main()

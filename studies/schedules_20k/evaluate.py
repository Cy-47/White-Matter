"""Full-test pass curves for Figure 7a, separate from Figure 7b scoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from studies.paper_20k import (
    PAPER_SEQUENCE_LENGTH, PAPER_TEST_SEQUENCES, PAPER_TEST_TARGETS,
    evaluate_fixed_passes, validate_paper_cache,
)
from studies.schedules_20k.matrix import arm_values, recipe_path, validate_recipe
from training.data import TokenCacheDataset
from training.precision import configure_precision
from training.recipes import load_recipe, model_recipe_keys
from white_matter.models import register_models


def validate_checkpoint(config) -> tuple[int, str]:
    name = getattr(config, "recipe_name", None)
    if not isinstance(name, str) or not name.startswith("schedules20k_seed"):
        raise ValueError("checkpoint is not from the Figure 7a suite")
    suffix = name.removeprefix("schedules20k_seed")
    seed_text, separator, arm = suffix.partition("_")
    if not separator or not seed_text.isdecimal():
        raise ValueError("malformed Figure 7a checkpoint recipe name")
    seed = int(seed_text)
    arm_values(arm)
    recipe = load_recipe(recipe_path(seed, arm))
    validate_recipe(recipe, recipe_path(seed, arm))
    expected = recipe.model
    if config.model_type != expected.model_type:
        raise ValueError("checkpoint family differs from its Figure 7a recipe")
    for key in model_recipe_keys(type(expected)):
        if key != "model_type" and getattr(config, key, None) != getattr(expected, key, None):
            raise ValueError(f"checkpoint {key} differs from its Figure 7a recipe")
    if getattr(config, "training_step", None) != 20_000:
        raise ValueError("checkpoint must be the final 20,000-step model")
    if getattr(config, "training_sequence_length", None) != PAPER_SEQUENCE_LENGTH:
        raise ValueError("checkpoint must have been trained on 2048-token rows")
    return seed, arm


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one Figure 7a held-out pass curve.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "cyclic16", "tp"), required=True)
    parser.add_argument("--first-pass", type=int, default=1)
    parser.add_argument("--last-pass", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.first_pass < 1 or args.last_pass < args.first_pass or args.last_pass > 32:
        parser.error("pass range must be inside 1..32")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    register_models()
    validate_paper_cache(args.data_dir)
    from evals.loading import load_complete_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_precision(device)
    model = load_complete_model(args.model, dtype=torch.float32).to(device).eval()
    seed, arm = validate_checkpoint(model.config)
    if args.compile and device.type == "cuda":
        from training.compile import compile_feedback

        compile_feedback(model, mode="default")
    dataset = TokenCacheDataset(
        args.data_dir, split="test", sequence_length=PAPER_SEQUENCE_LENGTH,
        n_train=9_765_625, n_val=2_000, n_test=PAPER_TEST_SEQUENCES,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == "cuda",
    )
    original_mode, original_groups = model.config.execution_mode, model.config.cyclic_groups
    if args.mode == "cyclic16":
        model.config.execution_mode, model.config.cyclic_groups = "cyclic", 16
    elif args.mode == "tp":
        model.config.execution_mode = "jacobi"
    rows = []
    try:
        for passes in range(args.first_pass, args.last_pass + 1):
            loss_sum, targets = evaluate_fixed_passes(model, loader, num_passes=passes)
            if targets != PAPER_TEST_TARGETS:
                raise RuntimeError(f"expected {PAPER_TEST_TARGETS} prediction targets, got {targets}")
            rows.append({"n_passes": passes, "lm_ce": loss_sum / targets, "perplexity": math.exp(loss_sum / targets)})
            print(json.dumps({"arm": arm, "mode": args.mode, **rows[-1]}), flush=True)
    finally:
        model.config.execution_mode, model.config.cyclic_groups = original_mode, original_groups
    result = {
        "protocol": "figure7a_sequential_final",
        "seed": seed,
        "arm": arm,
        "checkpoint": str(args.model),
        "evaluation_mode": args.mode,
        "n_seq": len(dataset),
        "n_tok": PAPER_TEST_TARGETS,
        "T": PAPER_SEQUENCE_LENGTH,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

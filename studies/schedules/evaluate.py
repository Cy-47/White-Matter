"""Full-test pass curves for Figure 7a, separate from Figure 7b scoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from studies.protocol import (
    PAPER_SEQUENCE_LENGTH,
    PAPER_TEST_TARGETS,
    evaluate_fixed_passes,
    paper_test_loader,
    validate_final_checkpoint,
    validate_paper_cache,
)
from studies.schedules.matrix import arm_values, evaluation_horizon, recipe_path, validate_recipe
from training.compile import compile_evaluation
from training.precision import configure_precision
from training.recipes import load_recipe
from white_matter.compilation import add_compile_argument, execution_policy
from white_matter.models import register_models


def validate_checkpoint(config) -> tuple[int, str]:
    name = getattr(config, "recipe_name", None)
    if not isinstance(name, str) or not name.startswith("schedules_seed"):
        raise ValueError("checkpoint is not from the Figure 7a suite")
    suffix = name.removeprefix("schedules_seed")
    seed_text, separator, arm = suffix.partition("_")
    if not separator or not seed_text.isdecimal():
        raise ValueError("malformed Figure 7a checkpoint recipe name")
    seed = int(seed_text)
    arm_values(arm)
    recipe = load_recipe(recipe_path(seed, arm))
    validate_recipe(recipe, recipe_path(seed, arm))
    validate_final_checkpoint(config, recipe.model)
    return seed, arm


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one Figure 7a held-out pass curve.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "cyclic16", "tp"), required=True)
    parser.add_argument("--first-pass", type=int, default=1)
    parser.add_argument("--last-pass", type=int, help="default: paper ceiling for this arm and mode (32, 96, or 128)")
    parser.add_argument("--batch-size", type=int, default=16)
    add_compile_argument(parser)
    args = parser.parse_args()
    with execution_policy(args.compile):
        _run(args, parser)


def _run(args, parser):
    if args.first_pass < 1 or (args.last_pass is not None and args.last_pass < args.first_pass):
        parser.error("pass range must be positive and increasing")
    if args.mode != "tp" and (args.last_pass or args.first_pass) > 32:
        parser.error("native and cyclic16 pass ranges must be inside 1..32")
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
    if args.last_pass is None:
        args.last_pass = evaluation_horizon(arm, args.mode)
    if args.first_pass > args.last_pass:
        parser.error("--first-pass exceeds the default observation ceiling; specify --last-pass")
    if args.compile:
        compile_evaluation(model, eager_pass_loop=True)
    loader = paper_test_loader(args.data_dir, batch_size=args.batch_size, device=device)
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
        "compiled": args.compile,
        "protocol": "figure7a_sequential_final",
        "seed": seed,
        "arm": arm,
        "checkpoint": str(args.model),
        "evaluation_mode": args.mode,
        "n_seq": len(loader.dataset),
        "n_tok": PAPER_TEST_TARGETS,
        "T": PAPER_SEQUENCE_LENGTH,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

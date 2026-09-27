"""Strict full-test evaluation for the Figure 7b ablations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from studies.protocol import (
    PAPER_SEQUENCE_LENGTH,
    PAPER_TEST_TARGETS,
    evaluate_fixed_passes,
    paper_test_loader,
    validate_final_checkpoint,
    validate_paper_cache,
)
from studies.rank.protocol import ARMS, RECIPE_DIR, validate_recipe
from training.precision import configure_precision
from training.recipes import load_recipe
from white_matter.models import register_models


def validate_checkpoint(config) -> tuple[str, int]:
    name = getattr(config, "recipe_name", None)
    if not isinstance(name, str) or not name.startswith("rank_"):
        raise ValueError("checkpoint is not from the Figure 7b sequential suite")
    arm = name.removeprefix("rank_")
    if arm not in ARMS:
        raise ValueError(f"unknown Figure 7b arm: {arm}")
    recipe = load_recipe(RECIPE_DIR / f"{arm}.yaml")
    validate_recipe(recipe, RECIPE_DIR / f"{arm}.yaml")
    validate_final_checkpoint(config, recipe.model)
    return arm, 1 if arm in {"vanilla", "k16_depth_causal"} else 3


def trainable_parameter_count(config) -> int:
    """Count the trainable architecture, independent of HF load-time flags."""
    # Transformers 5.10 restores every parameter with requires_grad=True,
    # including the frozen router weights of the static study arms.
    with torch.device("meta"):
        fresh = AutoModelForCausalLM.from_config(config)
    return sum(parameter.numel() for parameter in fresh.parameters() if parameter.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a final sequential Figure 7b checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    register_models()
    from studies.rank.depth_causal import register_model

    register_model()
    validate_paper_cache(args.data_dir)
    from evals.loading import load_complete_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_precision(device)
    model = load_complete_model(args.model, dtype=torch.float32).to(device).eval()
    arm, passes = validate_checkpoint(model.config)
    if args.compile and device.type == "cuda":
        from training.compile import compile_feedback

        compile_feedback(model, mode="default")
    loader = paper_test_loader(args.data_dir, batch_size=args.batch_size, device=device)
    loss_sum, targets = evaluate_fixed_passes(model, loader, num_passes=passes)
    if targets != PAPER_TEST_TARGETS:
        raise RuntimeError(f"expected {PAPER_TEST_TARGETS} prediction targets, got {targets}")
    result = {
        "protocol": "figure7b_sequential_final",
        "arm": arm,
        "checkpoint": str(args.model),
        "recipe": model.config.recipe_name,
        "split": "test",
        "n_seq": len(loader.dataset),
        "n_tok": targets,
        "T": PAPER_SEQUENCE_LENGTH,
        "n_passes": passes,
        "forward_mode": "depth_causal"
        if arm == "k16_depth_causal"
        else ("single_pass" if arm == "vanilla" else "cyclic"),
        "loss": "token-weighted next-token LM cross-entropy (cce_exact on CUDA)",
        "lm_ce": loss_sum / targets,
        "perplexity": math.exp(loss_sum / targets),
        "trainable_parameters": trainable_parameter_count(model.config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"arm": arm, "n_passes": passes, "perplexity": result["perplexity"]}, indent=2))


if __name__ == "__main__":
    main()

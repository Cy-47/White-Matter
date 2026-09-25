"""Strict full-test evaluation for the Figure 7b ablations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from studies.paper_20k import (
    PAPER_SEQUENCE_LENGTH, PAPER_TEST_SEQUENCES, PAPER_TEST_TARGETS,
    evaluate_fixed_passes, validate_paper_cache,
)
from studies.rank_20k.train import ARMS, RECIPE_DIR, validate_recipe
from training.data import TokenCacheDataset
from training.precision import configure_precision
from training.recipes import load_recipe, model_recipe_keys
from white_matter.models import register_models


def validate_checkpoint(config) -> tuple[str, int]:
    name = getattr(config, "recipe_name", None)
    if not isinstance(name, str) or not name.startswith("rank20k_"):
        raise ValueError("checkpoint is not from the Figure 7b sequential suite")
    arm = name.removeprefix("rank20k_")
    if arm not in ARMS:
        raise ValueError(f"unknown Figure 7b arm: {arm}")
    recipe = load_recipe(RECIPE_DIR / f"{arm}.yaml")
    validate_recipe(recipe, RECIPE_DIR / f"{arm}.yaml")
    expected = recipe.model
    if config.model_type != expected.model_type:
        raise ValueError("checkpoint model family differs from its recipe")
    for key in model_recipe_keys(type(expected)):
        if key == "model_type":
            continue
        if getattr(config, key, None) != getattr(expected, key, None):
            raise ValueError(f"checkpoint {key} differs from its Figure 7b recipe")
    if getattr(config, "training_step", None) != 20_000:
        raise ValueError("checkpoint must be the final 20,000-step model")
    if getattr(config, "training_sequence_length", None) != PAPER_SEQUENCE_LENGTH:
        raise ValueError("checkpoint must have been trained on 2048-token rows")
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
    from studies.rank_20k.depth_causal import register_model

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
    dataset = TokenCacheDataset(
        args.data_dir, split="test", sequence_length=PAPER_SEQUENCE_LENGTH,
        n_train=9_765_625, n_val=2_000, n_test=PAPER_TEST_SEQUENCES,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == "cuda",
    )
    loss_sum, targets = evaluate_fixed_passes(model, loader, num_passes=passes)
    if targets != PAPER_TEST_TARGETS:
        raise RuntimeError(f"expected {PAPER_TEST_TARGETS} prediction targets, got {targets}")
    result = {
        "protocol": "figure7b_sequential_final",
        "arm": arm,
        "checkpoint": str(args.model),
        "recipe": model.config.recipe_name,
        "split": "test",
        "n_seq": len(dataset),
        "n_tok": targets,
        "T": PAPER_SEQUENCE_LENGTH,
        "n_passes": passes,
        "forward_mode": "depth_causal" if arm == "k16_depth_causal" else ("single_pass" if arm == "vanilla" else "cyclic"),
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

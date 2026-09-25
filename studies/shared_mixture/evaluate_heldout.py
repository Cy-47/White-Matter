"""Three-pass held-out evaluation for the matched shared-mixture k=16 study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from studies.shared_mixture.model import SharedMixtureConfig, register_model
from studies.paper_20k import (
    PAPER_SEQUENCE_LENGTH,
    PAPER_TEST_SEQUENCES,
    PAPER_TEST_TARGETS,
    validate_paper_cache,
    evaluate_fixed_passes,
)
from training.data import TokenCacheDataset
from training.precision import configure_precision


def validate_checkpoint(config) -> None:
    if config.model_type not in {"white_matter", SharedMixtureConfig.model_type}:
        raise ValueError("three-pass evaluation requires a k=16 control or shared-mixture checkpoint")
    recipe_name, router_prior = (
        ("shared_mixture_k16_20k", "cyclic:0.25")
        if config.model_type == SharedMixtureConfig.model_type
        else ("rank20k_k16", "shifted_identity:0.25")
    )
    expected = {
        "vocab_size": 151_936,
        "num_hidden_layers": 16,
        "num_kv_channels": 16,
        "hidden_size": 512,
        "intermediate_size": 1_536,
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "head_dim": 96,
        "max_position_embeddings": 4_096,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1.0e-6,
        "residual_dtype": "fp32",
        "num_passes": 3,
        "cyclic_groups": 8,
        "router_layer_stride": 2,
        "router_prior": router_prior,
        "num_pre_layers": 0,
        "num_post_layers": 0,
        "execution_mode": "cyclic",
        "training_step": 20_000,
        "training_sequence_length": PAPER_SEQUENCE_LENGTH,
        "recipe_name": recipe_name,
        "eos_token_id": 151_643,
        "document_separator_token_id": 151_643,
        "pad_token_id": None,
    }
    for key, value in expected.items():
        if getattr(config, key, None) != value:
            raise ValueError(f"checkpoint {key} must equal {value!r} for the matched k=16 comparison")


@torch.inference_mode()
def evaluate_three_pass(model, loader: DataLoader) -> tuple[float, int]:
    """Return summed next-token CE and target count at exactly three passes."""
    return evaluate_fixed_passes(model, loader, num_passes=3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the matched k=16 study at three passes.")
    parser.add_argument("--model", required=True, help="Final HF checkpoint directory.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    register_model()
    validate_paper_cache(args.data_dir)
    from evals.loading import load_complete_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_precision(device)
    model = load_complete_model(args.model, dtype=torch.float32).to(device).eval()
    validate_checkpoint(model.config)
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
    loss_sum, targets = evaluate_three_pass(model, loader)
    if targets != PAPER_TEST_TARGETS:
        raise RuntimeError(f"expected {PAPER_TEST_TARGETS} test targets, got {targets}")
    result = {
        "protocol": "matched_k16_three_pass",
        "checkpoint": str(args.model),
        "split": "test",
        "T": PAPER_SEQUENCE_LENGTH,
        "n_seq": len(dataset),
        "n_tok": targets,
        "n_passes": 3,
        "forward_mode": "cyclic",
        "cyclic_groups": model.config.cyclic_groups,
        "loss": "token-weighted held-out next-token LM cross-entropy (cce_exact on CUDA)",
        "lm_ce": loss_sum / targets,
        "perplexity": math.exp(loss_sum / targets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"n_passes": 3, "perplexity": result["perplexity"]}, indent=2))


if __name__ == "__main__":
    main()

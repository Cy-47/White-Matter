"""Evaluate next-token perplexity on the cache's held-out test split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from benchmarks._measurement import checkpoint_files
from evals.execution import evaluate as evaluate_tokens
from evals.loading import load_complete_model
from training.data import TokenCacheDataset, load_cache_metadata
from white_matter.models import register_models
from white_matter.modules.precision import model_autocast_context

register_models()


def evaluate(model, loader, *, logits_chunk=256):
    with model_autocast_context(next(model.parameters()).device):
        return evaluate_tokens(model, loader, chunk_size=logits_chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paper held-out perplexity.")
    parser.add_argument("--model", required=True, help="HF checkpoint directory or Hub ID.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Preserve the paper checkpoint's master weights; GEMMs use BF16 autocast.
    model = load_complete_model(args.model, dtype=torch.float32).to(device)
    splits = load_cache_metadata(args.data_dir)["splits"]
    dataset = TokenCacheDataset(
        args.data_dir,
        split="test",
        sequence_length=int(getattr(model.config, "training_sequence_length", 2_048)),
        n_train=int(splits["n_train"]),
        n_val=int(splits["n_val"]),
        n_test=int(splits["n_test"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    loss_sum, token_count = evaluate(model, loader)
    result = {
        "model": args.model,
        "checkpoint": checkpoint_files(args.model) if Path(args.model).is_dir() else None,
        "model_config": model.config.to_dict(),
        "loss": loss_sum / token_count,
        "perplexity": math.exp(loss_sum / token_count),
        "tokens": token_count,
        "sequences": len(dataset),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

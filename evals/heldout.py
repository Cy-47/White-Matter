"""Evaluate next-token perplexity on the cache's held-out test split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from evals.loading import load_complete_model
from evals.scoring import score_tokens
from training.data import TokenCacheDataset, load_cache_metadata
from training.precision import attention_kernel_context
from white_matter.models import register_models
from white_matter.modules.precision import model_autocast_context

register_models()


@torch.inference_mode()
def evaluate(
    model: PreTrainedModel,
    loader: DataLoader,
    *,
    logits_chunk: int = 256,
) -> tuple[float, int]:
    if logits_chunk < 1:
        raise ValueError("logits_chunk must be positive")
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        with attention_kernel_context(str(device)):
            with model_autocast_context(str(device)):
                hidden = model.model(input_ids=input_ids, return_dict=True).last_hidden_state[:, :-1]
        targets = input_ids[:, 1:].reshape(-1)
        scores, _ = score_tokens(
            hidden.reshape(-1, hidden.shape[-1]), model.lm_head.weight, targets, chunk_size=logits_chunk,
        )
        total_loss -= float(scores.sum())
        total_tokens += targets.numel()
    return total_loss, total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paper held-out perplexity.")
    parser.add_argument("--model", required=True, help="HF checkpoint directory or Hub ID.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = load_complete_model(args.model, dtype=dtype).to(device)
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

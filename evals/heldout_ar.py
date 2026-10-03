"""Exact autoregressive held-out evaluation of the imported paper checkpoints.

The input split and next-token target shift match evals.heldout. Vocabulary
projection and cross-entropy are chunked so the full (batch, time, vocab)
logit tensor is never allocated. Each batch is independent, including its
document boundaries and cache state.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks._measurement import checkpoint_files
from evals.execution import score_hidden
from evals.loading import load_complete_model
from training.compile import compile_evaluation
from training.data import load_cache_metadata, split_row_indices
from training.precision import attention_kernel_context
from white_matter.compilation import add_compile_argument, execution_policy
from white_matter.models import register_models
from white_matter.modules.precision import model_autocast_context


@torch.inference_mode()
def forward_hidden(model, ids: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "configured":
        return model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    if mode != "ar":
        raise ValueError(f"unknown evaluation mode: {mode!r}")
    if model.config.model_type not in {"white_matter", "lckv"}:
        raise ValueError(f"AR evaluation does not support model_type={model.config.model_type!r}")
    # Use the complete model so layer execution, document handling, and residual
    # precision stay identical to inference. Each call allocates a fresh cache.
    prefill_mode = model.config.prefill_mode
    try:
        model.config.prefill_mode = "autoregressive"
        return model.model(input_ids=ids, use_cache=True, return_dict=True).last_hidden_state
    finally:
        model.config.prefill_mode = prefill_mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("ar", "configured"), default="ar")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ce-chunk", type=int, default=256)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--n-eval", type=int, default=0, help="0 scores the rest of the test split")
    parser.add_argument("--sequence-length", type=int, default=2048)
    add_compile_argument(parser)
    args = parser.parse_args()
    with execution_policy(args.compile):
        _run(args, parser)


def _run(args, parser):
    if min(args.batch_size, args.ce_chunk, args.sequence_length) < 1 or args.offset < 0 or args.n_eval < 0:
        raise ValueError("batch, CE chunk, sequence length must be positive; offset and n-eval nonnegative")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    register_models()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("exact AR held-out evaluation requires CUDA")
    # Keep the FP32 master weights from the paper checkpoint. Model GEMMs run
    # under BF16 autocast, as in the original held-out evaluation.
    model = load_complete_model(str(args.model), dtype=torch.float32).to(device).eval()
    if args.compile:
        compile_evaluation(model)

    cache_meta = load_cache_metadata(args.data_dir)
    splits = cache_meta["splits"]
    indices = split_row_indices(
        args.data_dir,
        "test",
        n_train=splits["n_train"],
        n_val=splits["n_val"],
        n_test=splits["n_test"],
    )
    if args.offset >= len(indices):
        raise ValueError("offset lies outside the test split")
    stop = len(indices) if args.n_eval == 0 else args.offset + args.n_eval
    if stop > len(indices):
        raise ValueError("requested rows extend past the test split")
    tokens = np.load(args.data_dir / "tokenized.npy", mmap_mode="r")
    if tokens.shape[1] < args.sequence_length:
        raise ValueError("token cache is shorter than requested sequence length")

    total_ce = 0.0
    total_tokens = 0
    started = time.monotonic()
    batch_size = args.batch_size
    offset = args.offset
    while offset < stop:
        count = min(batch_size, stop - offset)
        rows = np.array(
            tokens[indices.start + offset : indices.start + offset + count, : args.sequence_length],
            dtype=np.int64,
            copy=True,
        )
        ids = torch.from_numpy(rows).to(device)
        hidden = None
        try:
            torch.cuda.reset_peak_memory_stats()
            with attention_kernel_context(str(device)), model_autocast_context(str(device)):
                hidden = forward_hidden(model, ids, args.mode)
                ce = score_hidden(hidden, ids, model.lm_head.weight, args.ce_chunk)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            del ids, rows, hidden
            torch.cuda.empty_cache()
            if count == 1:
                raise
            batch_size = max(1, count // 2)
            print(f"OOM at batch={count}; retrying with batch={batch_size}", flush=True)
            continue
        total_ce += ce
        total_tokens += count * (args.sequence_length - 1)
        offset += count
        print(
            json.dumps(
                {
                    "done": offset - args.offset,
                    "total": stop - args.offset,
                    "batch": count,
                    "ppl_running": math.exp(total_ce / total_tokens),
                    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "elapsed_s": round(time.monotonic() - started, 1),
                }
            ),
            flush=True,
        )
        del ids, rows, hidden

    result = {
        "compiled": args.compile,
        "checkpoint": checkpoint_files(str(args.model)),
        "model_config": model.config.to_dict(),
        "model": str(args.model),
        "mode": args.mode,
        "model_type": model.config.model_type,
        "execution": "exact_token_serial_ar" if args.mode == "ar" else "checkpoint_configured",
        "split": "test",
        "data_dir": str(args.data_dir),
        "offset": args.offset,
        "sequences": stop - args.offset,
        "sequence_length": args.sequence_length,
        "tokens": total_tokens,
        "batch_size_requested": args.batch_size,
        "batch_size_final": batch_size,
        "ce_chunk": args.ce_chunk,
        "cross_entropy_sum": total_ce,
        "cross_entropy": total_ce / total_tokens,
        "perplexity": math.exp(total_ce / total_tokens),
        "elapsed_s": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

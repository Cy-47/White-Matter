"""Compiled BF16 decoder timing at the passes selected by FP32 evaluation."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from benchmarks._measurement import (
    checkpoint_files,
    environment,
    snapshot_source,
    summarize,
    verify_sources,
    write_json,
)
from evals.loading import load_complete_model
from studies.prefill_convergence.contiguous import _forward as contiguous_forward
from studies.prefill_convergence.evaluate import load_windows
from studies.prefill_convergence.protocol import MODES, schedules, validate_checkpoint
from white_matter.compilation import execution_policy
from white_matter.models import register_models
from white_matter.modules.precision import model_autocast_context
from white_matter.ops.strict_causal_attention import _standalone_flash_available


@torch.compiler.disable
def _observe_full_pass(_passes, _hidden):
    """Keep full pass outputs live without retaining all passes in GPU memory."""


def _compile_trial(model, x, positions):
    block = model.model.decoder.block

    def trial(mode, passes):
        if mode == "ar":
            cache = model.allocate_inference_cache(x.shape[1])
            return model.model.decoder(x, past_key_values=cache, position_ids=positions)
        if mode == "jacobi":
            return block.forward_jacobi(x, num_passes=passes, num_gradient_passes=0)
        if mode.startswith("contiguous"):
            return contiguous_forward(
                block,
                x,
                num_passes=passes,
                chunks=MODES[mode],
                backend="auto",
                on_pass=_observe_full_pass,
                output_final_state=False,
            )
        return block(x, num_passes=passes, cyclic_groups=MODES[mode], num_gradient_passes=0, on_pass=_observe_full_pass)

    return torch.compile(
        trial,
        dynamic=False,
        # Each schedule specializes the shared trial code, including the AR baseline.
        recompile_limit=max(len(MODES) + 1, torch._dynamo.config.recompile_limit),
        options={"emulate_precision_casts": True, "reorder_for_locality": False},
    )


@torch.inference_mode()
@execution_policy()
def measure(model, ids, selections, *, warmups=5, repetitions=30, ar_repetitions=10, include_ar=True):
    if not ids.is_cuda:
        raise ValueError("convergence timing requires CUDA")
    model.config.document_separator_token_id = None
    model.config.prefill_mode = "autoregressive"
    rows = {}
    with model_autocast_context(ids.device):
        x = model.model.embed_tokens(ids)
        positions = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
        trial = _compile_trial(model, x, positions)
        cases = {"ar": None, **selections} if include_ar else selections
        for mode, passes in cases.items():
            if passes is None and mode != "ar":
                continue

            for _ in range(warmups):
                trial(mode, passes)
            torch.cuda.synchronize()
            samples = []
            for _ in range(ar_repetitions if mode == "ar" else repetitions):
                start = perf_counter()
                trial(mode, passes)
                torch.cuda.synchronize()
                samples.append(perf_counter() - start)
            stats = summarize(samples)
            rows[mode] = dict(passes=passes, **stats, seconds_per_sequence=stats["median_seconds"] / len(ids))
            print(f"{mode}: {rows[mode]['seconds_per_sequence']:.6f} s/sequence", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True, help="Pooled quality JSON from analyze")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--modes", nargs="+", choices=["ar", *MODES], help="Subset for parallel timing workers.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    quality = json.loads(args.quality.read_text())
    if quality.get("schedules") != schedules(quality["modes"]):
        raise ValueError("quality must record current schedule semantics")
    modes = args.modes if args.modes is not None else ["ar", *quality["modes"]]
    if len(set(modes)) != len(modes) or set(modes) - {"ar", *quality["modes"]}:
        parser.error("timing modes must be unique and present in quality results")
    selected = {mode: quality["modes"][mode]["passes"] for mode in modes if mode != "ar"}
    if any(passes is None for passes in selected.values()):
        raise ValueError("extend quality evaluation for modes that have not reached the threshold")
    windows, meta = load_windows(args.windows)
    if not 1 <= args.batch_size <= len(windows):
        parser.error(f"batch size must be between 1 and {len(windows)}")
    identity = checkpoint_files(args.model)
    if quality["checkpoint"] != identity or quality["windows_sha256"] != meta["sha256"]:
        raise ValueError("timing inputs differ from quality inputs")
    register_models()
    model = load_complete_model(args.model, dtype=torch.float32).cuda().eval()
    validate_checkpoint(model.config)
    for module in model.modules():
        if hasattr(module, "attention_implementation"):
            module.attention_implementation = "sdpa"
    model.config._attn_implementation = "sdpa"
    ids = torch.from_numpy(windows[: args.batch_size].astype(np.int64)).cuda()
    source = snapshot_source(args.output.parent / "sources")
    rows = measure(model, ids, selected, include_ar="ar" in modes)
    if checkpoint_files(args.model) != identity:
        raise RuntimeError("checkpoint changed during timing")
    verify_sources(source)
    write_json(
        args.output,
        {
            "source": source,
            "schedules": schedules(selected),
            "protocol": "prefill_convergence",
            "checkpoint": identity,
            "windows_sha256": meta["sha256"],
            "sequence_length": 2048,
            "batch_size": args.batch_size,
            "precision": "bf16",
            "compiled": True,
            "scope": "full layer sweep every pass; decoder excluding embeddings, final norm, LM head and token selection",
            "contiguous_backend": "flash_attention_2" if _standalone_flash_available() else "torch_flash",
            "warmups": 5,
            "repetitions": 30,
            "ar_repetitions": 10,
            "rows": rows,
            "environment": environment(0),
        },
    )


if __name__ == "__main__":
    main()

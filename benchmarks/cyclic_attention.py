"""Measure first-call and warm cyclic attention forward/backward latency."""

import argparse
import json
import time
from contextlib import nullcontext, redirect_stderr
from pathlib import Path

import torch

from benchmarks._measurement import (
    MemoryMonitor,
    create_run,
    environment,
    finish_run,
    record_case,
    summarize,
    verify_sources,
    write_json,
)
from benchmarks.report import summarize_run
from white_matter.ops import cyclic_attention, prepare_cyclic_attention_metadata


@torch._dynamo.config.patch(fail_on_recompile_limit_hit=True)
def benchmark(args) -> dict:
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.set_device(device)
    dtype = torch.bfloat16 if cuda else torch.float32
    torch.manual_seed(args.seed)
    q = torch.randn(
        args.batch_size,
        args.kv_heads * args.gqa_ratio,
        args.length // args.stride,
        args.head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k, v = [
        torch.randn(
            args.batch_size, args.kv_heads, args.length + args.cache_padding, args.head_dim, device=device, dtype=dtype
        )[:, :, : args.length]
        .detach()
        .requires_grad_()
        for _ in range(2)
    ]
    metadata = None
    if args.documents:
        documents = (
            (torch.arange(args.length, device=device) // (args.length // 4)).expand(args.batch_size, -1).contiguous()
        )
        metadata = prepare_cyclic_attention_metadata(
            documents[:, :: args.stride],
            torch.cat((documents.new_full((args.batch_size, 1), -1), documents[:, :-1]), 1),
        )

    @torch.compile(fullgraph=True, dynamic=True, options={"emulate_precision_casts": True})
    def attention(q, k, v):
        return cyclic_attention(q, k, v, query_stride=args.stride, metadata=metadata, backend=args.backend)

    def step():
        out = attention(q, k, v)
        torch.autograd.grad(out, (q, k, v), torch.ones_like(out))

    cache_info = None
    if args.backend == "tilelang":
        from white_matter.ops.cyclic_attention._tilelang.registration import _get_kernel

        cache_info = _get_kernel.cache_info
    cache_before = cache_info()._asdict() if cache_info else None
    if cuda:
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    step()
    if cuda:
        torch.cuda.synchronize(device)
    first_call_seconds = time.perf_counter() - start
    cache_after_first = cache_info()._asdict() if cache_info else None
    for _ in range(2):
        step()
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    with (
        MemoryMonitor(torch.cuda.current_device()) if cuda else nullcontext() as monitor,
        torch.compiler.set_stance("fail_on_recompile"),
    ):
        for _ in range(args.repetitions):
            start = time.perf_counter()
            for _ in range(args.iterations):
                step()
            if cuda:
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) / args.iterations)
    memory = None
    if cuda:
        reserved = torch.cuda.max_memory_reserved(device)
        memory = dict(
            monitor.peaks,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=reserved,
            budget_accounted_bytes=reserved + monitor.peaks["sampled_non_torch_peak_bytes"],
        )
    return {
        # First-call time includes PyTorch compilation and any TileLang cache
        # loading/compilation. A builder miss is not necessarily a disk-cache miss.
        "first_call_seconds": first_call_seconds,
        "kernel_cache": {
            "before": cache_before,
            "after_first": cache_after_first,
            "after_warm": cache_info()._asdict() if cache_info else None,
        },
        "forward_backward": summarize(samples, q.shape[0] * q.shape[2]),
        "memory": memory,
        "environment": environment(torch.cuda.current_device() if cuda else None),
        "dtype": str(dtype),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["reference", "tilelang"], default="reference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--head-dim", type=int, choices=[64, 96, 128], default=96)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--gqa-ratio", type=int, default=2)
    parser.add_argument("--cache-padding", type=int, default=0, help="unused native K/V capacity")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--documents", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.length < 4 or args.stride < 1 or args.length % args.stride or min(args.iterations, args.repetitions) < 1:
        parser.error("length >= 4 must be divisible by positive stride; iterations and repetitions must be positive")
    if min(args.batch_size, args.kv_heads, args.gqa_ratio) < 1 or args.cache_padding < 0:
        parser.error("batch size, KV heads and GQA ratio must be positive; cache padding must be nonnegative")
    workload = {k: v for k, v in vars(args).items() if k != "output"}
    directory = create_run(
        args.output,
        "cyclic-attention",
        dict(workload, compiled=True, dynamic=True, compile_options={"emulate_precision_casts": True}),
    )
    source = json.loads((directory / "run.json").read_text())["source"]
    case = {"case_id": "0000-cyclic-attention", "workload": workload, "status": "pending"}
    path = directory / "cases" / "0000-cyclic-attention.json"
    write_json(path, case)
    with path.with_suffix(".log").open("w") as log, redirect_stderr(log):
        record_case(path, case, lambda: benchmark(args), lambda: verify_sources(source))
    if case["status"] != "failed":
        finish_run(directory, [case["case_id"]])
    summarize_run(directory)
    print(directory, flush=True)
    if case["status"] == "failed":
        raise SystemExit(case["error"])


if __name__ == "__main__":
    main()

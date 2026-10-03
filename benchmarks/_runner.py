"""Fresh benchmark workers, matched workloads, and memory-budget capacity search."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks._measurement import (
    checkpoint_files,
    create_run,
    digest,
    finish_run,
    fits_memory,
    record_case,
    verify_sources,
    write_json,
)
from benchmarks.report import summarize_run
from white_matter.compilation import execution_policy, warn_eager


def input_files(model):
    path = Path(model)
    return {path.name: digest(path)} if path.is_file() else checkpoint_files(model)


def worker_command(module, path, world_size):
    launch = [sys.executable, "-m", module]
    if world_size > 1:
        launch = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            str(world_size),
            "--module",
            module,
        ]
    return [*launch, "--worker", str(path)]


def add_run_arguments(parser):
    parser.add_argument("--attention-backend", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--prompt-lengths", "--sequence-lengths", nargs="+", type=int, default=[2048])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--worker-repeats", type=int, default=1)
    parser.add_argument("--memory-budget-gib", nargs="+", type=float, default=[])
    parser.add_argument("--memory-headroom-gib", type=float, default=0.5)
    parser.add_argument("--max-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)


def validate_run(parser, args):
    if (
        not args.models
        or min(
            *args.batch_sizes,
            *args.prompt_lengths,
            args.repetitions,
            args.max_batch_size,
            args.warmups,
            args.worker_repeats,
        )
        < 1
    ):
        parser.error("models and positive workloads, warmups and repetitions are required")
    budgets = [args.memory_headroom_gib, *args.memory_budget_gib]
    if (
        not all(math.isfinite(b) for b in budgets)
        or args.memory_headroom_gib < 0
        or any(b <= args.memory_headroom_gib for b in args.memory_budget_gib)
    ):
        parser.error("finite budgets must exceed finite nonnegative headroom")
    if args.profile and (
        len(args.models) != 1 or len(args.batch_sizes) != 1 or len(args.prompt_lengths) != 1 or args.memory_budget_gib
    ):
        parser.error("profile requires one model/batch/length and no capacity search")
    if not args.compiled:
        warn_eager()


def run_worker(path: Path, benchmark) -> None:
    case = json.loads(path.read_text())
    run = json.loads((path.parent.parent / "run.json").read_text())
    model = case["workload"]["model"]

    def verify():
        verify_sources(run["source"])
        if input_files(model) != run["checkpoints"][model]:
            raise RuntimeError("checkpoint changed since run creation")

    rank = int(os.environ.get("RANK", "0"))
    distributed = case["workload"].get("world_size", 1) > 1
    destination = path.parent / "ranks" / f"{path.stem}.rank{rank}.json" if distributed else path

    def execute():
        with execution_policy(case["workload"].get("compiled", True)):
            return benchmark(argparse.Namespace(**case["workload"]))

    record_case(destination, case, execute, verify)
    if case["status"] == "failed" or (distributed and case["status"] == "cuda_oom"):
        raise SystemExit(1)
    if distributed and rank == 0:
        write_json(path, case)


def run_sweep(args, *, kind, module, workload_keys, metadata):
    args.models = [str(Path(model).resolve()) for model in args.models]
    checkpoints = {model: input_files(model) for model in args.models}
    protocol = {k: v for k, v in vars(args).items() if k not in {"output", "worker"}}
    protocol.update(metadata, memory_window="steady_state", setup_memory_budget=False)
    directory = create_run(args.output, kind + ("-profile" if args.profile else ""), protocol, checkpoints)
    for index, model in enumerate(args.models):
        if Path(model).is_file():
            target = directory / "recipes" / f"{index}-{Path(model).name}"
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(model, target)
            if input_files(model) != checkpoints[model] or digest(target) != next(iter(checkpoints[model].values())):
                raise RuntimeError("recipe changed while archiving")
    print(directory, flush=True)
    measured = {}

    def measure(model, length, batch, repeat):
        key = (model, length, batch, repeat)
        if key in measured:
            return measured[key]
        case_id = f"{len(measured):04d}-{Path(model).name}-p{length}-b{batch}-r{repeat}"
        path = directory / "cases" / f"{case_id}.json"
        workload = {k: getattr(args, k) for k in workload_keys}
        workload.update(model=model, prompt_length=length, batch_size=batch, repeat=repeat, profile=None)
        if args.profile:
            trace = directory / "profiles" / f"{case_id}.trace.json"
            trace.parent.mkdir(parents=True, exist_ok=True)
            workload["profile"] = str(trace)
        case = {"case_id": case_id, "order": len(measured), "workload": workload, "status": "pending"}
        write_json(path, case)
        with path.with_suffix(".log").open("w") as log:
            process = subprocess.run(
                worker_command(module, path, getattr(args, "world_size", 1)), stdout=log, stderr=subprocess.STDOUT
            )
        case = json.loads(path.read_text())
        if process.returncode or case["status"] not in {"ok", "cuda_oom", "profiled"}:
            ranks = [json.loads(p.read_text()) for p in (path.parent / "ranks").glob(f"{path.stem}.rank*.json")]
            oom = any(r["status"] == "cuda_oom" for r in ranks) and not any(r["status"] == "failed" for r in ranks)
            case.update(status="cuda_oom" if oom else "failed", returncode=process.returncode)
            write_json(path, case)
        measured[key] = case
        summarize_run(directory)
        if case["status"] == "failed":
            raise RuntimeError(f"benchmark worker failed; inspect {path.with_suffix('.log')}")
        print(f"{case_id}: {case['status']}", flush=True)
        return case

    for repeat in range(1 if args.profile else args.worker_repeats):
        for length_index, length in enumerate(args.prompt_lengths):
            models = args.models if (repeat + length_index) % 2 == 0 else args.models[::-1]
            for batch_index, batch in enumerate(args.batch_sizes):
                for model in models if batch_index % 2 == 0 else models[::-1]:
                    measure(model, length, batch, repeat)
            for budget in args.memory_budget_gib:
                capacities = []
                limit = (budget - args.memory_headroom_gib) * 2**30
                for model in models:

                    def fits(batch, model=model, length=length, repeat=repeat, limit=limit):
                        case = measure(model, length, batch, repeat)
                        return fits_memory(case, limit)

                    # Start beyond every known feasible point, including matched-batch probes.
                    low = max(
                        (
                            b
                            for (m, p, b, r), c in measured.items()
                            if (m, p, r) == (model, length, repeat)
                            and b <= args.max_batch_size
                            and fits_memory(c, limit)
                        ),
                        default=0,
                    )
                    high = low + 1
                    while high <= args.max_batch_size and fits(high):
                        low, high = high, high * 2
                    high = min(high, args.max_batch_size + 1)
                    while low + 1 < high:
                        middle = (low + high) // 2
                        if fits(middle):
                            low = middle
                        else:
                            high = middle
                    capacities.append(low)
                if capacities and min(capacities) > 0:
                    for model in models:
                        measure(model, length, min(capacities), repeat)
    finish_run(directory, [c["case_id"] for c in measured.values()])
    summary = summarize_run(directory)
    if any(row["nonmonotonic"] for row in summary["capacity"]):
        print("Observed nonmonotonic memory admission; see capacity flags in summary.json", flush=True)

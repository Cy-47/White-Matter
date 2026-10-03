"""Complete production training updates on deterministic synthetic packed inputs."""

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from benchmarks._measurement import MemoryMonitor, environment, summarize
from benchmarks._runner import add_run_arguments, run_sweep, run_worker, validate_run
from training.distributed import initialize_all_reduce, setup_distributed
from training.optim import build_optimizers, step_optimizers
from training.precision import configure_precision
from training.recipes import load_recipe
from training.step import prepare_training_forward, training_gradients
from white_matter.compilation import add_compile_argument


def synthetic_batches(recipe, batch_size, *, device, rank, count):
    """Fixed allocation; rank-specific documents change across microbatches."""
    generator = torch.Generator(device=device).manual_seed(recipe.seed + rank + 1)
    length = recipe.data.sequence_length
    batches = []
    for index in range(count):
        ids = torch.randint(recipe.model.vocab_size, (batch_size, length), device=device, generator=generator)
        ids[ids == recipe.data.eos_token_id] = (recipe.data.eos_token_id + 1) % recipe.model.vocab_size
        spacing = max(2, length // 4)
        for row in range(batch_size):
            ids[row, 1 + (index * 29 + row * 71 + rank) % (spacing - 1) :: spacing] = recipe.data.eos_token_id
        batches.append(ids)
    return batches


def benchmark(args):
    rank, world, local_rank, device, _ = setup_distributed()
    if not torch.cuda.is_available():
        raise RuntimeError("training benchmark requires CUDA")
    if world != args.world_size:
        raise RuntimeError("worker world size differs from the requested workload")
    initialize_all_reduce(device)
    recipe = load_recipe(args.model)
    recipe = replace(
        recipe,
        seed=args.seed,
        data=replace(recipe.data, sequence_length=args.prompt_length),
        global_batch_size=args.batch_size * world * recipe.gradient_accumulation_steps,
    )
    if recipe.optimizer.distributed_muon and world == 1:
        raise ValueError("distributed_muon requires --world-size greater than one")
    if recipe.ar_cuda_graph and not args.compiled:
        raise ValueError("eager comparison requires ar_cuda_graph=false in the recipe")
    torch.manual_seed(recipe.seed)
    configure_precision(device)
    torch.cuda.reset_peak_memory_stats(device)
    setup_start = time.perf_counter()
    print(
        f"{recipe.name}: rank={rank}/{world} batch={args.batch_size} length={args.prompt_length} "
        f"compiled={args.compiled} loss={recipe.loss_backend} accumulation={recipe.gradient_accumulation_steps} "
        f"distributed_muon={recipe.optimizer.distributed_muon}",
        flush=True,
    )
    recipe.model._attn_implementation = args.attention_backend
    model = AutoModelForCausalLM.from_config(recipe.model).to(device=device, dtype=torch.float32).train()
    runner = prepare_training_forward(model, recipe, args.batch_size, compiled=args.compiled)
    opt = recipe.optimizer
    optimizers = build_optimizers(
        model,
        base_lr=opt.learning_rate,
        weight_decay=opt.weight_decay,
        adam_beta1=opt.adam_beta1,
        adam_beta2=opt.adam_beta2,
        muon_momentum=opt.muon_momentum,
        muon_ns_steps=opt.muon_ns_steps,
        device=device,
        distributed_muon=opt.distributed_muon,
        compiled=args.compiled,
    )
    batches = synthetic_batches(
        recipe, args.batch_size, device=device, rank=rank, count=max(2, recipe.gradient_accumulation_steps)
    )
    update_index = 0

    def step():
        nonlocal update_index
        inputs = (batches[(update_index + i) % len(batches)] for i in range(recipe.gradient_accumulation_steps))
        loss, norm = training_gradients(
            model, runner, optimizers, inputs, recipe, world=world, check_gradients=update_index == 0
        )
        if not torch.isfinite(norm) or not math.isfinite(loss):
            raise RuntimeError("nonfinite training loss or gradient norm")
        step_optimizers(optimizers)
        update_index += 1
        return loss, float(norm)

    for _ in range(args.warmups):
        loss, norm = step()
    print(f"warmup complete: loss={loss:.6f} gradient_norm={norm:.6f}", flush=True)
    torch.cuda.synchronize(device)
    setup_memory = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    setup_seconds = time.perf_counter() - setup_start
    torch.cuda.empty_cache()
    samples = []
    compiler_guard = torch.compiler.set_stance("fail_on_recompile") if args.compiled else nullcontext()
    if args.profile:
        trace = str(Path(args.profile).with_name(Path(args.profile).stem + f".rank{rank}.json"))
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as profile:
            with compiler_guard:
                step()
            torch.cuda.synchronize(device)
        profile.export_chrome_trace(trace)
        return {"trace": trace, "environment": environment(local_rank)}
    for _ in range(args.repetitions):
        if world > 1:
            dist.barrier()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        with MemoryMonitor(local_rank) as monitor, compiler_guard:
            start = time.perf_counter()
            loss, norm = step()
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
        memory = dict(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            **monitor.peaks,
        )
        memory["budget_accounted_bytes"] = max(
            memory["sampled_device_peak_bytes"], memory["peak_reserved_bytes"] + memory["sampled_non_torch_peak_bytes"]
        )
        samples.append({"seconds": seconds, "loss": loss, "grad_norm": norm, "memory": memory})
    local = {
        "samples": samples,
        "environment": environment(local_rank),
        "setup_memory": setup_memory,
        "setup_seconds": setup_seconds,
    }
    ranks = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(local, ranks, dst=0)
    else:
        ranks[0] = local
    if rank != 0:
        return local
    seconds = [max(r["samples"][i]["seconds"] for r in ranks) for i in range(args.repetitions)]
    memory = {key: max(sample["memory"][key] for r in ranks for sample in r["samples"]) for key in memory}
    resolved = asdict(recipe)
    resolved["model"] = recipe.model.to_dict()
    return {
        "training": summarize(
            seconds, args.batch_size * args.prompt_length * world * recipe.gradient_accumulation_steps
        ),
        "memory": memory,
        "ranks": ranks,
        "parameters": model.num_parameters(),
        "recipe": resolved,
        "setup_seconds": max(r["setup_seconds"] for r in ranks),
        "compiled": args.compiled,
        "loss_backend": recipe.loss_backend,
        "config": model.config.to_dict(),
        "environment": environment(local_rank),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", dest="models", nargs="+", help="Training recipe YAML files")
    add_run_arguments(parser)
    parser.set_defaults(batch_sizes=[1])
    add_compile_argument(parser, "--compiled")
    parser.add_argument("--world-size", type=int, default=1, help="Local GPU processes per fresh worker")
    args = parser.parse_args()
    if args.worker:
        # On failure, exit promptly so the launcher can terminate blocked peers.
        run_worker(args.worker, benchmark)
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    validate_run(parser, args)
    if args.world_size < 1 or min(args.prompt_lengths) < 2:
        parser.error("world-size must be positive and sequence lengths at least two")
    for path in args.models:
        recipe = load_recipe(path)
        if recipe.optimizer.distributed_muon and args.world_size == 1:
            parser.error("recipe distributed_muon requires --world-size greater than one")
    args.phase = "training"
    run_sweep(
        args,
        kind="training",
        module="benchmarks.training",
        workload_keys=("phase", "warmups", "repetitions", "seed", "compiled", "world_size", "attention_backend"),
        metadata={
            "compiled": args.compiled,
            "prompt_distribution": "synthetic_packed_documents",
            "learning_rate": "recipe_peak_constant",
            "tokens": "batch * length * world * accumulation",
        },
    )


if __name__ == "__main__":
    main()

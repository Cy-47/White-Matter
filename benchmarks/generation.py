"""Compiled full-model prefill/decode measurements and independent capacity searches.

Run from the checkout: python -m benchmarks.generation --models wm-hf vanilla-hf.
"""

import argparse
import hashlib
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from benchmarks._measurement import (
    MemoryMonitor,
    environment,
    summarize,
)
from benchmarks._runner import add_run_arguments, run_sweep, run_worker, validate_run
from white_matter.models import register_models
from white_matter.models.generation import DecodeGraph
from white_matter.models.generation import prefill as prefill_forward

# Keep explicit BF16 rounding and avoid locality-induced full-prefix cache copies.
COMPILE_OPTIONS = {"emulate_precision_casts": True, "reorder_for_locality": False}


def configure_prefill_mode(config):
    """Use each architecture's measured finite-pass prefill policy."""
    if config.model_type == "white_matter":
        config.prefill_mode = "cyclic"
    elif config.model_type == "lckv":
        config.prefill_mode = "jacobi"


def collect_trials(trial, repetitions: int) -> tuple[list[dict], str]:
    """Keep raw timings and compare complete token bytes outside timed work."""
    samples = []
    expected = None
    for _ in range(repetitions):
        result, tokens = trial()
        if expected is not None and tokens != expected:
            raise RuntimeError("repeated trials changed generation outputs")
        expected = tokens
        samples.append(result)
    return samples, hashlib.sha256(expected).hexdigest()


@torch.no_grad()
def prepare_decode(model, prompt, batch_size, capacity):
    """Prefill one prompt, then copy its KV into independent batch rows."""
    prefix = model.allocate_inference_cache(prompt.shape[1])
    first = model(prompt, past_key_values=prefix, use_cache=True, logits_to_keep=1).logits[:, -1:].argmax(-1)
    return prefix.repeat_prefix(batch_size, capacity), first.expand(batch_size, -1).clone()


@torch.no_grad()
@torch._dynamo.config.patch(fail_on_recompile_limit_hit=True)
def benchmark(args):
    register_models()
    if not torch.cuda.is_available():
        raise RuntimeError("generation benchmark requires CUDA")
    torch.manual_seed(args.seed)
    if Path(args.model).is_file():
        from training.recipes import load_recipe

        recipe = load_recipe(args.model)
        recipe.model._attn_implementation = "flash_attention_2"
        model = AutoModelForCausalLM.from_config(recipe.model)
        # Match mixed inference storage: matrices BF16, norms and gains FP32.
        from white_matter.modules import KVPool

        dtype = getattr(torch, args.parameter_dtype)
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                module.to(dtype=dtype)
            elif isinstance(module, KVPool):
                for parameter in (module.k_proj_weight, module.v_proj_weight):
                    parameter.data = parameter.data.to(dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=getattr(torch, args.parameter_dtype),
            attn_implementation="flash_attention_2",
        )
    model = model.cuda().eval()
    # EOS is an ordinary token in this fixed-length, single-document workload.
    model.config.document_separator_token_id = None
    for module in model.modules():
        if hasattr(module, "num_splits"):
            module.num_splits = args.num_splits
    configure_prefill_mode(model.config)
    prompt = torch.randint(
        model.config.vocab_size, (1 if args.phase == "decode" else args.batch_size, args.prompt_length), device="cuda"
    )
    generated = torch.empty(
        (args.batch_size, 1 if args.phase == "prefill" else args.tokens), dtype=torch.long, device="cuda"
    )
    graph = None
    phases = ["prefill", "decode"] if args.phase == "end-to-end" else [args.phase]

    def trial():
        if args.phase == "decode":
            # The prefix is immutable. Rewind lengths; each trial overwrites its suffix.
            cache._restore_position(prefix_position)
            generated[:, :1] = first_token
        else:
            cache.reset()
        torch.cuda.synchronize()
        sample, reserved = {}, 0
        for phase in phases:
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            if phase == "prefill":
                logits = prefill_forward(model, prompt, cache, batch_size=args.prefill_batch_size or args.batch_size)
                generated[:, :1] = logits[:, -1].argmax(-1, keepdim=True)
                del logits
            else:
                for index in range(1, args.tokens):
                    token = generated[:, index - 1 : index]
                    logits = (
                        graph(token)
                        if graph is not None
                        else model(token, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
                    )
                    generated[:, index : index + 1] = logits[:, -1].argmax(-1, keepdim=True)
                    del logits
            torch.cuda.synchronize()
            sample[phase] = time.perf_counter() - start
            sample[f"{phase}_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            reserved = max(reserved, torch.cuda.max_memory_reserved())
        sample["peak_reserved_bytes"] = reserved
        assert cache.get_seq_length() == args.prompt_length + generated.shape[1] - 1
        return sample, generated.cpu().numpy().tobytes()

    setup = time.perf_counter()
    # Allocate and initialize native cache layers before decode capture.
    if args.phase == "decode":
        cache, first_token = prepare_decode(model, prompt, args.batch_size, args.prompt_length + args.tokens)
        # Preparation is outside decode's workload and memory budget.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        prefix_position = cache._snapshot_position()
    else:
        cache = model.allocate_inference_cache(args.prompt_length + (args.tokens if "decode" in phases else 0))
        prefill_forward(model, prompt, cache, batch_size=args.prefill_batch_size or args.batch_size)
    # Compile after allocation; setup is excluded from steady timing.
    if args.compiled:
        model.compile(options=COMPILE_OPTIONS)
    if args.phase != "prefill" and args.cuda_graph and args.tokens > 1:
        graph = DecodeGraph(model, cache)
    setup_memory = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    for _ in range(args.warmups):
        sample, _ = trial()
        setup_memory["peak_allocated_bytes"] = max(
            setup_memory["peak_allocated_bytes"],
            *(value for key, value in sample.items() if key.endswith("_peak_allocated_bytes")),
        )
        setup_memory["peak_reserved_bytes"] = max(setup_memory["peak_reserved_bytes"], sample["peak_reserved_bytes"])
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup
    torch.cuda.empty_cache()
    if args.profile:
        with (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
            ) as profile,
            torch.compiler.set_stance("fail_on_recompile"),
        ):
            trial()
        profile.export_chrome_trace(str(args.profile))
        return {"trace": str(args.profile), "environment": environment(0), "config": model.config.to_dict()}
    compiler_guard = torch.compiler.set_stance("fail_on_recompile") if args.compiled else nullcontext()
    with MemoryMonitor() as monitor, compiler_guard:
        samples, token_digest = collect_trials(trial, args.repetitions)
    memory = {key: max(s[key] for s in samples) for key in samples[0] if key.endswith("_bytes")}
    memory.update(
        monitor.peaks,
        budget_accounted_bytes=max(
            memory["peak_reserved_bytes"] + monitor.peaks["sampled_non_torch_peak_bytes"],
            monitor.peaks["sampled_device_peak_bytes"],
        ),
        kv_capacity_bytes=sum(
            t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values)
        ),
    )
    tokens = {"prefill": args.prompt_length, "decode": args.tokens - 1}
    timings = {phase: summarize([s[phase] for s in samples], args.batch_size * tokens[phase]) for phase in phases}
    if len(phases) == 2:
        timings["total"] = summarize([s["prefill"] + s["decode"] for s in samples], args.batch_size * args.tokens)
    return dict(
        **timings,
        setup_seconds=setup_seconds,
        setup_memory=setup_memory,
        memory=memory,
        parameters=model.num_parameters(),
        config=model.config.to_dict(),
        generated_tokens_sha256=token_digest,
        environment=environment(0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", "--recipes", nargs="+", help="Local HF checkpoints or training recipe YAML files")
    add_run_arguments(parser)
    parser.add_argument("--phase", choices=["prefill", "decode", "end-to-end"], default="end-to-end")
    parser.add_argument("--tokens", type=int, default=128, help="Includes the first token produced by prefill")
    parser.add_argument("--parameter-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-splits", type=int, default=0, help="FA decode splits; zero keeps automatic selection")
    parser.add_argument(
        "--prefill-batch-size", type=int, default=1, help="Prefill microbatch; zero uses the resident batch"
    )
    args = parser.parse_args()
    if args.worker:
        run_worker(args.worker, benchmark)
        return
    validate_run(parser, args)
    if args.tokens < 1:
        parser.error("tokens must be positive")
    if args.phase == "decode" and args.tokens < 2:
        parser.error("decode requires at least two generated tokens")
    if args.num_splits < 0 or args.prefill_batch_size < 0:
        parser.error("num-splits and prefill-batch-size must be nonnegative")
    run_sweep(
        args,
        kind="generation",
        module="benchmarks.generation",
        workload_keys=(
            "phase",
            "tokens",
            "warmups",
            "repetitions",
            "parameter_dtype",
            "cuda_graph",
            "num_splits",
            "prefill_batch_size",
            "seed",
            "compiled",
        ),
        metadata={
            "compiled": args.compiled,
            "prompt_distribution": "uniform_vocabulary",
            "compile_options": COMPILE_OPTIONS,
        },
    )


if __name__ == "__main__":
    main()

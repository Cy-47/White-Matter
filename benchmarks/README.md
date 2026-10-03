# Benchmarks

Run GPU benchmarks from the checkout after installing `pip install -e '.[research]'`. GPU workloads
require a compatible CUDA environment and TileLang. CCE recipes additionally
require the [documented Cut Cross-Entropy revision](../docs/reproduction.md).
These utilities are checkout tools, not part of the pip API.

Benchmarks default to PyTorch SDPA for normal attention. Use
`--attention-backend flash_attention_2` with `.[research,flash-attn]` to select
standalone normal attention. Cyclic attention uses TileLang. Convergence timing
records the resolved strict-causal backend. See the
[installation guide](../docs/installation.md#backend-selection) for backend selection
and setup.

## Full models

The Feedback Transformer control is `recipes/benchmarks/feedback_transformer_1p3b.yaml`.
It uses the paper's learned, static softmax mixture of the embedding and all
layer outputs, one shared K/V cache, and exact sequential prefill/decode over
the full causal prefix. It is a Qwen3-shaped architecture control; its random
weights measure runtime, not the quality of Fan et al.'s original checkpoint.
For the paper's runtime workload, select `--attention-backend flash_attention_2`
and `--cuda-graph` for generation, then use a 2,048-token prompt, 128 timed
decode steps (`--tokens 129`), the full resident prefill batch
(`--prefill-batch-size 0`), and a 40 GiB budget with 0.5 GiB headroom.

Use recipe YAML files for seeded random weights, or local HF exports for generation.
Recipe initialization does not require downloading weights or data. It measures execution,
not model quality. Both generation inputs and training packed documents are synthetic.

```bash
python -m benchmarks.training --recipes recipes/paper/white_matter_1p3b.yaml recipes/paper/vanilla_1p3b.yaml \
  --sequence-lengths 2048 --batch-sizes 1 --memory-budget-gib 40
python -m benchmarks.generation --recipes recipes/paper/white_matter_1p3b.yaml recipes/paper/vanilla_1p3b.yaml \
  --prompt-lengths 2048 --batch-sizes 24 --phase prefill --prefill-batch-size 0 --memory-budget-gib 40
python -m benchmarks.generation --recipes recipes/paper/white_matter_1p3b.yaml recipes/paper/vanilla_1p3b.yaml \
  --prompt-lengths 2048 --batch-sizes 24 --phase decode --tokens 129 --memory-budget-gib 40
```

Generation is compiled and uses CUDA graph replay by default with either SDPA
or external FlashAttention. Use `--no-cuda-graph` to disable replay. `--tokens` includes the first token selected during prefill: 129 means 128
timed decode steps. Decode prefix preparation is excluded. Each cache row owns storage.
Autoregressive prefill keeps its token loop in Python and compiles reusable
tensor steps. Prompt length therefore does not unroll the token loop into the graph.
Use `--no-compiled` to benchmark eager model execution when whole-model compilation
is impractical; the run record retains this setting.
`--prefill-batch-size 0` processes the complete resident batch; the default is one.
The `end-to-end` phase measures prefill followed by token generation.

Training compiles by default; `--no-compiled` explicitly selects eager execution.
Compilation includes Muon's Newton–Schulz tensor computation, preserving its BF16
casts. Momentum/parameter updates use foreach kernels and AdamW uses its fused kernel;
learning-rate scheduling and optimizer state management stay outside compilation.
It calls the same forward setup, loss, clipped-gradient calculation, and optimizers as
`training.engine`. Recipe settings control CCE (`cce_exact`), accumulation, pass counts,
residual precision, AR checkpointing/graphs, and distributed Muon. Parameters and gradients
remain FP32. Updates use the recipe's constant peak learning rate, rather than its training
schedule. Token throughput counts batch × sequence length × accumulation × world size.
`--batch-sizes` means the per-device microbatch. Synthetic document patterns vary across
microbatches/ranks; a small fixed input pool is reused across updates to bound input memory.

For distributed training, run the CLI once inside a multi-GPU allocation and pass
`--world-size N`. Each case launches fresh local workers and uses the production gradient
reducer. Set `optimizer.distributed_muon: true` in the recipe to shard Muon state; a
single-GPU request with this setting fails rather than silently changing the optimizer.
The result retains every rank's samples; throughput uses the slowest rank per update and
memory admission uses the largest per-device footprint. Multi-node launches are not supported.

## Measurement and results

Training defaults to batch 1; generation sweeps batches 1, 2, 4, 8, 16, 32, and 64.
Shared defaults: sequence length 2048, seed 1337, three warmups, five timed repetitions, one worker
per shape, capacity-search ceiling 1024. Increase `--worker-repeats` for independent trials
with reversed model order. Setup/compilation and warmed measurement are separate; the memory
budget applies to warmed execution, with 0.5 GiB headroom. Setup may require more physical
memory than that budget. Training records setup allocated/reserved peaks separately.
Budget accounting uses the maximum of sampled total device usage and peak reserved plus
stable external overhead. Use otherwise idle GPUs. Allocator peaks are exact; sampled device
usage can miss short external allocations. No CPU offloading or microbatch substitution is
performed by capacity search.

One runner handles matched batches and expansion/bisection searches for both workloads.
It checks the adjacent rejected batch unless the search ceiling is reached, flags observed
nonmonotonic admission, and measures both models at the smaller capacity as an additional
matched-batch comparison. Each prefill/decode invocation searches independently.

```text
outputs/                         # Default, relative to working directory; gitignored
  sources/<sha256>/               # Shared source archives, including training code
  <timestamp>-training/          # Or generation, cyclic-attention, or *-profile
    run.json                     # Inputs, source identity, execution settings
    recipes/                     # Exact recipe copies, when used
    cases/*.json                 # Results and raw timings
    cases/*.log                  # Worker logs
    cases/ranks/*.json           # Distributed worker outcomes
    profiles/*.json              # Chrome traces; separate instrumented runs
    summary.json
    RESULTS.md
```

Override the root with `--output`. A failed worker preserves its log and stops the sweep;
OOM is a recorded capacity result. Source/input changes invalidate results even after OOM.

```bash
python -m benchmarks.report outputs/<run-id> --csv outputs/table.csv
python -m benchmarks.training --recipes recipes/paper/white_matter_1p3b.yaml \
  --batch-sizes 1 --sequence-lengths 2048 --profile
python -m benchmarks.cyclic_attention --backend tilelang --device cuda --documents
```

Profiles do not contribute throughput results. Training writes one Chrome trace per rank.

## Cyclic attention operator

```bash
python -m benchmarks.cyclic_attention --backend tilelang --device cuda --length 2048 \
  --batch-size 2 --head-dim 96 --kv-heads 3 --gqa-ratio 2 --stride 8 --documents
python -m benchmarks.cyclic_attention --backend tilelang --device cuda --length 2048 \
  --cache-padding 256
```

The operator benchmark records first-call wall time separately from warmed forward/backward
latency. First-call time includes PyTorch compilation and TileLang compilation or disk-cache
loading; it is not an isolated compiler measurement. `kernel_cache` records Python kernel
cache counters before the first call, after it, and after warm measurements. A cache miss can
still load a compiled kernel from disk. `--cache-padding` exercises native views with unused
K/V capacity; `--batch-size`, `--head-dim`, `--kv-heads`, and `--gqa-ratio` select the workload.
Each CLI invocation starts a new process, so use the GPU shape-reuse regression test to check
reuse across different shapes within one process.

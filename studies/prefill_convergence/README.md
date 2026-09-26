# Controlled prefill convergence

This study trains the four-layer, width-512, k=4 control with exact
autoregressive execution for 800 updates (78,643,200 tokens). Its quality
reference is exact token-serial AR, evaluated in FP32. The timing experiment
uses compiled BF16 on one RTX A6000 at physical batch 32 by default, matching the paper.
Use `benchmark --batch-size 64` to measure another batch size; timings and
plots record the selected size, and timing shards must all use the same size.

Run from the checkout with the training and GPU dependencies installed:

```bash
python -m studies.prefill_convergence.prepare_data \
  --data-dir /path/to/cache --output outputs/convergence/windows.npy
torchrun --standalone --nproc-per-node=8 -m training.train \
  --recipe studies/prefill_convergence/recipes/exact_ar_4l.yaml \
  --data-dir /path/to/cache --output-dir outputs/convergence/model
python -m studies.prefill_convergence.evaluate \
  --model outputs/convergence/model/final --windows outputs/convergence/windows.npy \
  --output outputs/convergence/quality.json
python -m studies.prefill_convergence.analyze \
  --quality outputs/convergence/quality.json --output outputs/convergence/selected.json
python -m studies.prefill_convergence.benchmark \
  --model outputs/convergence/model/final --windows outputs/convergence/windows.npy \
  --quality outputs/convergence/selected.json --output outputs/convergence/timing.json
python -m studies.prefill_convergence.analyze \
  --quality outputs/convergence/quality.json --timing outputs/convergence/timing.json \
  --output outputs/convergence/results.json
python scripts/plot_experiments.py convergence \
  --input outputs/convergence/results.json --output outputs/convergence/figure.pdf
```

Data preparation selects the first 192 disjoint adjacent pairs of test rows
without EOS, retaining their source indices and a content fingerprint. It
stores 4096-token windows to preserve the original selection procedure;
evaluation and timing use each window's first 2048 tokens.

Quality evaluation observes every pass of one trajectory per schedule. Defaults
are 80 Jacobi passes, 32 passes for each cyclic schedule, and 80 passes for
each contiguous schedule. Both families cover 2, 4, 8, 16, 32, and 64 chunks. To distribute
quality evaluation, use four jobs with `--count 48` and offsets 0, 48, 96, 144,
then pass all four JSON files to `analyze --quality`. Analysis rejects missing,
overlapping, or incompatible shards. It pools CE sums before selecting the
first pass satisfying `CE <= AR_CE + log(1.01)`. A schedule that does not reach
the threshold is recorded as `not_reached`; extend the quality pass limit
before attempting to report its convergence time.

The FP32 scope disables model autocast, TF32, and reduced-precision attention
backends. It is inference-only and restores settings on exit. Timing uses
five warmups, 30 iterative trials, and 10 full AR rollouts; it records medians
and raw samples. The timed region covers the decoder block, including its KV
construction, and excludes embeddings, final normalization, output head, and
token selection. It is distinct from the full-generation runtime benchmark.
Every schedule executes all decoder layers on every timed pass. A no-op
pass observer disables the unused-final-layer shortcut in both cyclic and
contiguous schedules, matching the Jacobi workload. Compilation and input preparation occur before timing. Timing uses the same
checkpoint, token windows, and fixed channel selection as quality evaluation.

## Contiguous-chunk comparison

`contiguous16` partitions the sequence into 16 consecutive chunks (128 tokens
at length 2048), visiting them left to right on each pass. Each chunk reads
fixed KV state throughout its layer sweep, then publishes its updated KV for
later chunks. Tokens still attend only to strictly earlier tokens and the
learned dummy. This differs from cyclic groups, whose tokens are interleaved.
The iteration lives in this study's `contiguous.py`; the package is unchanged.

The default suite includes the full contiguous sweep with an 80-pass limit. Select just the
matched comparison with `evaluate --modes jacobi cyclic16 contiguous16`; use
`--contiguous-passes` to extend its trajectory. FP32 evaluation uses explicit
causal masks. Compiled BF16 timing uses the existing FlashAttention layer
dispatch with prefix-sliced KV and bottom-right causal alignment. CUDA tests
compare that actual path with the masked reference, including final KV state.

For a trusted legacy exact-AR checkpoint and its metadata sidecar:

```bash
python -m studies.prefill_convergence.import_checkpoint \
  --checkpoint /path/to/final_model.pt --output outputs/convergence/model/final
```

The importer validates the control's metadata and tensor names/shapes, retains
FP32 weights, and records the source fingerprint. The quality and timing JSONs
record schedule semantics; analysis rejects incompatible shards or timing.
The convergence plot includes perplexity versus passes, the exact-AR reference
and 1% threshold, and measured time at the first threshold crossing.

Submit `slurm/run.sbatch` from the checkout with the active environment and
your cluster account/partition. Its `validate` action runs the CUDA gates;
`quality MODEL WINDOWS OUTPUT [options]` runs one quality shard and
`timing MODEL WINDOWS OUTPUT --quality SELECTED` runs the batch-32 benchmark.
The launcher requests one RTX A6000 and isolates its compiler cache per job.

For eight-GPU evaluation, use eight quality workers with `--count 24` and
offsets 0, 24, 48, 72, 96, 120, 144, 168. Pool all eight outputs before timing.
Timing workers can select disjoint cases with `benchmark --modes`, e.g.
`--modes cyclic64 contiguous64`. Assign AR and Jacobi to separate workers and
one cyclic/contiguous pair to each of the remaining six workers. Each case
retains physical batch 32 on a single GPU. Pass all timing files to
`analyze --timing`; merging checks source identity, software, GPU type, and
protocol, rejects duplicated cases, and retains each worker's environment.

Render the complete trajectories and chunk-count comparison with:

```bash
python scripts/plot_experiments.py convergence-sweep \
  --input outputs/convergence/results.json --output outputs/convergence/sweep.pdf
```

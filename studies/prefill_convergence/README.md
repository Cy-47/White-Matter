# Controlled prefill convergence

This study trains the four-layer, width-512, k=4 control with exact
autoregressive execution for 800 updates (78,643,200 tokens). Its quality
reference is exact token-serial AR, evaluated in FP32. The timing experiment
uses compiled BF16 on one RTX A6000 at physical batch 64.

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
are 80 Jacobi passes and 32 passes for each of C2/C4/C8/C16/C32. To distribute
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
Compilation and input preparation occur before timing. Timing uses the same
checkpoint, token windows, and fixed channel selection as quality evaluation.

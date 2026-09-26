# Shared-mixture full-cache ablation

At every token, one dynamic router makes a **single hidden mixture** of all
feedback-layer inputs. One pre-mix RMSNorm and learned source gain are shared
by the key and value branches. The mixed hidden vector is normalized once,
then sixteen independent key and sixteen independent value projection matrices
produce the cached pairs. Per-channel post-mix gains and key normalization
remain independent, as in the main WhiteMatter model. Layer `ell` reads pair
`ell`; the cache has the same size as full-cache WhiteMatter.

The router has one output row. Its `cyclic:0.25` prior gives every source layer
equal initial weight. The control uses a sixteen-row router with the
`shifted_identity:0.25` prior. Both router weight matrices start at zero.

| Model | Trainable parameters | Excluding embeddings |
| --- | ---: | ---: |
| Full-rank control | 131,846,656 | 54,055,424 |
| Shared mixture | 129,806,352 | 52,015,120 |

## Training

The matched recipes use 16 layers, hidden size 512, 20,000 steps, global batch
size 8, and seed 1337. Both use three cyclic passes (one detached and two with
gradients), eight cyclic groups, router stride 2, and the same optimizer.
Run both arms on the same cache, read sequentially. See
[cache requirements](../../docs/reproduction.md) for data preparation.

```bash
sbatch studies/rank/slurm/train.sbatch \
  studies/rank/recipes/k16.yaml \
  /path/to/fineweb_edu_cache outputs/studies/rank/k16
sbatch studies/shared_mixture/slurm/train.sbatch \
  studies/shared_mixture/recipes/shared_k16.yaml \
  /path/to/fineweb_edu_cache outputs/studies/shared_mixture/shared_k16
```

For an 8B-token run on eight GPUs, use `--gpus-per-node=8` with
`studies/shared_mixture/recipes/shared_k16_8b.yaml`. The corresponding control
is `recipes/paper/white_matter_k16.yaml`; train both with the same data and
reading order.

## Three-pass evaluation

Evaluate the final 20,000-step checkpoints at exactly three passes on the full
5,000-sequence test split (10,235,000 targets). Compare against the dynamic
full-rank control using this same protocol; scores selected from a pass sweep
are not directly comparable.

```bash
python -m studies.shared_mixture.evaluate_heldout \
  --model outputs/studies/rank/k16/final \
  --data-dir /path/to/fineweb_edu_cache \
  --output outputs/studies/rank/k16/heldout_3pass.json
python -m studies.shared_mixture.evaluate_heldout \
  --model outputs/studies/shared_mixture/shared_k16/final \
  --data-dir /path/to/fineweb_edu_cache \
  --output outputs/studies/shared_mixture/shared_k16/heldout_3pass.json
```

The evaluator defaults to batch 16 and compiled CUDA, and writes token-weighted
perplexity. Use `studies/shared_mixture/slurm/evaluate.sbatch` to submit it to Slurm.
The `studies.shared_mixture.evaluate_lm_eval` wrapper accepts the same arguments
as `python -m evals.lm_eval` for downstream evaluation.

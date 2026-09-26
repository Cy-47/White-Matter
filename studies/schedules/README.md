# Figure 7a: iteration schedules

The recipe matrix has 24 schedule cells at each of seeds 1337 and 1338:
no-gradient passes 1/2/4, gradient passes 1/2, and TP/C4/C8/C16 execution.
TP uses token-parallel Jacobi iteration; C4, C8, and C16 use 4, 8, and 16
cyclic groups, respectively.
Every recipe uses the same 16-layer k8 architecture, 20,000 updates, global
batch 8, length 2048, and sequential reading of a validated paper cache.
The TP recipes use activation checkpointing for their gradient passes.

For a cell, submit its recipe and keep its outputs under the matching seed and
arm directory:

```bash
sbatch studies/schedules/slurm/train.sbatch \
  studies/schedules/recipes/seed1337/ng1_g2_c8.yaml \
  /path/to/fineweb_edu_cache \
  outputs/studies/schedules/seed1337/ng1_g2_c8
```

Figure 7a evaluates each final checkpoint over passes 1–32 under the trained
schedule, with a separate 32-pass C16 approximation and Jacobi convergence
curve. Use `studies/schedules/slurm/evaluate.sbatch` with mode `native`,
`cyclic16`, or `tp`, writing `eval_<mode>.json` inside the arm's output
directory. For C16-trained cells the native curve is also the C16 curve; for
TP-trained cells it is also the TP curve. Once all curves are available,
`python -m studies.schedules.analyze` writes per-seed and two-seed mean
summaries. This analysis belongs to Figure 7a; Figure 7b uses the
three-pass protocol in its own study.

Aggregate both seeds' curves before selecting metrics:

```bash
python -m studies.schedules.analyze --results-dir outputs/studies/schedules \
  --per-seed-output outputs/studies/schedules/per_seed.csv \
  --averaged-output outputs/studies/schedules/mean.csv
```

The mean table computes each metric from the pointwise mean perplexity curve,
not by averaging separately selected seed metrics. Native and C16 curves cover
exactly passes 1–32. Jacobi defaults to the paper's observation ceiling for each
arm: 128 for `ng4_g1_c8`, 96 for `ng4_g2_c4` and `ng4_g2_c16`, and 32 otherwise.
These defaults apply to both seeds and the Slurm evaluator. The evaluator's
`--last-pass` option can extend Jacobi curves further. Aggregation requires at
least the paper's ceiling and identical Jacobi pass ranges across seeds.

An empty `jacobi_passes_within_1pct` field means the threshold was not reached:
`jacobi_passes_within_1pct_censored` is true and `jacobi_max_evaluated_pass`
records the observation ceiling. The schedule plot renders this as a bound
such as `>96`. The paper's averaged curves cross at 36 for `ng4_g2_c4` and 105
for `ng4_g1_c8`; `ng4_g2_c16` remains above the threshold through pass 96.
If a TP-trained cell has a separate `eval_tp.json`, aggregation uses it;
otherwise its native curve supplies the Jacobi curve.

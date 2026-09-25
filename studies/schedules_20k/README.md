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
sbatch studies/schedules_20k/slurm/train.sbatch \
  studies/schedules_20k/recipes/seed1337/ng1_g2_c8.yaml \
  /path/to/fineweb_edu_cache \
  outputs/studies/schedules_20k/seed1337/ng1_g2_c8
```

Figure 7a evaluates each final checkpoint over passes 1–32 under the trained
schedule, with a separate 32-pass C16 approximation and Jacobi convergence
curve. Use `studies/schedules_20k/slurm/evaluate.sbatch` with mode `native`,
`cyclic16`, or `tp`, writing `eval_<mode>.json` inside the arm's output
directory. For C16-trained cells the native curve is also the C16 curve; for
TP-trained cells it is also the TP curve. Once all curves are available,
`python -m studies.schedules_20k.analyze` writes per-seed and two-seed mean
summaries. This 32-pass analysis belongs to Figure 7a; Figure 7b uses the
three-pass protocol in its own study.

# Figure 7b: pool rank and connectivity

This study compares pool rank, static routing, and depth-causal connectivity
with 16-layer models trained for 20,000 steps at global batch size 8.
All recipes read the shuffled FineWeb-Edu cache sequentially.

The eight WhiteMatter rank/static arms use one detached and two gradient
passes with C8, router stride 2, seed 1337, and exactly the paper's optimizer.
`k1` starts from `top:0.25`; k2/4/8/12 use `cyclic:0.25`; k16 uses
`shifted_identity:0.25`. Static arms freeze the router's content-dependent
weight and train its bias. The depth-causal arm uses an identity prior and one
depth-sequential pass. Vanilla also uses one pass.

Training and evaluation require compatible paper cache metadata and the
complete 5,000-sequence test split. Iterative arms are scored at **exactly
three passes**; vanilla and depth-causal are scored at one pass. Compare these
results using the same pass count and data order; fixed-pass scores differ
from scores selected by minimizing perplexity over a pass sweep.

```bash
for arm in k1 k2 k4 k8 k12 k16 k1_static k16_static vanilla; do
  sbatch studies/rank_20k/slurm/train.sbatch \
    "studies/rank_20k/recipes/${arm}.yaml" \
    /path/to/fineweb_edu_cache \
    "outputs/studies/rank_20k/${arm}"
done
```

The depth-causal run can be submitted separately with
`studies/rank_20k/recipes/k16_depth_causal.yaml`. Each final model can then
be evaluated with `studies/rank_20k/slurm/evaluate.sbatch`, writing
`outputs/studies/rank_20k/<arm>/heldout.json`. Collect the nine rank/baseline arms with:

```bash
python -m studies.rank_20k.analyze \
  --results-dir outputs/studies/rank_20k \
  --output outputs/studies/rank_20k/summary.csv
```

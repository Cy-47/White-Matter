# Reproduction tools

The repository and source distribution include the complete reproduction code.
The wheel installs only the reusable `white_matter` package.

| Directory | Purpose |
| --- | --- |
| `src/white_matter/` | Model families, feedback blocks, and attention kernels |
| `training/`, `evals/` | Shared training and evaluation commands |
| `recipes/paper/`, `recipes/analysis/` | Paper configurations and exact-AR control |
| `recipes/benchmarks/`, `benchmarks/` | Runtime controls and measurement tools |
| `studies/` | Ablation models, recipes, evaluation protocols, and analysis |
| `scripts/` | Data preparation, checkpoint conversion, and package checks |
| `slurm/` | Portable cluster launch examples |
| `tests/` | CPU checks, CUDA correctness gates, and typing contracts |

Run checkout commands from the repository root after installing the extras in
[the main guide](../README.md). Activate that environment before submitting
Slurm jobs. Pass your account, partition, and GPU type at submission time.

CUDA study evaluators and recipes with `loss_backend: cce` also require the
tested Cut Cross-Entropy revision, which provides `cce_exact`:

```bash
pip install 'cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git@b7a02791b234e187b524fb1dba6a812d521b203a'
```

Install the `dev` extra to run tests and build distributions. The optional
lm-eval tests additionally require the harness revision in the main guide.

## Data

Build the public FineWeb-Edu cache with the documented eight-worker row order:

```bash
python scripts/prepare_fineweb_edu.py --output /path/to/fineweb_edu_cache --workers 8
```

Study commands validate the source and packing configuration, split sizes,
and array layout. They require eight workers because the worker count affects
row order. Use the same cache for every arm in a comparison and retain its
metadata with results; metadata validation does not verify token-array contents.

## Legacy checkpoint conversion

Training in this repository exports Hugging Face checkpoints directly. For a
trusted checkpoint in the legacy `.pt` format, use:

```bash
python scripts/import_paper_eval_checkpoints.py \
  --architecture lckv_w4 \
  --checkpoint /path/to/final_model.pt \
  --output outputs/imported/lckv_w4
```

Use `--help` for supported architectures. Conversion validates tensor names and
shapes, preserves tied embeddings, and refuses to overwrite an existing export.
It loads Python pickle data, so the input must come from a trusted source.

## Exact autoregressive held-out evaluation

Evaluate WhiteMatter or LCKV through the complete model's exact AR prefill:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  slurm/evaluate_paper_ar_heldout.sbatch \
  outputs/imported/lckv_w4 /path/to/fineweb_edu_cache outputs/evals/lckv_ar.json \
  --batch-size 16
```

The launcher uses the active Python environment and accepts the options from
`python -m evals.heldout_ar --help`, including `--mode configured` to evaluate
the checkpoint's original execution schedule.

## Additional evaluation tasks

Install the pinned lm-eval harness from the main guide, then run the portable
launcher with an explicit checkpoint and output path:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  slurm/evaluate_paper_additions.sbatch \
  outputs/imported/lckv_w4 outputs/evals/lckv_extra.json smoke --batch-size 8
```

The groups are `smoke`, `likelihood` (BLiMP, SciQ, ReCoRD), and `generation`
(SQuAD completion). Extra arguments go to `evals.lm_eval`; WhiteMatter generation
requires an explicit prefill policy such as `--prefill-mode cyclic`.

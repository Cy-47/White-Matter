# Reproduction tools

The repository and source distribution include the complete reproduction code.
The wheel installs only the reusable `white_matter` package.

| Directory | Purpose |
| --- | --- |
| `src/white_matter/` | Model families, feedback blocks, and attention kernels |
| `training/`, `evals/` | Shared training and evaluation commands |
| `recipes/paper/` | Main quality configurations |
| `recipes/benchmarks/`, `benchmarks/` | Runtime controls and measurement tools |
| `studies/` | Ablation models, recipes, evaluation protocols, and analysis |
| `scripts/` | Data preparation, checkpoint conversion, and package checks |
| `slurm/` | Portable cluster launch examples |
| `tests/` | CPU checks, CUDA correctness gates, and typing contracts |

Run checkout commands from the repository root after installing the extras in
[the main guide](../README.md). Activate that environment before submitting
Slurm jobs. Pass your account, partition, and GPU type at submission time.

Training automatically resumes an output directory containing `ckpt_full.pt`.
If metrics or model exports exist without that checkpoint, choose a new output
directory; training refuses to mix a fresh trajectory with existing artifacts.
To resume an external checkpoint, pass `--resume-from-checkpoint` and a new
output directory.

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
(SQuAD completion). Extra arguments go to `evals.lm_eval`; use `--prefill-mode cyclic` to
override the checkpoint's configured WhiteMatter prefill policy.

## Main-body experiment map

The checkout supports the current paper's main-body experiments. Rebuilding
the data follows the documented protocol; exact published numbers additionally
require the original token cache and checkpoints. Throughput depends on the
GPU and software environment.

| Experiment | Entry points | Collected result |
| --- | --- | --- |
| Main training at both scales | `training.train`, `recipes/paper/*.yaml` | Complete `final/` checkpoint |
| Held-out quality | `evals.heldout` | `heldout.json` |
| Exact-AR quality | `evals.heldout_ar` for WhiteMatter/LCKV | `heldout_ar.json` |
| All downstream table tasks | `evals.lm_eval` | `lm_eval.json` |
| Quality tables and model sizes | `evals.paper` | `quality.csv` |
| Controlled prefill convergence | `studies.prefill_convergence` | Joined quality/timing JSON |
| Batch-64 inference throughput/memory | `benchmarks.paper`, `benchmarks.paper_report` | `runtime.csv` |
| Schedule matrix | `studies.schedules` | Per-seed and mean-curve metric CSVs |
| Rank and connectivity | `studies.rank`, `studies.shared_mixture` | Complete ablation CSV |
| Main-body compute-cost ratios | `benchmarks.flops` | Per-operation and per-token FLOP JSON |

### Main quality suite

Use `outputs/<recipe-name>/` for each model, with the final HF export at
`final/`. Run held-out and downstream evaluation on every main recipe except
`lckv_w13_1p3b` (a runtime control). For WhiteMatter and LCKV also run exact AR:

```bash
python -m evals.heldout --model outputs/white_matter_k8/final \
  --data-dir /path/to/cache --output outputs/white_matter_k8/heldout.json
python -m evals.heldout_ar --model outputs/white_matter_k8/final \
  --data-dir /path/to/cache --output outputs/white_matter_k8/heldout_ar.json
python -m evals.lm_eval --model outputs/white_matter_k8/final \
  --output outputs/white_matter_k8/lm_eval.json
```

The default task list covers every displayed main-body task, including BLiMP,
SciQ, ReCoRD, and SQuAD-Completion v1. WhiteMatter prefill defaults to its
configured execution schedule. Task subsets may be saved as `lm_eval_*.json`
inside each model directory. Keep the checkpoint, harness revision, and
protocol identical across subsets. Full result collection requires all nine
models, complete splits, final checkpoints, and matching checkpoint hashes:

```bash
python -m evals.paper --results-dir outputs --output outputs/quality.csv
```

The collector averages the 11 unrounded non-perplexity scores. WikiText uses
word perplexity; LAMBADA uses token perplexity. Parameter counts count the tied
embedding/output matrix once. KV fractions are relative to each architecture's
own full-depth cache; `kv_channels` also permits comparisons across depths.

Quality evaluation retains FP32 checkpoint weights with BF16 decoder autocast
on CUDA. Held-out scoring also uses BF16 head autocast; downstream likelihoods
use an FP32 head, matching the archived harness. Runtime benchmarks instead
store parameters in BF16.

### Runtime and FLOPs

```bash
python -m benchmarks.paper --dry-run
python -m benchmarks.paper --output outputs/runtime
python -m benchmarks.paper_report outputs/runtime/<prefill-run> outputs/runtime/<decode-run> \
  --output outputs/runtime.csv
python -m benchmarks.flops --output outputs/flops.json
```

The runtime preset measures four architectures at batch 64, prompt length
2048, and 128 timed decode steps. It uses the full resident prefill batch;
capacity search is a separate optional benchmark. The report rejects missing
cases, model configurations that differ from the preset recipes, non-BF16
parameter storage, protocol mismatches, and mixed GPU types. Use RTX A6000 for comparison
with the paper.

FLOPs count actual decoder forward/backward execution with the production
CUDA attention operators, two operations per multiply-add, causal pair counts,
and the five-GEMM attention-backward convention. They exclude the LM head,
optimizer, and document segmentation. Compiler fusion is disabled during
counting. Selective checkpointing saves matrix products; any remaining
recomputed matrix products are visible in the per-operation record. FusedKV
uses the paper's cached inference workload: all source-layer prompt tokens,
only the last token through reconstruction layers, and explicit cache-fusion
arithmetic. Vanilla and FusedKV counts are checked against independent formulas.

### Figures

Install `pip install -e '.[analysis]'` to render PDF/PNG figures from collected
results, without the manuscript checkout:

```bash
python scripts/plot_experiments.py quality --input outputs/quality.csv --output outputs/quality.pdf
python scripts/plot_experiments.py runtime --input outputs/runtime.csv --output outputs/runtime.pdf
python scripts/plot_experiments.py schedules --input outputs/studies/schedules/mean.csv --output outputs/schedules.pdf
python scripts/plot_experiments.py rank --input outputs/studies/rank/summary.csv --output outputs/rank.pdf
```

See [controlled convergence](../studies/prefill_convergence/README.md) for its
full data/training/measurement workflow and figure command. Each ablation's
README documents its training and evaluation protocol.

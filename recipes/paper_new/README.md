# New paper experiments

This recipe collection repeats the paper's main quality comparisons with two WhiteMatter
settings: `use_dummy_token: false` and `include_top_output: true`. Each token
attends to earlier tokens in its document. The first token receives zero
attention output. The mixer uses the feedback block input and all L layer
outputs, giving L+1 sources.

## Models and training budgets

| Scale  | Models                                                        |   Training tokens | Global batch |
| ------ | ------------------------------------------------------------- | ----------------: | -----------: |
| D=512  | WhiteMatter k8/k16, vanilla 16/24 layers, LCKV w4/w7, FusedKV |  Approximately 8B |          128 |
| D=1792 | WhiteMatter k14, vanilla 28 layers                            | Approximately 10B |           32 |

All nine recipes are self-contained in this directory. The baseline recipes match
`recipes/paper/`. WhiteMatter retains the paper's optimizer, seed, depth, width,
KV channel count, and pass schedule. L+1 changes the source pool and router
parameter counts; evaluation counts parameters from each checkpoint.

Analysis recipes are grouped in `rank/`, `schedules/`, and `shared_mixture/`.
They retain the 20,000-step, batch-8 budgets of the original studies. The
48 schedule recipes cover both seeds; rank has ten arms, and shared-mixture
has one arm. The depth-causal control uses no dummy and only the available
source prefix at each layer, so `include_top_output` remains false.

Train an analysis recipe on one GPU:

```bash
torchrun --standalone --nproc-per-node=1 -m studies.train_new \
  --recipe recipes/paper_new/rank/k8.yaml \
  --data-dir data/paper_new --output-dir outputs/paper_new/rank/k8
```

`studies.train_new` checks the matched recipe and continuous-source cache.
The original study evaluators still validate the original protocols; their
migration is separate from these training runs.

## Data

Install the research dependencies and pinned evaluation harness described in
the repository [README](../../README.md) and
[reproduction guide](../../docs/reproduction.md).

All models use the continuous-source FineWeb-Edu cache
(`ordering_protocol: continuous-source-v1`). Training reads its rows
sequentially, with `shuffle=False`, including distributed training.
The cache uses EOS packing and 2048-token rows.

To build this cache:

```bash
python scripts/prepare_fineweb_edu.py --output data/paper_new
```

For an existing cache, pass its directory as `--data-dir` in the commands below.
Use the same cache for every baseline, WhiteMatter model, and held-out
evaluation. Record its metadata and a SHA-256 checksum of `tokenized.npy` with
the results. The packing order changes the held-out rows, so rerun baseline
training and evaluation on this cache as well.

## Train

Run from the repository root on eight GPUs:

```bash
torchrun --standalone --nproc-per-node=8 -m training.train \
  --recipe recipes/paper_new/white_matter_k8.yaml \
  --data-dir data/paper_new \
  --output-dir outputs/paper_new/white_matter_k8
```

Repeat for every recipe, using its filename stem as the output directory name.
The shared [Slurm launcher](../../slurm/README.md) accepts these recipe paths:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  slurm/train.sbatch \
  recipes/paper_new/white_matter_k8.yaml \
  /path/to/cache outputs/paper_new/white_matter_k8
```

Training compiles tensor execution by default. Each output directory contains
resume state and a final Hugging Face export. Start with fresh directories for
this recipe collection; subsequent launches resume their own checkpoints.

## Evaluate and collect

Run held-out perplexity and the complete downstream suite for all nine models:

```bash
python -m evals.heldout --model outputs/paper_new/white_matter_k8/final \
  --data-dir data/paper_new --output outputs/paper_new/white_matter_k8/heldout.json
python -m evals.lm_eval --model outputs/paper_new/white_matter_k8/final \
  --output outputs/paper_new/white_matter_k8/lm_eval.json
```

Also evaluate exact autoregressive perplexity for each WhiteMatter and LCKV
checkpoint:

```bash
python -m evals.heldout_ar --model outputs/paper_new/white_matter_k8/final \
  --data-dir data/paper_new --output outputs/paper_new/white_matter_k8/heldout_ar.json
```

Keep the harness revision and evaluation settings fixed across models. Once
all evaluations finish, collect the table against these recipes:

```bash
python -m evals.paper --results-dir outputs/paper_new \
  --recipe-dir recipes/paper_new --output outputs/paper_new/quality.csv
python scripts/plot_experiments.py quality --input outputs/paper_new/quality.csv \
  --output outputs/paper_new/quality.pdf
```

The collector checks model settings, final training steps, complete evaluation
splits, checkpoint hashes, and harness consistency. In particular, it rejects
WhiteMatter checkpoints with a dummy slot or an L-source pool. Keep the source
revision, cache fingerprint, recipes, and raw evaluation JSONs with each table.

# Slurm

Supply cluster-specific account, partition, module, and filesystem settings at
submission time or in a site-local wrapper. Training recipes remain portable
across clusters.

The paper trained on one node with eight GPUs:

```bash
sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_PARTITION \
  slurm/train.sbatch \
  recipes/paper/white_matter_k8.yaml \
  /path/to/fineweb_edu_cache \
  outputs/white_matter_k8
```

The recipe fixes every value that can change learned weights. Slurm controls
only resources and placement. Requeuing is safe: a job automatically resumes
`OUTPUT_DIR/ckpt_full.pt` with model, optimizer, data position, and RNG state.

Held-out perplexity and lm-eval each use one GPU:

```bash
sbatch slurm/evaluate_heldout.sbatch \
  outputs/white_matter_k8/final \
  /path/to/fineweb_edu_cache \
  outputs/white_matter_k8/heldout.json

sbatch slurm/evaluate_lm_eval.sbatch \
  outputs/white_matter_k8/final \
  Qwen/Qwen3-0.6B-Base \
  outputs/white_matter_k8/lm_eval.json
```

Prefill/decode throughput and memory-capacity search also use one GPU. Install
`.[benchmarks]`, activate that environment, and pass the benchmark CLI options:

```bash
sbatch slurm/benchmark_generation.sbatch --models /path/to/wm /path/to/vanilla \
  --batch-sizes 1 2 4 8 --prompt-lengths 128 512 1024 2048 \
  --tokens 128 --memory-budget-gib 40 --output outputs/benchmarks
```

See [the measurement protocol](../docs/inference.md). Use an otherwise idle GPU
and identical precision and execution options for both model families.

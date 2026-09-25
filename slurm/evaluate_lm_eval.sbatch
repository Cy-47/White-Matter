#!/bin/bash
#SBATCH --job-name=white-matter-lm-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: sbatch slurm/evaluate_lm_eval.sbatch MODEL TOKENIZER OUTPUT_JSON" >&2
  exit 2
fi

python -m evals.lm_eval \
  --model "$1" \
  --tokenizer "$2" \
  --output "$3"

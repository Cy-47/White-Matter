#!/bin/bash
#SBATCH --job-name=paper-ar-heldout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "usage: sbatch slurm/evaluate_paper_ar_heldout.sbatch MODEL DATA_DIR OUTPUT_JSON [EVAL_OPTIONS...]" >&2
  exit 2
fi
model=$1
data_dir=$2
output=$3
shift 3
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec python -m evals.heldout_ar \
  --model "$model" --data-dir "$data_dir" --output "$output" "$@"

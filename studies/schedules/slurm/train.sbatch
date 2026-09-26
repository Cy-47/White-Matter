#!/bin/bash
#SBATCH --job-name=wm-schedules
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ "$#" -ne 3 ]; then
  echo "usage: sbatch studies/schedules/slurm/train.sbatch RECIPE DATA_DIR OUTPUT_DIR" >&2
  exit 2
fi

RECIPE=$1
DATA_DIR=$2
OUTPUT_DIR=$3
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$OUTPUT_DIR/.torch/inductor}

torchrun --standalone --nproc-per-node=1 \
  -m studies.schedules.train \
  --recipe "$RECIPE" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR"

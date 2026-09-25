#!/bin/bash
#SBATCH --job-name=white-matter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: sbatch slurm/train.sbatch RECIPE DATA_DIR OUTPUT_DIR" >&2
  exit 2
fi

RECIPE=$1
DATA_DIR=$2
OUTPUT_DIR=$3
GPU_COUNT=${SLURM_GPUS_ON_NODE:-8}

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Preserve compiler artifacts across allocations using PyTorch's native setting.
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$OUTPUT_DIR/.torch/inductor}

torchrun \
  --standalone \
  --nproc-per-node="$GPU_COUNT" \
  -m training.train \
  --recipe "$RECIPE" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR"

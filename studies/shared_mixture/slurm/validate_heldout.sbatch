#!/bin/bash
#SBATCH --job-name=wm-heldout-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$SLURM_SUBMIT_DIR/outputs/studies/shared_mixture/validation/.torch/inductor}
python -c 'import torch, tilelang, flash_attn; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), flush=True)'
python -m pytest tests/gpu/test_shared_mixture_heldout.py -m gpu -vv --tb=short

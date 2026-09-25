#!/bin/bash
#SBATCH --job-name=wm-shared-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ "$#" -ne 3 ]; then
  echo "usage: sbatch studies/shared_mixture/slurm/evaluate.sbatch MODEL DATA_DIR OUTPUT_JSON" >&2
  exit 2
fi

OUTPUT=$3
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$(dirname "$OUTPUT")/.torch/inductor}
python -m studies.shared_mixture.evaluate_heldout \
  --model "$1" --data-dir "$2" --output "$OUTPUT"

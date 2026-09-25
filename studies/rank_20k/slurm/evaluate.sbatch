#!/bin/bash
#SBATCH --job-name=wm-rank-eval
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
  echo "usage: sbatch studies/rank_20k/slurm/evaluate.sbatch MODEL DATA_DIR OUTPUT_JSON" >&2
  exit 2
fi

OUTPUT=$3
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$(dirname "$OUTPUT")/.torch/inductor}
python -m studies.rank_20k.evaluate \
  --model "$1" --data-dir "$2" --output "$OUTPUT"

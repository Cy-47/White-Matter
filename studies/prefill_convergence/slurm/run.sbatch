#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=04:00:00
#SBATCH --job-name=contiguous_convergence
#SBATCH --output=slurm_logs/%j_%x.out
set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/wm-convergence-${SLURM_JOB_ID}"

ACTION=${1:?usage: run.sbatch validate|quality|timing MODEL WINDOWS OUTPUT [extra arguments]}
shift
case "$ACTION" in
  validate)
    python -m pytest tests/gpu/test_contiguous_convergence.py -q -s \
      --junitxml="outputs/convergence-validation-${SLURM_JOB_ID}.xml"
    ;;
  quality|timing)
    MODEL=${1:?model}; WINDOWS=${2:?windows}; OUTPUT=${3:?output}
    shift 3
    MODULE=evaluate
    if [[ "$ACTION" == timing ]]; then MODULE=benchmark; fi
    python -m "studies.prefill_convergence.${MODULE}" \
      --model "$MODEL" --windows "$WINDOWS" --output "$OUTPUT" "$@"
    ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac

#!/bin/bash
#SBATCH --job-name=paper-additions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "usage: sbatch slurm/evaluate_paper_additions.sbatch MODEL OUTPUT_JSON {smoke|likelihood|generation} [EVAL_OPTIONS...]" >&2
  exit 2
fi
model=$1
output=$2
group=$3
shift 3
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
case "$group" in
  smoke) tasks=(blimp sciq record squad_completion); extras=(--limit 2 --no-compile) ;;
  likelihood) tasks=(blimp sciq record); extras=() ;;
  generation) tasks=(squad_completion); extras=(--no-compile) ;;
  *) echo "unknown task group: $group" >&2; exit 2 ;;
esac
python -m evals.lm_eval \
  --model "$model" --output "$output" \
  --tasks "${tasks[@]}" "${extras[@]}" "$@"

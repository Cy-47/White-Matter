#!/bin/bash
#SBATCH --job-name=white-matter-validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Submit from the repository root with the package and GPU extras installed.
PYTHON=${PYTHON:-python}
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_COMPILE_THREADS=2
REPORT_DIR=${REPORT_DIR:-outputs/validation-$SLURM_JOB_ID}
mkdir -p "$REPORT_DIR"
"$PYTHON" -c 'import torch, flash_attn, tilelang; assert torch.cuda.device_count() >= 2; print(torch.__version__, torch.cuda.get_device_name(), flush=True)'
"$PYTHON" -m pytest tests/gpu tests/unit/test_generation.py -m gpu -v --tb=short --junitxml="$REPORT_DIR/gpu.xml"
"$PYTHON" - "$REPORT_DIR/gpu.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
assert root.findall(".//testcase"), "no GPU tests collected"
assert not root.findall(".//skipped"), "the production gate must not skip tests"
PY

"""Run the main-body batch-64 inference workload through the shared benchmark."""

import argparse
import subprocess
import sys
from pathlib import Path

RECIPES = (
    "recipes/paper/vanilla_1p3b.yaml",
    "recipes/benchmarks/feedback_transformer_1p3b.yaml",
    "recipes/paper/lckv_w13_1p3b.yaml",
    "recipes/paper/white_matter_1p3b.yaml",
)
PARAMETER_DTYPE = "bfloat16"


def commands(output):
    common = [
        sys.executable,
        "-m",
        "benchmarks.generation",
        "--recipes",
        *RECIPES,
        "--batch-sizes",
        "64",
        "--prompt-lengths",
        "2048",
        "--tokens",
        "129",
        "--prefill-batch-size",
        "0",
        "--warmups",
        "3",
        "--repetitions",
        "5",
        "--parameter-dtype",
        PARAMETER_DTYPE,
        "--compiled",
        "--cuda-graph",
        "--output",
        str(output),
    ]
    return [common + ["--phase", phase] for phase in ("prefill", "decode")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/runtime"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for command in commands(args.output):
        if args.dry_run:
            import shlex

            print(shlex.join(command))
        else:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

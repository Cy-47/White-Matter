"""Collect the matched three-pass Figure 7b ablation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from studies.paper_20k import PAPER_TEST_TARGETS


BASELINE_ARMS = ("k1", "k2", "k4", "k8", "k12", "k16", "k1_static", "k16_static", "vanilla")


def collect(root: Path, *, include_depth_causal: bool = False) -> list[dict]:
    arms = (*BASELINE_ARMS, "k16_depth_causal") if include_depth_causal else BASELINE_ARMS
    rows = []
    for arm in arms:
        path = root / arm / "heldout.json"
        result = json.loads(path.read_text())
        expected_passes = 1 if arm in {"vanilla", "k16_depth_causal"} else 3
        if (
            result.get("protocol") != "figure7b_sequential_final"
            or result.get("arm") != arm
            or result.get("n_tok") != PAPER_TEST_TARGETS
            or result.get("n_passes") != expected_passes
        ):
            raise ValueError(f"invalid Figure 7b result: {path}")
        rows.append({
            "arm": arm,
            "n_passes": expected_passes,
            "perplexity": result["perplexity"],
            "lm_ce": result["lm_ce"],
            "trainable_parameters": result["trainable_parameters"],
            "checkpoint": result["checkpoint"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Figure 7b ablation results.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-depth-causal", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    rows = collect(args.results_dir, include_depth_causal=args.include_depth_causal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()

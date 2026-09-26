"""Aggregate the Figure 7a pass curves across the two training seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from studies.protocol import PAPER_TEST_TARGETS
from studies.schedules.matrix import GRAD, MODES, NO_GRAD, SEEDS, evaluation_horizon


def _read_curve(path: Path, seed: int, arm: str, mode: str) -> dict[int, float]:
    result = json.loads(path.read_text())
    if (
        result.get("protocol") != "figure7a_sequential_final"
        or result.get("seed") != seed
        or result.get("arm") != arm
        or result.get("evaluation_mode") != mode
        or result.get("n_tok") != PAPER_TEST_TARGETS
    ):
        raise ValueError(f"invalid Figure 7a result: {path}")
    curve = {int(row["n_passes"]): float(row["perplexity"]) for row in result["rows"]}
    count = len(result["rows"])
    minimum = evaluation_horizon(arm, mode)
    if count < minimum or set(curve) != set(range(1, count + 1)) or (mode != "tp" and count != 32):
        raise ValueError(f"Figure 7a {mode} curve requires consecutive passes from 1 through {minimum}"
                         f"{' or beyond' if mode == 'tp' else ''}: {path}")
    if any(not math.isfinite(value) or value <= 0 for value in curve.values()):
        raise ValueError(f"invalid perplexity in {path}")
    return curve


def curve_metrics(native, cyclic16, tp) -> dict:
    """Derive metrics after any seed averaging, with earliest-pass tie breaking."""
    best_pass = min(sorted(native), key=native.get)
    best = native[best_pass]
    crossing = next((n for n in sorted(tp) if tp[n] <= best * 1.01), None)
    return {
        "native_best_perplexity": best,
        "native_best_pass": best_pass,
        "cyclic16_pass32_perplexity": cyclic16[32],
        "jacobi_passes_within_1pct": crossing,
        "jacobi_passes_within_1pct_censored": crossing is None,
        "jacobi_max_evaluated_pass": max(tp),
    }


def collect(root: Path) -> tuple[list[dict], list[dict]]:
    per_seed, averaged = [], []
    for no_grad in NO_GRAD:
        for grad in GRAD:
            for mode in MODES:
                arm = f"ng{no_grad}_g{grad}_{mode}"
                identity = dict(arm=arm, no_gradient_passes=no_grad, gradient_passes=grad, schedule=mode)
                curves = []
                for seed in SEEDS:
                    directory = root / f"seed{seed}" / arm
                    native = _read_curve(directory / "eval_native.json", seed, arm, "native")
                    cyclic16 = native if mode == "c16" else _read_curve(
                        directory / "eval_cyclic16.json", seed, arm, "cyclic16",
                    )
                    tp_path = directory / "eval_tp.json"
                    tp = native if mode == "tp" and not tp_path.exists() else _read_curve(tp_path, seed, arm, "tp")
                    curves.append((native, cyclic16, tp))
                    per_seed.append(dict(seed=seed, **identity, **curve_metrics(native, cyclic16, tp)))
                if any(set(curve[2]) != set(curves[0][2]) for curve in curves[1:]):
                    raise ValueError(f"Figure 7a Jacobi pass horizons differ across seeds: {arm}")
                means = [
                    {p: sum(curve[index][p] for curve in curves) / len(curves) for p in curves[0][index]}
                    for index in range(3)
                ]
                averaged.append(dict(**identity, **curve_metrics(*means)))
    return per_seed, averaged


def _write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Figure 7a held-out pass curves.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--per-seed-output", type=Path, required=True)
    parser.add_argument("--averaged-output", type=Path, required=True)
    args = parser.parse_args()
    rows, averaged = collect(args.results_dir)
    _write_csv(args.per_seed_output, rows)
    _write_csv(args.averaged_output, averaged)


if __name__ == "__main__":
    main()

"""Aggregate the Figure 7a pass curves across the two training seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from studies.paper_20k import PAPER_TEST_TARGETS
from studies.schedules_20k.matrix import GRAD, MODES, NO_GRAD, SEEDS


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
    if set(curve) != set(range(1, 33)):
        raise ValueError(f"Figure 7a pass curve must cover 1..32: {path}")
    return curve


def collect(root: Path) -> tuple[list[dict], list[dict]]:
    per_seed = []
    for seed in SEEDS:
        for no_grad in NO_GRAD:
            for grad in GRAD:
                for mode in MODES:
                    arm = f"ng{no_grad}_g{grad}_{mode}"
                    directory = root / f"seed{seed}" / arm
                    native = _read_curve(directory / "eval_native.json", seed, arm, "native")
                    cyclic16 = (
                        native if mode == "c16" else _read_curve(directory / "eval_cyclic16.json", seed, arm, "cyclic16")
                    )
                    tp = native if mode == "tp" else _read_curve(directory / "eval_tp.json", seed, arm, "tp")
                    best_pass = min(native, key=native.get)
                    best = native[best_pass]
                    within = next((n for n in range(1, 33) if tp[n] <= best * 1.01), None)
                    per_seed.append({
                        "seed": seed, "arm": arm, "no_gradient_passes": no_grad,
                        "gradient_passes": grad, "schedule": mode,
                        "native_best_perplexity": best,
                        "native_best_pass": best_pass,
                        "cyclic16_pass32_perplexity": cyclic16[32],
                        "jacobi_passes_within_1pct": within,
                    })
    averaged = []
    for no_grad in NO_GRAD:
        for grad in GRAD:
            for mode in MODES:
                arm = f"ng{no_grad}_g{grad}_{mode}"
                rows = [row for row in per_seed if row["arm"] == arm]
                if len(rows) != len(SEEDS):
                    raise RuntimeError(f"missing seed result for {arm}")
                summary = {"arm": arm, "no_gradient_passes": no_grad, "gradient_passes": grad, "schedule": mode}
                for key in ("native_best_perplexity", "native_best_pass", "cyclic16_pass32_perplexity", "jacobi_passes_within_1pct"):
                    values = [row[key] for row in rows]
                    summary[key] = None if any(value is None for value in values) else sum(values) / len(values)
                averaged.append(summary)
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

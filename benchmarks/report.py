"""Regenerate benchmark summaries and CSV without loading Torch or a model."""

import argparse
import csv
import json
from pathlib import Path

from benchmarks._measurement import fits_memory, write_json


def summarize_run(directory: Path) -> dict:
    run = json.loads((directory / "run.json").read_text())
    cases = [json.loads(path.read_text()) for path in sorted((directory / "cases").glob("*.json"))]
    rows = []
    for case in cases:
        row = dict(case_id=case["case_id"], status=case["status"], **case["workload"])
        result = case.get("result", {})
        row["memory"] = result.get("memory")
        for name in ("prefill", "decode", "total", "training", "forward_backward"):
            timing = result.get(name)
            row[name] = {k: v for k, v in timing.items() if k != "samples_seconds"} if timing else None
        rows.append(row)
    capacity = []
    protocol = run["protocol"]
    groups = {(r["model"], r["prompt_length"], r["repeat"]) for r in rows if "model" in r}
    for model, length, repeat in sorted(groups) if protocol.get("memory_budget_gib") else []:
        selected = [
            c
            for c in cases
            if tuple(c["workload"].get(k) for k in ("model", "prompt_length", "repeat")) == (model, length, repeat)
            and c["workload"]["batch_size"] <= protocol["max_batch_size"]
        ]
        for budget in protocol.get("memory_budget_gib", []):
            limit = (budget - protocol["memory_headroom_gib"]) * 2**30
            admitted = [c for c in selected if fits_memory(c, limit)]
            maximum = max((c["workload"]["batch_size"] for c in admitted), default=0)
            rejected = [
                c["workload"]["batch_size"]
                for c in selected
                if c["status"] in {"ok", "cuda_oom"} and not fits_memory(c, limit)
            ]
            metric = {"end-to-end": "total"}.get(protocol["phase"], protocol["phase"])
            best = max(admitted, key=lambda c: c["result"][metric]["tokens_per_second"] or 0) if admitted else None
            capacity.append(
                {
                    "model": model,
                    "prompt_length": length,
                    "repeat": repeat,
                    "budget_gib": budget,
                    "maximum_feasible_batch": maximum,
                    "search_capped": maximum == protocol["max_batch_size"],
                    "boundary_checked": maximum + 1 in rejected,
                    "nonmonotonic": any(b < maximum for b in rejected),
                    "best_measured_throughput_batch": best["workload"]["batch_size"] if best else None,
                    "best_case_id": best["case_id"] if best else None,
                }
            )
    summary = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "measurements": rows,
        "capacity": capacity,
        "complete": bool(rows)
        and run.get("completed_cases") == sorted(r["case_id"] for r in rows)
        and all(r["status"] in {"ok", "cuda_oom", "profiled"} for r in rows),
    }
    lines = ["| Model | Phase | Batch | Tokens/s | Accounted GiB |", "|---|---|---:|---:|---:|"]
    for row in rows:
        for phase in ("prefill", "decode", "total", "training", "forward_backward"):
            timing = row.get(phase)
            if timing and timing.get("tokens_per_second") is not None:
                memory = (row.get("memory") or {}).get("budget_accounted_bytes")
                gib = f"{memory / 2**30:.2f}" if memory is not None else "—"
                lines.append(
                    f"| {Path(row.get('model', 'attention')).name} | {phase} | "
                    f"{row.get('batch_size', 1)} | {timing['tokens_per_second']:,.0f} | {gib} |"
                )
    (directory / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json(directory / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    summary = summarize_run(args.run)
    if args.csv:
        fields = [
            "run_id",
            "case_id",
            "status",
            "model",
            "repeat",
            "batch_size",
            "prompt_length",
            "workload_phase",
            "phase",
            "median_seconds",
            "tokens_per_second",
            "budget_accounted_bytes",
        ]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for case in summary["measurements"]:
                for phase in ("prefill", "decode", "total", "training", "forward_backward"):
                    timing = case[phase]
                    if timing:
                        writer.writerow(
                            dict(
                                run_id=summary["run_id"],
                                **{k: case.get(k) for k in fields[1:6]},
                                prompt_length=case.get("prompt_length", case.get("length")),
                                workload_phase=case.get("phase"),
                                phase=phase,
                                median_seconds=timing["median_seconds"],
                                tokens_per_second=timing["tokens_per_second"],
                                budget_accounted_bytes=(case["memory"] or {}).get("budget_accounted_bytes"),
                            )
                        )
    print(args.run / "summary.json")


if __name__ == "__main__":
    main()

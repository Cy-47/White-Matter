"""Collect the eight matched batch-64 runtime cases into a paper-ready table."""

import argparse
import csv
import json
from pathlib import Path

from benchmarks.paper import PARAMETER_DTYPE, RECIPES


def collect(directories):
    from benchmarks.generation import configure_prefill_mode
    from training.recipes import load_recipe, model_recipe_keys

    expected_models = {}
    for recipe in RECIPES:
        config = load_recipe(Path(__file__).resolve().parents[1] / recipe).model
        config.document_separator_token_id = None
        configure_prefill_mode(config)
        keys = model_recipe_keys(type(config)) | {"execution_mode", "prefill_mode"}
        expected_models[config.model_type] = {key: getattr(config, key) for key in keys if hasattr(config, key)}
    cases = {}
    hardware = None
    for directory in directories:
        run = json.loads((directory / "run.json").read_text())
        files = list((directory / "cases").glob("*.json"))
        if sorted(run["completed_cases"] or []) != sorted(p.stem for p in files):
            raise ValueError("runtime run is incomplete")
        for path in files:
            case = json.loads(path.read_text())
            work = case["workload"]
            if (
                case["status"] != "ok"
                or work["batch_size"] != 64
                or work["prompt_length"] != 2048
                or work["tokens"] != 129
                or not work["compiled"]
                or not work["cuda_graph"]
                or work["prefill_batch_size"] != 0
                or work["repetitions"] != 5
                or work.get("parameter_dtype") != PARAMETER_DTYPE
            ):
                raise ValueError(f"case differs from paper runtime protocol: {path}")
            result = case["result"]
            family, phase = result["config"]["model_type"], work["phase"]
            if family not in expected_models or phase not in ("prefill", "decode") or (family, phase) in cases:
                raise ValueError("unexpected or duplicate runtime case")
            for key, value in expected_models[family].items():
                if key not in result["config"] or result["config"][key] != value:
                    raise ValueError(f"{path}: {key} differs from the paper runtime recipe (expected {value!r})")
            gpu = result["environment"]["gpu"]
            if hardware is not None and hardware != gpu:
                raise ValueError("runtime cases use different GPU types")
            hardware = gpu
            cases[family, phase] = {
                "model": family,
                "phase": phase,
                "tokens_per_second": result[phase]["tokens_per_second"],
                "peak_gib": result["memory"]["budget_accounted_bytes"] / 2**30,
                "gpu": gpu,
                "source_sha256": run["source"]["sha256"],
            }
    if set(cases) != {(family, phase) for family in expected_models for phase in ("prefill", "decode")}:
        raise ValueError("need all four architectures in both phases")
    rows = []
    for phase in ("prefill", "decode"):
        for family in expected_models:
            row = cases[family, phase]
            row["relative_throughput"] = row["tokens_per_second"] / cases["vanilla", phase]["tokens_per_second"]
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = collect(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

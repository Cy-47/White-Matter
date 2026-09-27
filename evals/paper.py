"""Current main-body task selection and complete quality-table collection."""

import argparse
import csv
import json
import math
from pathlib import Path

# task, metric, full evaluation sample count; scores remain unrounded fractions.
METRICS = (
    ("wikitext", "word_perplexity", None),
    ("lambada_openai", "perplexity", 5153),
    ("lambada_openai", "acc", 5153),
    ("blimp", "acc", 67000),
    ("piqa", "acc", 1838),
    ("hellaswag", "acc_norm", 10042),
    ("arc_easy", "acc", 2376),
    ("arc_challenge", "acc_norm", 1172),
    ("winogrande", "acc", 1267),
    ("openbookqa", "acc_norm", 500),
    ("sciq", "acc", 1000),
    ("record", "em", 10000),
    ("squad_completion", "contains", 2984),
)
PAPER_TASKS = list(dict.fromkeys(task for task, _, _ in METRICS))
PAPER_MODELS = (
    "vanilla_16l",
    "fusedkv",
    "lckv_w4",
    "lckv_w7",
    "white_matter_k8",
    "white_matter_k16",
    "vanilla_24l",
    "vanilla_1p3b",
    "white_matter_1p3b",
)


def downstream_metrics(payloads):
    results, samples, versions = {}, {}, {}
    identity = None
    for payload in payloads:
        if payload.get("limit") is not None or payload.get("sample_range") is not None:
            raise ValueError("paper tables require complete evaluation splits")
        if payload.get("num_fewshot") != 0 or payload.get("max_length") != 1024:
            raise ValueError("paper tables require zero-shot context 1024")
        current = (payload.get("checkpoint"), payload.get("model_config"), payload.get("harness"))
        if not all(current):
            raise ValueError("missing checkpoint/configuration/harness provenance")
        if identity is not None and current != identity:
            raise ValueError("downstream artifacts have different checkpoints or protocols")
        identity = current
        for task, result in payload["results"].items():
            if task in results and result != results[task]:
                raise ValueError(f"conflicting task result: {task}")
            results[task] = result
        samples.update(payload.get("n_samples", {}))
        versions.update(payload.get("task_versions", {}))
    output = {}
    for task, metric, count in METRICS:
        key = f"{metric},none"
        if task == "blimp" and task not in results:
            parts = [r["acc,none"] for name, r in results.items() if name.startswith("blimp_")]
            if len(parts) != 67:
                raise ValueError("BLiMP requires all 67 subtasks")
            value = sum(parts) / 67
        else:
            value = results[task][key]
        measured = samples.get(task, {}).get("effective")
        if task == "blimp" and measured is None:
            measured = sum(v.get("effective", 0) for name, v in samples.items() if name.startswith("blimp_"))
        if count is not None and measured != count:
            raise ValueError(f"{task}: expected {count} samples, got {measured}")
        if task == "squad_completion" and versions.get(task) != 1:
            raise ValueError("SQuAD completion task version 1 required")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"invalid metric: {task}/{metric}")
        output[f"{task}/{metric}"] = value
    output["average"] = sum(output[f"{t}/{m}"] for t, m, _ in METRICS[2:]) / 11
    return output, identity


def collect(root):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from white_matter.models import register_models

    register_models()
    rows = []
    harness = None
    for name in PAPER_MODELS:
        directory = root / name
        heldout = json.loads((directory / "heldout.json").read_text())
        if heldout["sequences"] != 5000 or heldout["tokens"] != 5000 * 2047:
            raise ValueError(f"{name}: incomplete held-out result")
        paths = sorted(directory.glob("lm_eval*.json"))
        if not paths:
            raise ValueError(f"{name}: missing downstream artifacts")
        scores, identity = downstream_metrics([json.loads(p.read_text()) for p in paths])
        if heldout.get("checkpoint") != identity[0]:
            raise ValueError(f"{name}: held-out and downstream checkpoints differ")
        from benchmarks._measurement import checkpoint_files
        from training.recipes import load_recipe

        if checkpoint_files(str(directory / "final")) != identity[0]:
            raise ValueError(f"{name}: final checkpoint differs from evaluated weights")
        if harness is not None and identity[2] != harness:
            raise ValueError("models were evaluated with different harness revisions")
        harness = identity[2]
        recipe = load_recipe(Path(__file__).parents[1] / "recipes/paper" / f"{name}.yaml")
        settings = identity[1]
        if (
            settings.get("num_passes", 1) != getattr(recipe.model, "num_passes", 1)
            or settings.get("execution_mode") != recipe.model.execution_mode
        ):
            raise ValueError(f"{name}: evaluation schedule differs from paper recipe")
        config = AutoConfig.from_pretrained(directory / "final")
        if getattr(config, "training_step", None) != recipe.steps:
            raise ValueError(f"{name}: checkpoint is not the final training step")
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        parameters = model.num_parameters()
        non_embedding = parameters - model.get_input_embeddings().weight.numel()
        layers = config.num_hidden_layers
        channels = (
            config.num_kv_channels
            if config.model_type == "white_matter"
            else config.num_pre_layers + config.num_post_layers + 1
            if config.model_type == "lckv"
            else layers // 2
            if config.model_type == "fusedkv"
            else layers
        )
        row = dict(
            model=name,
            parameters=parameters,
            non_embedding_parameters=non_embedding,
            kv_channels=channels,
            kv_fraction=channels / layers,
            kv_relative_to_vanilla=channels / (16 if config.hidden_size == 512 else 28),
            heldout_perplexity=heldout["perplexity"],
            **scores,
        )
        if config.model_type in {"white_matter", "lckv"}:
            ar = json.loads((directory / "heldout_ar.json").read_text())
            if (
                ar.get("checkpoint") != identity[0]
                or ar.get("mode") != "ar"
                or ar.get("sequences") != 5000
                or ar.get("tokens") != 5000 * 2047
            ):
                raise ValueError(f"{name}: invalid exact-AR result")
            row["ar_perplexity"] = ar["perplexity"]
        else:
            row["ar_perplexity"] = None
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = collect(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

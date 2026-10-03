"""Paper runtime reports require the prescribed architecture and precision."""

import json
from pathlib import Path

import pytest

from benchmarks.generation import configure_prefill_mode
from benchmarks.paper import commands
from benchmarks.paper_report import collect
from training.recipes import load_recipe


@pytest.fixture
def runtime_run(tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = commands(tmp_path)[0]
    assert command[command.index("--attention-backend") + 1] == "flash_attention_2"
    recipes = command[command.index("--recipes") + 1 : command.index("--batch-sizes")]
    directory = tmp_path / "run"
    (directory / "cases").mkdir(parents=True)
    names = []
    for recipe in recipes:
        config = load_recipe(root / recipe).model
        config.document_separator_token_id = None
        configure_prefill_mode(config)
        for phase in ("prefill", "decode"):
            name = f"{config.model_type}-{phase}"
            names.append(name)
            case = {
                "status": "ok",
                "workload": {
                    "batch_size": 64,
                    "prompt_length": 2048,
                    "tokens": 129,
                    "compiled": True,
                    "cuda_graph": True,
                    "prefill_batch_size": 0,
                    "repetitions": 5,
                    "parameter_dtype": "bfloat16",
                    "phase": phase,
                },
                "result": dict(
                    config=config.to_dict(),
                    environment={"gpu": "RTX A6000"},
                    memory={"budget_accounted_bytes": 2**30},
                    **{phase: {"tokens_per_second": 100}},
                ),
            }
            (directory / "cases" / f"{name}.json").write_text(json.dumps(case))
    (directory / "run.json").write_text(
        json.dumps(
            {
                "completed_cases": names,
                "source": {"sha256": "source"},
            }
        )
    )
    return directory


def test_collect_accepts_complete_paper_runtime(runtime_run, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    rows = collect([runtime_run])
    assert len(rows) == 8
    assert all(row["relative_throughput"] == 1 and row["peak_gib"] == 1 for row in rows)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("config", "hidden_size", 16),
        ("config", "num_hidden_layers", 2),
        ("config", "num_kv_channels", 1),
        ("config", "num_passes", 1),
        ("config", "cyclic_groups", 16),
        ("config", "prefill_mode", "autoregressive"),
        ("config", "execution_mode", "jacobi"),
        ("config", "residual_dtype", "fp32"),
        ("config", "document_separator_token_id", 151643),
        ("workload", "parameter_dtype", "float32"),
        ("workload", "parameter_dtype", None),
        ("workload", "attention_backend", "sdpa"),
    ],
)
def test_collect_rejects_incompatible_runtime(runtime_run, section, key, value):
    path = runtime_run / "cases" / "white_matter-decode.json"
    case = json.loads(path.read_text())
    target = case["result"]["config"] if section == "config" else case["workload"]
    if value is None:
        del target[key]
    else:
        target[key] = value
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="paper runtime"):
        collect([runtime_run])


def test_collect_rejects_missing_model_fields(runtime_run):
    path = runtime_run / "cases" / "vanilla-prefill.json"
    case = json.loads(path.read_text())
    del case["result"]["config"]["pad_token_id"]
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="pad_token_id"):
        collect([runtime_run])

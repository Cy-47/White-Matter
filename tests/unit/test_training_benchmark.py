"""Training workload options, packed inputs, and distributed capacity failures."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks import _runner, training
from benchmarks._measurement import source_files
from training.recipes import load_recipe


def test_training_source_is_in_benchmark_identity():
    assert "training/step.py" in source_files()


def test_packed_inputs_are_deterministic_and_rank_specific():
    recipe = load_recipe("recipes/paper/white_matter_1p3b.yaml")
    kwargs = {"device": "cpu", "rank": 0, "count": 2}
    a = training.synthetic_batches(recipe, 2, **kwargs)
    b = training.synthetic_batches(recipe, 2, **kwargs)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a[0], a[1])
    assert (a[0] == recipe.data.eos_token_id).sum() >= 6
    assert not torch.equal(a[0], training.synthetic_batches(recipe, 2, device="cpu", rank=1, count=2)[0])


@pytest.mark.parametrize("distributed", [False, True])
def test_training_uses_shared_capacity_search(tmp_path, monkeypatch, distributed):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training",
            "--recipes",
            "recipes/paper/white_matter_1p3b.yaml",
            "--output",
            str(tmp_path),
            "--batch-sizes",
            "1",
            "--memory-budget-gib",
            "10",
            "--max-batch-size",
            "8",
            "--world-size",
            "2" if distributed else "1",
            "--no-compiled",
        ],
    )

    def worker(command, **kwargs):
        path = Path(command[-1])
        case = json.loads(path.read_text())
        assert case["workload"]["compiled"] is False
        batch = case["workload"]["batch_size"]
        if distributed and batch > 3:
            rank = path.parent / "ranks" / f"{path.stem}.rank1.json"
            rank.parent.mkdir(exist_ok=True)
            rank.write_text(json.dumps({"status": "cuda_oom"}))
            return SimpleNamespace(returncode=1)
        case.update(
            status="ok",
            result={
                "training": {"tokens_per_second": batch * 100, "median_seconds": 1},
                "memory": {"budget_accounted_bytes": (6 + batch) * 2**30},
            },
        )
        path.write_text(json.dumps(case))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(_runner.subprocess, "run", worker)
    training.main()
    run = next(tmp_path.glob("*-training"))
    report = json.loads((run / "summary.json").read_text())
    assert report["complete"]
    assert report["capacity"][0]["maximum_feasible_batch"] == 3
    assert report["capacity"][0]["boundary_checked"]
    assert "training" in (run / "RESULTS.md").read_text()
    assert list((run / "recipes").glob("*.yaml"))


def test_failed_worker_exits_without_waiting_for_collective_shutdown(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["training", "--worker", "case.json"])
    shutdown = []

    def fail(*args):
        raise SystemExit(1)

    monkeypatch.setattr(training, "run_worker", fail)
    monkeypatch.setattr(training.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(training.dist, "destroy_process_group", lambda: shutdown.append(True))
    with pytest.raises(SystemExit):
        training.main()
    assert not shutdown, "a failed rank must exit so the launcher can terminate its peers"

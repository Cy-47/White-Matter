"""Measurement failures and partial sweeps cannot become valid benchmark evidence."""

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from benchmarks import _measurement as measurement
from benchmarks.generation import collect_trials
from benchmarks.report import summarize_run


def test_monitor_propagates_background_errors(monkeypatch):
    def read(_):
        if threading.current_thread() is not threading.main_thread():
            raise OSError("sampling unavailable")
        return 100, 80

    monkeypatch.setattr(measurement, "_device_memory", read)
    monitor = measurement.MemoryMonitor()

    def sample_until_failure():
        with monitor:
            monitor._thread.join(timeout=5)
            assert not monitor._thread.is_alive()

    with pytest.raises(RuntimeError, match="sampling failed"):
        sample_until_failure()
    assert isinstance(monitor._error, OSError)


def test_monitor_joins_pending_sample_before_finalizing(monkeypatch):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    monitor = measurement.MemoryMonitor()

    def read(_):
        if threading.current_thread() is monitor._thread:
            started.set()
            assert release.wait(5)
            # Allocator state may change between the two readings.
            return 200, 100
        return 100, 90

    monkeypatch.setattr(measurement, "_device_memory", read)
    monitor.__enter__()
    assert started.wait(5)

    def close():
        monitor.__exit__(None, None, None)
        finished.set()

    closer = threading.Thread(target=close)
    closer.start()
    assert monitor._stop.wait(5)
    assert not finished.is_set()
    release.set()
    closer.join(timeout=5)
    assert finished.is_set()
    assert not monitor._thread.is_alive()
    assert monitor.peaks == {"sampled_device_peak_bytes": 200, "sampled_non_torch_peak_bytes": 10}


def test_monitor_excludes_setup_and_preserves_trial_error(monkeypatch):
    used = 1000
    monkeypatch.setattr(measurement, "_device_memory", lambda _: (used, 0))
    monitor = measurement.MemoryMonitor()
    used = 10  # Warmup is over before the measurement context starts.
    with pytest.raises(ValueError, match="trial failed"), monitor:
        raise ValueError("trial failed")
    assert not monitor._thread.is_alive()
    assert monitor.peaks["sampled_device_peak_bytes"] == 10


def test_exact_tokens_reject_equal_sum_sequences():
    outputs = iter([bytes([1, 2]), bytes([2, 1])])
    with pytest.raises(RuntimeError, match="changed generation outputs"):
        collect_trials(lambda: ({"seconds": 1}, next(outputs)), 2)


def test_source_snapshot_includes_untracked_code_and_detects_changes(tmp_path, monkeypatch):
    code = tmp_path / "new.py"
    code.write_text("value = 1\n")
    monkeypatch.setattr(measurement, "source_files", lambda: {"src/new.py": code})
    original = measurement.snapshot_source(tmp_path / "sources")
    assert (Path(original["archive"]) / "src/new.py").read_text() == code.read_text()
    measurement.verify_sources(original)
    code.write_text("value = 2\n")
    with pytest.raises(RuntimeError, match="source changed"):
        measurement.verify_sources(original)
    assert measurement.snapshot_source(tmp_path / "sources")["sha256"] != original["sha256"]


def test_report_requires_finished_sweep_and_flags_nonmonotonicity(tmp_path):
    measurement.write_json(
        tmp_path / "run.json",
        {
            "run_id": "test",
            "complete": False,
            "protocol": {"phase": "decode", "memory_budget_gib": [10], "memory_headroom_gib": 0, "max_batch_size": 8},
        },
    )
    for batch, used in [(1, 5), (2, 11), (3, 9), (4, 12)]:
        measurement.write_json(
            tmp_path / "cases" / f"{batch}.json",
            {
                "case_id": str(batch),
                "status": "ok",
                "workload": {"model": "wm", "repeat": 0, "batch_size": batch, "prompt_length": 2048},
                "result": {
                    "memory": {"budget_accounted_bytes": used * 2**30},
                    "decode": {"median_seconds": 1, "tokens_per_second": batch, "samples_seconds": [1]},
                },
            },
        )
    summary = summarize_run(tmp_path)
    assert not summary["complete"]
    capacity = summary["capacity"][0]
    assert capacity["maximum_feasible_batch"] == 3
    assert capacity["nonmonotonic"]
    assert capacity["boundary_checked"]
    assert "samples_seconds" not in summary["measurements"][0]["decode"]
    measurement.finish_run(tmp_path, ["1", "2", "3", "4"])
    assert summarize_run(tmp_path)["complete"]
    (tmp_path / "cases" / "4.json").unlink()
    assert not summarize_run(tmp_path)["complete"]


def test_profile_is_complete_but_not_a_throughput_result(tmp_path):
    measurement.write_json(tmp_path / "run.json", {"run_id": "profile", "completed_cases": ["case"], "protocol": {}})
    measurement.write_json(
        tmp_path / "cases" / "case.json",
        {"case_id": "case", "status": "profiled", "workload": {}, "result": {"trace": "trace.json"}},
    )
    summary = summarize_run(tmp_path)
    assert summary["complete"]
    assert not summary["capacity"]
    assert summary["measurements"][0]["decode"] is None


def test_reporting_does_not_import_torch():
    subprocess.run(
        [sys.executable, "-c", "import sys; import benchmarks.report; assert 'torch' not in sys.modules"], check=True
    )


def test_finished_profile_persists_source_identity(tmp_path, monkeypatch):
    from benchmarks import _runner, generation

    measurement.write_json(tmp_path / "run.json", {"source": {}, "checkpoints": {"wm": {}}})
    path = tmp_path / "cases" / "case.json"
    measurement.write_json(
        path, {"case_id": "case", "status": "pending", "workload": {"model": "wm", "profile": "trace.json"}}
    )
    verifies = []
    monkeypatch.setattr(_runner, "verify_sources", lambda _: verifies.append(True))
    monkeypatch.setattr(_runner, "input_files", lambda _: {})
    monkeypatch.setattr(generation, "benchmark", lambda _: {"trace": "trace.json"})
    _runner.run_worker(path, generation.benchmark)
    assert len(verifies) == 2
    assert json.loads(path.read_text())["status"] == "profiled"


def test_csv_preserves_run_and_workload_identity(tmp_path, monkeypatch):
    import csv

    from benchmarks import report

    measurement.write_json(
        tmp_path / "run.json", {"run_id": "distinct-run", "completed_cases": ["case"], "protocol": {}}
    )
    measurement.write_json(
        tmp_path / "cases" / "case.json",
        {
            "case_id": "case",
            "status": "ok",
            "workload": {"model": "wm", "repeat": 0, "phase": "end-to-end", "batch_size": 3, "prompt_length": 128},
            "result": {
                "memory": {"budget_accounted_bytes": 123},
                "decode": {"median_seconds": 1, "tokens_per_second": 3},
            },
        },
    )
    path = tmp_path / "table.csv"
    monkeypatch.setattr(sys, "argv", ["report", str(tmp_path), "--csv", str(path)])
    report.main()
    with path.open() as stream:
        (row,) = csv.DictReader(stream)
    assert (row["run_id"], row["workload_phase"], row["phase"]) == ("distinct-run", "end-to-end", "decode")
    assert row["budget_accounted_bytes"] == "123"


def test_concurrent_snapshots_never_publish_partial_files(tmp_path, monkeypatch):
    code = tmp_path / "model.py"
    code.write_text("complete source")
    monkeypatch.setattr(measurement, "source_files", lambda: {"model.py": code})
    copy = measurement.shutil.copyfile
    nested = []
    published = []

    def interrupted_copy(source, target):
        if not nested:
            Path(target).write_text("partial")
            nested.append(None)
            nested[0] = measurement.snapshot_source(tmp_path / "archive")
            published.append((Path(nested[0]["archive"]) / "model.py").stat().st_ino)
        return copy(source, target)

    monkeypatch.setattr(measurement.shutil, "copyfile", interrupted_copy)
    actual = measurement.snapshot_source(tmp_path / "archive")
    assert nested[0] == actual
    # Replacing another writer's published inode can invalidate NFS readers.
    assert (Path(actual["archive"]) / "model.py").stat().st_ino == published[0]

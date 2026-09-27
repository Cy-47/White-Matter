"""Shared benchmark records, source identity and bounded GPU memory sampling."""

import hashlib
import importlib.metadata
import json
import shutil
import socket
import statistics
import tempfile
import threading
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC
from functools import cache
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_files() -> dict[str, Path]:
    import white_matter

    root = Path(__file__).resolve().parents[1]
    package = Path(white_matter.__file__).resolve().parent
    return {
        **{f"src/white_matter/{p.relative_to(package)}": p for p in sorted(package.rglob("*.py"))},
        **{f"benchmarks/{p.name}": p for p in sorted((root / "benchmarks").glob("*.py"))},
        **{
            str(p.relative_to(root)): p
            for directory in ("training", "evals", "studies")
            for p in sorted((root / directory).rglob("*.py"))
        },
        "pyproject.toml": root / "pyproject.toml",
    }


def snapshot_source(directory: Path) -> dict:
    files = source_files()
    hashes = {name: digest(path) for name, path in files.items()}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    archive = directory / identity
    for name, path in files.items():
        target = archive / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=target.parent) as staging:
                complete = Path(staging) / target.name
                shutil.copyfile(path, complete)
                # Publish once: replacing another writer's inode can invalidate
                # concurrent readers on shared filesystems.
                with suppress(FileExistsError):
                    target.hardlink_to(complete)
        if digest(target) != hashes[name]:
            raise RuntimeError(f"source changed while taking snapshot: {name}")
    return {"sha256": identity, "files": hashes, "archive": str(archive.resolve())}


def verify_sources(expected: dict) -> None:
    actual = {name: digest(path) for name, path in source_files().items()}
    if actual != expected["files"]:
        raise RuntimeError("benchmark/library source changed since run creation")


def checkpoint_files(model: str) -> dict[str, str]:
    path = Path(model)
    files = [path / "config.json", *sorted(path.glob("*.safetensors"))]
    index = path / "model.safetensors.index.json"
    if index.exists():
        files.append(index)
    if not files[0].is_file() or not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"{model} must be a local HF export with safetensors")
    return {p.name: digest(p) for p in files}


def summarize(samples: list[float], tokens: int = 0) -> dict:
    median = statistics.median(samples)
    quartiles = statistics.quantiles(samples, n=4, method="inclusive") if len(samples) > 1 else [median] * 3
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "iqr_seconds": quartiles[2] - quartiles[0],
        "tokens_per_second": tokens / median if tokens else None,
    }


def environment(device: int | None = None) -> dict:
    import torch

    result = {"hostname": socket.gethostname(), "torch": torch.__version__, "cuda": torch.version.cuda, "packages": {}}
    for name in ("transformers", "flash-attn", "tilelang", "triton", "cut-cross-entropy", "nvidia-ml-py"):
        with suppress(importlib.metadata.PackageNotFoundError):
            result["packages"][name] = importlib.metadata.version(name)
    if device is not None:
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu=properties.name, gpu_uuid=str(properties.uuid), total_memory_bytes=properties.total_memory)
    return result


@cache
def _device_total_memory(device: int) -> int:
    import pynvml
    import torch

    # NVML's fixed capacity includes device reservations excluded by CUDA's total.
    return pynvml.nvmlDeviceGetMemoryInfo(torch.cuda._get_pynvml_handler(device)).total


def _device_memory(device: int) -> tuple[int, int]:
    import torch

    free, _ = torch.cuda.mem_get_info(device)
    return _device_total_memory(device) - free, torch.cuda.memory_reserved(device)


class MemoryMonitor:
    """Sample only one measurement window; join before exposing its final peaks.

    Callers synchronize before entering and leaving. Only those stable boundaries
    can separate non-Torch memory from allocator memory: concurrent readings are
    not atomic. During execution, sample total device usage without subtraction.
    CUDA memory queries avoid NVML's delayed readings. Sampling errors propagate.
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.peaks = {"sampled_device_peak_bytes": 0, "sampled_non_torch_peak_bytes": 0}
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self, *, baseline: bool = False) -> None:
        used, reserved = _device_memory(self.device)
        self.peaks["sampled_device_peak_bytes"] = max(self.peaks["sampled_device_peak_bytes"], used)
        if baseline:
            self.peaks["sampled_non_torch_peak_bytes"] = max(
                self.peaks["sampled_non_torch_peak_bytes"], used - reserved
            )

    def _run(self) -> None:
        try:
            while not self._stop.wait(0.01):
                self._sample()
        except Exception as error:
            self._error = error

    def __enter__(self):
        self._sample(baseline=True)  # Fail synchronously before entering the timed region.
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join()
        if exc is None:
            if self._error is not None:
                raise RuntimeError("GPU memory sampling failed") from self._error
            self._sample(baseline=True)  # No background writer remains at this boundary.


def create_run(output: Path, kind: str, protocol: dict, checkpoints: dict | None = None) -> Path:
    from datetime import datetime

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + kind
    directory = output.resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    source = snapshot_source(output.resolve() / "sources")
    write_json(
        directory / "run.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "kind": kind,
            "completed_cases": None,
            "protocol": protocol,
            "source": source,
            "checkpoints": checkpoints or {},
        },
    )
    return directory


def finish_run(directory: Path, case_ids: list[str]) -> None:
    path = directory / "run.json"
    run = json.loads(path.read_text())
    run["completed_cases"] = sorted(case_ids)
    write_json(path, run)


def fits_memory(case: dict, limit: float) -> bool:
    return case["status"] == "ok" and case["result"]["memory"]["budget_accounted_bytes"] <= limit


def record_case(path: Path, case: dict, measure: Callable[[], dict], verify: Callable[[], None]) -> None:
    """Persist outcomes; verify identity after OOM as well as successful measurements."""
    import traceback

    import torch
    from torch._dynamo.exc import BackendCompilerFailed

    try:
        verify()
        try:
            result = measure()
            case.update(status="profiled" if case["workload"].get("profile") else "ok", result=result)
        except (torch.OutOfMemoryError, BackendCompilerFailed) as error:
            if isinstance(error, BackendCompilerFailed) and not isinstance(
                error.inner_exception, torch.OutOfMemoryError
            ):
                raise
            case.update(status="cuda_oom", error=str(error))
        verify()
    except Exception as error:
        case.update(status="failed", error=str(error))
        traceback.print_exc()
    write_json(path, case)

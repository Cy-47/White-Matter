"""Rank-zero JSONL metrics, rolling averages, and resume truncation."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _read_metric_step(line: str) -> int:
    """metrics.jsonl line → its step (huge sentinel if unparseable, so
    malformed / post-checkpoint rows are dropped on resume-truncate)."""
    try:
        return int(json.loads(line).get("step", 1 << 60))
    except Exception:
        return 1 << 60


class MetricsLogger:
    """Write step metrics and checkpointable rolling means; other ranks are no-ops."""

    def __init__(
        self,
        metrics_path: Path,
        *,
        enabled: bool,
        rolling_keys: tuple[str, ...] = ("loss",),
        window: int = 200,
        constant_fields: Mapping[str, Any] | None = None,
    ):
        self.path = Path(metrics_path)
        self.enabled = enabled
        self.constant_fields = dict(constant_fields or {})
        self.rolling: dict[str, deque[float]] = {k: deque(maxlen=window) for k in rolling_keys}
        self._file = self.path.open("a", buffering=1) if enabled else None

    def _add_constant_fields(self, row: dict[str, Any]) -> None:
        conflicts = {key for key, value in self.constant_fields.items() if key in row and row[key] != value}
        if conflicts:
            raise ValueError("metric row contradicts constant field(s): " + ", ".join(sorted(conflicts)))
        row.update(self.constant_fields)

    def restore_rolling(self, state: dict[str, list[float]]) -> None:
        for k, vals in state.items():
            if k in self.rolling:
                self.rolling[k].extend(vals)

    def rolling_state(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self.rolling.items()}

    def truncate_to(self, start_step: int) -> None:
        """Drop rows with step >= start_step (written after the checkpoint we
        are resuming from), then re-open for appending."""
        if not self.enabled:
            return
        if self._file is not None:
            self._file.close()
        if self.path.exists():
            kept = [
                line
                for line in self.path.read_text().splitlines()
                if line.strip() and _read_metric_step(line) < start_step
            ]
            self.path.write_text("\n".join(kept) + ("\n" if kept else ""))
        self._file = self.path.open("a", buffering=1)

    def log_step(self, row: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return row
        self._add_constant_fields(row)
        for k in self.rolling:
            if k in row:
                self.rolling[k].append(float(row[k]))
        for k, dq in self.rolling.items():
            if dq:
                row[f"{k}_rolling"] = sum(dq) / len(dq)
        self._file.write(json.dumps(row) + "\n")
        return row

    def log_raw(self, row: dict[str, Any]) -> None:
        """Write a row verbatim (skipped-step markers, eval rows)."""
        if self.enabled and self._file is not None:
            self._add_constant_fields(row)
            self._file.write(json.dumps(row) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

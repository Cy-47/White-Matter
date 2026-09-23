"""Vectorized EOS packing with a resumable partial-row buffer.

Each document appends one EOS. Complete rows are emitted in bulk; buf retains
only the trailing tokens for the next call.
"""

from __future__ import annotations
from typing import Iterable
import numpy as np


class Packer:
    def __init__(self, seq_len: int, eos_id: int, buf: list[int] | None = None):
        self.seq_len = int(seq_len)
        self.eos_id = int(eos_id)
        self.buf: list[int] = list(buf) if buf else []  # carry-over tokens (< seq_len)
        self._rows: list[np.ndarray] = []  # pending (k, seq_len) int32 blocks

    def add_doc(self, ids: Iterable[int]) -> None:
        # NumPy's bulk conversion avoids constructing Python ints one at a time.
        buf = self.buf
        buf.extend(ids.tolist() if hasattr(ids, "tolist") else ids)
        buf.append(self.eos_id)
        n = self.seq_len
        if len(buf) >= n:
            k = len(buf) // n
            take = k * n
            self._rows.append(np.asarray(buf[:take], dtype=np.int32).reshape(k, n))
            self.buf = buf[take:]  # remainder < seq_len (single small slice)

    def take_rows(self) -> np.ndarray:
        """Return (and clear) the rows emitted since the last call."""
        if not self._rows:
            return np.empty((0, self.seq_len), dtype=np.int32)
        arr = self._rows[0] if len(self._rows) == 1 else np.concatenate(self._rows, axis=0)
        self._rows = []
        return arr

"""Memory-mapped token caches and deterministic loader resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader


def load_cache_metadata(cache_dir: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((Path(cache_dir) / "cache_meta.json").read_text()))


def split_row_indices(
    cache_dir: Path,
    split: str,
    *,
    n_train: int,
    n_val: int,
    n_test: int,
) -> range:
    """Indices into the pre-shuffled cache: contiguous train, val, then test."""
    n_total = int(load_cache_metadata(cache_dir)["n_total"])
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    if min(n_train, n_val, n_test) < 0:
        raise ValueError(f"Split sizes must be non-negative, got train={n_train}, val={n_val}, test={n_test}")
    if n_train + n_val + n_test > n_total:
        raise ValueError("Requested split sizes exceed available sequences")
    bounds = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_train + n_val + n_test),
    }
    if split not in bounds:
        raise ValueError(f"unknown split {split!r}; expected 'train' | 'val' | 'test'")
    return range(*bounds[split])


class TokenCacheDataset(torch.utils.data.Dataset):
    """Read one contiguous split of the pre-shuffled tokenized.npy cache."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        split: str,
        sequence_length: int,
        n_train: int,
        n_val: int,
        n_test: int,
    ):
        cache_dir = Path(cache_dir)
        npy_path = cache_dir / "tokenized.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"{npy_path} not found; this dataset requires the tokenized.npy memmap layout.")
        arr = np.load(npy_path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] < sequence_length:
            raise ValueError(f"{npy_path} has shape {arr.shape}; need (N, ≥{sequence_length}).")
        self.sequence_length = int(sequence_length)
        self.indices = split_row_indices(
            cache_dir,
            split,
            n_train=int(n_train),
            n_val=int(n_val),
            n_test=int(n_test),
        )
        # Keep the cache memory-mapped. Materializing the full 20B-token train
        # split once per rank can consume hundreds of GiB without changing a
        # single training example; the OS page cache already shares these rows.
        self._mmap = arr

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        row = self._mmap[self.indices[idx], : self.sequence_length]
        # Own the int64 buffer before exposing the read-only memmap to PyTorch.
        return {"input_ids": torch.from_numpy(np.array(row, dtype=np.int64, copy=True))}


def _new_loader_iterator(loader: DataLoader) -> object:
    """Create a loader iterator without perturbing the model's CPU RNG."""
    with torch.random.fork_rng(devices=[]):
        return iter(loader)


def _resume_sequential_loader(
    loader: DataLoader,
    *,
    total_batches: int,
    sampler: object | None = None,
) -> tuple[object, int, int, int]:
    """Create a sequential loader iterator at an exact consumed-batch offset."""
    if total_batches < 0:
        raise ValueError(f"total_batches must be non-negative, got {total_batches}")
    batches_per_epoch = len(loader)
    if batches_per_epoch <= 0:
        raise ValueError("cannot resume an empty loader")

    epoch, skip_batches = divmod(total_batches, batches_per_epoch)
    if sampler is not None:
        sampler.set_epoch(epoch)
    iterator = _new_loader_iterator(loader)
    for _ in range(skip_batches):
        next(iterator)
    return iterator, epoch, skip_batches, batches_per_epoch

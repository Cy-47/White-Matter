"""Build the paper's EOS-packed FineWeb-Edu cache.

Workers stream disjoint parquet shards into one shared tokenized.npy memmap.
Rows follow worker order, then shard order. Train/val/test are contiguous slices
of this pre-shuffled source. cache_meta.json records the source, packing, and splits.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty

import numpy as np
from _packing import Packer

DEFAULT_DATASET = "karpathy/fineweb-edu-100b-shuffle"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
N_TRAIN, N_VAL, N_TEST = 9_765_625, 2_000, 5_000
SEQUENCE_LENGTH = 2_048
TEXT_BATCH_SIZE, MIN_TEXT_CHARS = 8_000, 100


def worker(
    *,
    worker_id: int,
    num_workers: int,
    n_target: int,
    row_offset: int,
    tok_path: str,
    q: Queue,
):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    packer = Packer(SEQUENCE_LENGTH, tok.eos_token_id)

    ds = load_dataset(DEFAULT_DATASET, split=DEFAULT_SPLIT, streaming=True)
    # Shard by parquet file: worker i gets the i-th contiguous block of shards,
    # so downloads run concurrently and rows stay disjoint across workers.
    ds = ds.shard(num_shards=num_workers, index=worker_id, contiguous=True)

    # Open the shared output memmap and write only this worker's disjoint slice
    # [row_offset, row_offset + n_target). r+ = existing file, no realloc.
    out = np.lib.format.open_memmap(tok_path, mode="r+")
    assert out.shape == (N_TRAIN + N_VAL + N_TEST, SEQUENCE_LENGTH)

    written = 0  # rows written into the memmap slice
    docs_seen = 0
    buf: list[str] = []
    t0 = time.time()
    last_log = t0

    def tokenize_batch(texts: list[str]) -> bool:
        # Returns True once the worker has produced enough rows.
        nonlocal written, docs_seen, last_log
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        for ids in enc:
            docs_seen += 1
            packer.add_doc(ids)
        # Each tokenizer batch already yields a contiguous block of packed rows.
        rows = packer.take_rows()[: n_target - written]
        out[row_offset + written : row_offset + written + len(rows)] = rows
        written += len(rows)
        now = time.time()
        if now - last_log >= 30.0:
            q.put((worker_id, written, now - t0))
            last_log = now
        return written >= n_target

    for sample in ds:
        txt = sample.get(DEFAULT_TEXT_FIELD)
        if not txt:
            continue
        txt = txt.strip()
        if len(txt) < MIN_TEXT_CHARS:
            continue
        buf.append(txt)
        if len(buf) >= TEXT_BATCH_SIZE:
            done = tokenize_batch(buf)
            buf = []
            if done:
                break
    if buf and written < n_target:
        tokenize_batch(buf)

    out.flush()

    if written < n_target:
        raise RuntimeError(
            f"worker {worker_id}: shard exhausted before target "
            f"(written={written}, target={n_target}, docs_seen={docs_seen})"
        )
    q.put((worker_id, written, time.time() - t0))


def main():
    ap = argparse.ArgumentParser(description="Build a 20B-token FineWeb-Edu cache with the paper packing protocol.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    # Resolve the delimiter from the selected tokenizer rather than assuming
    # Qwen's EOS id. Workers load this same tokenizer and use its EOS token for
    # packing, so persisting the resolved value keeps downstream document masks
    # aligned.
    from transformers import AutoTokenizer

    metadata_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if metadata_tokenizer.eos_token_id is None:
        raise ValueError(f"tokenizer {DEFAULT_MODEL!r} has no eos_token_id; EOS packing requires one")
    eos_id = int(metadata_tokenizer.eos_token_id)

    n_total = N_TRAIN + N_VAL + N_TEST
    if args.workers < 1:
        ap.error("--workers must be positive")
    W = args.workers
    args.output.mkdir(parents=True, exist_ok=True)
    tok_path = args.output / "tokenized.npy"
    if tok_path.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {tok_path}")

    # Per-worker row targets sum exactly to n_total; offsets are the cumsum so
    # each worker owns a disjoint contiguous slice of the output.
    per_worker = [n_total // W] * W
    for i in range(n_total - sum(per_worker)):
        per_worker[i] += 1
    offsets = [0]
    for c in per_worker[:-1]:
        offsets.append(offsets[-1] + c)
    print(
        f"workers={W} n_total={n_total} (~{N_TRAIN * SEQUENCE_LENGTH / 1e9:.2f}B train tok) "
        f"per_worker={per_worker[:4]}{'...' if W > 4 else ''}",
        flush=True,
    )

    # Pre-allocate the full output memmap ONCE; workers write disjoint slices.
    est_gb = n_total * SEQUENCE_LENGTH * 4 / 1e9
    print(f"allocating {tok_path} shape=({n_total},{SEQUENCE_LENGTH}) int32 (~{est_gb:.1f} GB)...", flush=True)
    mm = np.lib.format.open_memmap(tok_path, mode="w+", dtype=np.int32, shape=(n_total, SEQUENCE_LENGTH))
    del mm  # workers reopen in r+ mode

    q: Queue = Queue()
    procs = []
    t0 = time.time()
    for i in range(W):
        p = Process(
            target=worker,
            kwargs={
                "worker_id": i,
                "num_workers": W,
                "n_target": per_worker[i],
                "row_offset": offsets[i],
                "tok_path": str(tok_path),
                "q": q,
            },
            daemon=False,
        )
        p.start()
        procs.append(p)

    progress = dict.fromkeys(range(W), 0)
    while any(progress[i] < target for i, target in enumerate(per_worker)):
        try:
            wid, n_done, elapsed = q.get(timeout=30)
        except Empty:
            failed = [p for p in procs if p.exitcode not in (None, 0)]
            if failed:
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                for p in procs:
                    p.join()
                details = ", ".join(f"pid={p.pid} exitcode={p.exitcode}" for p in failed)
                raise RuntimeError(f"cache-build worker failed: {details}") from None
            if all(p.exitcode is not None for p in procs):
                raise RuntimeError(
                    f"all cache-build workers exited before completion messages were received; progress={progress}"
                ) from None
            agg = sum(progress.values())
            rate = agg / max(time.time() - t0, 1e-9)
            eta = (n_total - agg) / max(rate, 1e-9)
            print(
                f"[heartbeat] {(time.time() - t0) / 60:.1f}m agg={agg}/{n_total} "
                f"({rate:.0f} row/s, eta {eta / 60:.0f}m) per_worker={progress}",
                flush=True,
            )
            continue
        progress[wid] = n_done
        agg = sum(progress.values())
        rate = agg / max(time.time() - t0, 1e-9)
        eta = (n_total - agg) / max(rate, 1e-9)
        print(
            f"[w{wid}] {n_done}/{per_worker[wid]}  agg={agg}/{n_total}  {rate:.0f} row/s  eta {eta / 60:.0f}m",
            flush=True,
        )
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker pid={p.pid} exited with code {p.exitcode}")

    metadata = {
        "n_total": int(n_total),
        "max_length": SEQUENCE_LENGTH,
        "model": DEFAULT_MODEL,
        "dataset": DEFAULT_DATASET,
        "dataset_config": "",
        "dataset_split": DEFAULT_SPLIT,
        "text_field": DEFAULT_TEXT_FIELD,
        "packing": "eos_crossdoc",
        "eos_id": eos_id,
        "approx_train_tokens": N_TRAIN * SEQUENCE_LENGTH,
        "splits": {"n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST},
        "build": {
            "tool": "prepare_fineweb_edu.py",
            "num_workers": W,
            "per_worker_target": per_worker,
            "elapsed_minutes": round((time.time() - t0) / 60.0, 2),
        },
    }
    (args.output / "cache_meta.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"DONE {tok_path} shape=({n_total},{SEQUENCE_LENGTH}) in "
        f"{(time.time() - t0) / 60:.1f}m  (~{N_TRAIN * SEQUENCE_LENGTH / 1e9:.2f}B train tokens)",
        flush=True,
    )


if __name__ == "__main__":
    main()

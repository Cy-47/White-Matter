"""Build an EOS-packed FineWeb-Edu cache.

The default packs the source continuously in dataset order.
``--reproduce-paper-order`` uses the paper's eight independent partitions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

try:
    from ._packing import Packer
except ImportError:  # Direct script invocation.
    from _packing import Packer

DEFAULT_DATASET = "karpathy/fineweb-edu-100b-shuffle"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "cache_fineweb_edu_20b_len2048"
N_TRAIN, N_VAL, N_TEST = 9_765_625, 2_000, 5_000
SEQUENCE_LENGTH = 2_048
TEXT_BATCH_SIZE, MIN_TEXT_CHARS = 8_000, 100
NUM_PARTITIONS = 8
PARTITION_PROTOCOL = "fixed-eight-v1"
SOURCE_PROTOCOL = "continuous-source-v1"


def resolve_tokenizer_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    try:
        import gigatoken  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "gigatoken":
            raise
        warnings.warn(
            "Gigatoken is not installed; falling back to slower Hugging Face tokenization. "
            "Install with: pip install 'gigatoken>=0.10,<1'. "
            "Select --tokenizer-backend hf to use Hugging Face explicitly.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "hf"
    return "gigatoken"


def load_tokenizer(backend: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"tokenizer {DEFAULT_MODEL!r} has no eos_token_id; EOS packing requires one")
    if backend == "gigatoken":
        try:
            import gigatoken
        except ImportError as exc:
            raise ImportError("Gigatoken backend requires: pip install 'gigatoken>=0.10,<1'") from exc
        tokenizer = gigatoken.Tokenizer(tokenizer).as_hf()
    return tokenizer


def tokenizer_metadata(backend: str) -> dict:
    packages = ["transformers", "tokenizers"] + (["gigatoken"] if backend == "gigatoken" else [])
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {"backend": backend, "versions": versions}


def write_rows(dataset, tokenizer, out, *, n_target: int, row_offset: int = 0) -> tuple[int, int]:
    packer = Packer(SEQUENCE_LENGTH, tokenizer.eos_token_id)
    written = 0
    docs_seen = 0
    texts = []

    def flush() -> None:
        nonlocal written, docs_seen, texts
        for token_ids in tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]:
            docs_seen += 1
            packer.add_doc(token_ids)
        rows = packer.take_rows()[: n_target - written]
        out[row_offset + written : row_offset + written + len(rows)] = rows
        written += len(rows)
        texts = []

    for sample in dataset:
        text = sample.get(DEFAULT_TEXT_FIELD)
        if not text:
            continue
        text = text.strip()
        if len(text) < MIN_TEXT_CHARS:
            continue
        texts.append(text)
        if len(texts) < TEXT_BATCH_SIZE:
            continue
        flush()
        if written >= n_target:
            break
    if texts and written < n_target:
        flush()
    return written, docs_seen


def write_source_order_cache(tok_path: Path, tokenizer, *, n_target: int) -> None:
    from datasets import load_dataset

    out = np.lib.format.open_memmap(tok_path, mode="r+")
    written, _ = write_rows(
        load_dataset(DEFAULT_DATASET, split=DEFAULT_SPLIT, streaming=True),
        tokenizer,
        out,
        n_target=n_target,
    )
    out.flush()
    if written < n_target:
        raise RuntimeError(f"dataset exhausted before target (written={written}, target={n_target})")


def write_partition(
    job: tuple[dict, str],
) -> tuple[int, int, float]:
    task, tok_path = job
    partition_id = task["partition_id"]
    num_partitions = task["num_partitions"]
    n_target = task["n_target"]
    row_offset = task["row_offset"]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from datasets import load_dataset

    tok = load_tokenizer(task.get("tokenizer_backend", "hf"))
    ds = load_dataset(DEFAULT_DATASET, split=DEFAULT_SPLIT, streaming=True)
    # Logical partitions are independent of process count. Each gets the same
    # source shards, packing state, row quota, and output offset on every build.
    ds = ds.shard(num_shards=num_partitions, index=partition_id, contiguous=True)

    # Open the shared output memmap and write only this partition's disjoint slice
    # [row_offset, row_offset + n_target). r+ = existing file, no realloc.
    out = np.lib.format.open_memmap(tok_path, mode="r+")
    assert out.shape == (N_TRAIN + N_VAL + N_TEST, SEQUENCE_LENGTH)

    t0 = time.time()

    written, docs_seen = write_rows(
        ds,
        tok,
        out,
        n_target=n_target,
        row_offset=row_offset,
    )

    out.flush()

    if written < n_target:
        raise RuntimeError(
            f"partition {partition_id}: shard exhausted before target "
            f"(written={written}, target={n_target}, docs_seen={docs_seen})"
        )
    return partition_id, written, time.time() - t0


def partition_tasks(n_total: int, num_partitions: int = NUM_PARTITIONS) -> list[dict]:
    """Assign contiguous output slices and independent packing boundaries."""
    tasks = []
    offset = 0
    for partition_id in range(num_partitions):
        target = n_total // num_partitions + int(partition_id < n_total % num_partitions)
        tasks.append(
            {"partition_id": partition_id, "num_partitions": num_partitions, "n_target": target, "row_offset": offset}
        )
        offset += target
    return tasks


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Build a 20B-token FineWeb-Edu cache.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--workers", type=int, default=8, help="Processes for --reproduce-paper-order (maximum 8).")
    ap.add_argument(
        "--reproduce-paper-order",
        action="store_true",
        help="Use the paper's eight-partition data order.",
    )
    ap.add_argument(
        "--tokenizer-backend",
        choices=("auto", "hf", "gigatoken"),
        default="auto",
        help="auto uses Gigatoken when installed, otherwise Hugging Face",
    )
    args = ap.parse_args(argv)
    if args.workers < 1:
        ap.error("--workers must be positive")
    return args


def main():
    args = parse_args()
    args.tokenizer_backend = resolve_tokenizer_backend(args.tokenizer_backend)

    # Resolve the delimiter from the selected tokenizer rather than assuming
    # Qwen's EOS id. Workers load this same tokenizer and use its EOS token for
    # packing, so persisting the resolved value keeps downstream document masks
    # aligned.
    metadata_tokenizer = load_tokenizer(args.tokenizer_backend)
    eos_id = int(metadata_tokenizer.eos_token_id)

    n_total = N_TRAIN + N_VAL + N_TEST
    num_partitions = NUM_PARTITIONS
    worker_count = min(args.workers, num_partitions)
    args.output.mkdir(parents=True, exist_ok=True)
    tok_path = args.output / "tokenized.npy"
    if tok_path.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {tok_path}")

    tasks = partition_tasks(n_total, num_partitions) if args.reproduce_paper_order else []
    for task in tasks:
        task["tokenizer_backend"] = args.tokenizer_backend
    targets = [task["n_target"] for task in tasks]
    mode = "paper-partitions" if args.reproduce_paper_order else "source-order"
    print(f"mode={mode} tokenizer={args.tokenizer_backend} n_total={n_total}", flush=True)

    # Pre-allocate the output so the build never accumulates the full cache in memory.
    est_gb = n_total * SEQUENCE_LENGTH * 4 / 1e9
    print(f"allocating {tok_path} shape=({n_total},{SEQUENCE_LENGTH}) int32 (~{est_gb:.1f} GB)...", flush=True)
    mm = np.lib.format.open_memmap(tok_path, mode="w+", dtype=np.int32, shape=(n_total, SEQUENCE_LENGTH))
    del mm  # workers reopen in r+ mode

    t0 = time.time()
    if args.reproduce_paper_order:
        jobs = [(task, str(tok_path)) for task in tasks]
        pool = ProcessPoolExecutor(max_workers=worker_count)
        try:
            futures = [pool.submit(write_partition, job) for job in jobs]
            for future in as_completed(futures):
                partition_id, written, elapsed = future.result()
                print(f"partition {partition_id}: {written} rows in {elapsed / 60:.1f}m", flush=True)
        except BaseException:
            # Python 3.11–3.13 have no public executor API for stopping workers.
            # Kill before shutdown so failed or interrupted builds cannot wait
            # for other partitions to finish downloading or tokenizing.
            for process in tuple((pool._processes or {}).values()):
                if process.is_alive():
                    process.kill()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    else:
        write_source_order_cache(tok_path, metadata_tokenizer, n_target=n_total)

    metadata = {
        "n_total": int(n_total),
        "max_length": SEQUENCE_LENGTH,
        "model": DEFAULT_MODEL,
        "tokenizer": tokenizer_metadata(args.tokenizer_backend),
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
            **(
                {
                    "num_workers": worker_count,
                    "partition_protocol": PARTITION_PROTOCOL,
                    "num_partitions": num_partitions,
                    "per_partition_target": targets,
                }
                if args.reproduce_paper_order
                else {"ordering_protocol": SOURCE_PROTOCOL}
            ),
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

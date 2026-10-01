"""Cache bytes and split boundaries must not depend on process scheduling."""

import json
import multiprocessing
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import prepare_fineweb_edu as builder


@pytest.mark.parametrize("backend", ["hf", "gigatoken", "auto"])
@pytest.mark.parametrize("workers", [1, 2, 3, 4, 8, 12])
def test_paper_cache_contents_are_worker_invariant(tmp_path, monkeypatch, workers, backend):
    monkeypatch.setattr(builder, "N_TRAIN", 27)
    monkeypatch.setattr(builder, "N_VAL", 3)
    monkeypatch.setattr(builder, "N_TEST", 5)
    monkeypatch.setattr(builder, "SEQUENCE_LENGTH", 4)
    monkeypatch.setattr(builder, "TEXT_BATCH_SIZE", 2)
    monkeypatch.setattr(builder, "MIN_TEXT_CHARS", 0)

    class Source:
        def shard(self, *, num_shards, index, contiguous):
            assert num_shards == 8
            assert contiguous
            return [{"text": f"{index},{j}"} for j in range(20)]

    class Tokenizer:
        eos_token_id = 99

        def __call__(self, texts, **kwargs):
            # Vary document length so row packing crosses document boundaries.
            return {
                "input_ids": [
                    [int(t.split(",")[0]) * 100 + int(t.split(",")[1])] * (int(t.split(",")[1]) % 5 + 1) for t in texts
                ]
            }

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **kw: Source()))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())),
    )

    monkeypatch.setitem(
        sys.modules, "gigatoken", SimpleNamespace(Tokenizer=lambda hf: SimpleNamespace(as_hf=lambda: hf))
    )

    def build(name):
        path = tmp_path / name
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(35, 4))
        del arr
        tasks = builder.partition_tasks(35)
        # Reverse execution order to exercise scheduling-independent writes.
        for task in reversed(tasks):
            builder.write_partition((task, str(path)))
        return np.load(path)

    expected = build("original.npy")

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("test requires fork so child processes inherit fake dataset modules")
    context = multiprocessing.get_context("fork")
    monkeypatch.setattr(builder, "ProcessPoolExecutor", partial(ProcessPoolExecutor, mp_context=context))
    output = tmp_path / "candidate"
    argv = [
        "prepare_fineweb_edu",
        "--output",
        str(output),
        "--workers",
        str(workers),
        "--reproduce-paper-order",
        "--tokenizer-backend",
        backend,
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        argv,
    )
    builder.main()
    metadata = json.loads((output / "cache_meta.json").read_text())
    assert metadata["tokenizer"]["backend"] == ("gigatoken" if backend == "auto" else backend)
    assert metadata["build"]["num_workers"] == min(workers, 8)
    assert metadata["build"]["num_partitions"] == 8
    assert metadata["build"]["partition_protocol"] == "fixed-eight-v1"
    actual = np.load(output / "tokenized.npy")
    np.testing.assert_array_equal(actual, expected)
    for start, end in [(0, 27), (27, 30), (30, 35)]:
        np.testing.assert_array_equal(actual[start:end], expected[start:end])
    # Original partition quotas: first three receive five rows, the rest four.
    assert [t["n_target"] for t in builder.partition_tasks(35)] == [5, 5, 5, 4, 4, 4, 4, 4]
    assert [t["row_offset"] for t in builder.partition_tasks(35)] == [0, 5, 10, 15, 19, 23, 27, 31]


def _killed_partition(job):
    if job[0]["partition_id"] == 0:
        os.kill(os.getpid(), signal.SIGKILL)
    time.sleep(60)


def _failed_partition(job):
    output = Path(job[1]).parent
    if job[0]["partition_id"] == 0:
        # Ensure another worker is running when the ordinary exception occurs.
        deadline = time.monotonic() + 5
        while not list(output.glob("worker-*.pid")) and time.monotonic() < deadline:
            time.sleep(0.01)
        raise RuntimeError("simulated dataset read failure")
    (output / f"worker-{os.getpid()}.pid").write_text(str(os.getpid()))
    time.sleep(60)


@pytest.mark.parametrize(
    ("worker", "error", "message"),
    [
        (_killed_partition, BrokenProcessPool, "terminated abruptly"),
        (_failed_partition, RuntimeError, "simulated dataset read failure"),
    ],
    ids=["killed", "exception"],
)
def test_worker_failure_aborts_build(tmp_path, monkeypatch, worker, error, message):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("test requires fork so child processes inherit patched builder settings")
    context = multiprocessing.get_context("fork")
    monkeypatch.setattr(builder, "ProcessPoolExecutor", partial(ProcessPoolExecutor, mp_context=context))
    monkeypatch.setattr(builder, "write_partition", worker)
    monkeypatch.setattr(builder, "N_TRAIN", 8)
    monkeypatch.setattr(builder, "N_VAL", 0)
    monkeypatch.setattr(builder, "N_TEST", 0)
    monkeypatch.setattr(builder, "SEQUENCE_LENGTH", 4)
    monkeypatch.setattr(
        builder,
        "parse_args",
        lambda: SimpleNamespace(output=tmp_path, workers=2, reproduce_paper_order=True, tokenizer_backend="hf"),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(eos_token_id=99))
        ),
    )

    def run():
        os.setsid()
        with pytest.raises(error, match=message):
            builder.main()
        assert not multiprocessing.active_children()

    process = context.Process(target=run)
    process.start()
    process.join(timeout=10)
    if process.is_alive():
        # Include worker descendants when cleaning up a hung build.
        os.killpg(process.pid, signal.SIGKILL)
        process.join()
        pytest.fail("build hung after a worker failed")
    assert process.exitcode == 0
    assert not (tmp_path / "cache_meta.json").exists()
    if worker is _failed_partition:
        pid_files = list(tmp_path.glob("worker-*.pid"))
        assert pid_files, "no blocking worker started before the failure"
        for path in pid_files:
            with pytest.raises(ProcessLookupError):
                os.kill(int(path.read_text()), 0)


def test_defaults():
    args = builder.parse_args([])
    assert args.output == builder.DEFAULT_OUTPUT
    assert args.tokenizer_backend == "auto"
    assert args.workers == 8
    assert not args.reproduce_paper_order


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_default_preserves_continuous_source_order(tmp_path, monkeypatch, workers):
    monkeypatch.setitem(sys.modules, "gigatoken", None)
    monkeypatch.setattr(builder, "N_TRAIN", 27)
    monkeypatch.setattr(builder, "N_VAL", 3)
    monkeypatch.setattr(builder, "N_TEST", 5)
    monkeypatch.setattr(builder, "SEQUENCE_LENGTH", 4)
    monkeypatch.setattr(builder, "TEXT_BATCH_SIZE", 2)
    monkeypatch.setattr(builder, "MIN_TEXT_CHARS", 0)

    documents = [{"text": str(index)} for index in range(50)]

    class Tokenizer:
        eos_token_id = 99

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[int(text)] * (int(text) % 5 + 1) for text in texts]}

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **kw: iter(documents)))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())),
    )
    tokenizer = Tokenizer()
    packer = builder.Packer(4, tokenizer.eos_token_id)
    for document in documents:
        packer.add_doc(tokenizer([document["text"]])["input_ids"][0])
    expected = packer.take_rows()[:35]

    def build(name, count):
        output = tmp_path / name
        monkeypatch.setattr(
            sys,
            "argv",
            ["prepare_fineweb_edu", "--output", str(output), "--workers", str(count)],
        )
        builder.main()
        return np.load(output / "tokenized.npy"), json.loads((output / "cache_meta.json").read_text())

    actual, metadata = build(f"{workers}-workers", workers)
    np.testing.assert_array_equal(actual, expected)
    assert metadata["build"]["ordering_protocol"] == "continuous-source-v1"

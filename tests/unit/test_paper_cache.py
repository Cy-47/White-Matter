"""The public cache builder's metadata is accepted without archive-only files."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from studies.protocol import validate_paper_cache


def _metadata():
    total = 9_772_625
    return {
        "n_total": total, "max_length": 2048, "model": "Qwen/Qwen3-0.6B-Base",
        "dataset": "karpathy/fineweb-edu-100b-shuffle", "dataset_config": "",
        "dataset_split": "train", "text_field": "text", "packing": "eos_crossdoc",
        "eos_id": 151643, "approx_train_tokens": 9_765_625 * 2048,
        "splits": {"n_train": 9_765_625, "n_val": 2000, "n_test": 5000},
        "build": {
            "tool": "prepare_fineweb_edu.py", "num_workers": 8,
            "per_worker_target": [total // 8 + int(i < total % 8) for i in range(8)],
            "elapsed_minutes": 12.34,
        },
    }


def test_public_cache_metadata_and_layout_are_accepted(tmp_path, monkeypatch):
    metadata = _metadata()
    (tmp_path / "cache_meta.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: SimpleNamespace(
        shape=(9_772_625, 2048), dtype=np.dtype("int32"),
    ))
    assert validate_paper_cache(tmp_path) == metadata


@pytest.mark.parametrize("field,value", [("packing", "truncate"), ("eos_id", 0), ("dataset", "different")])
def test_public_cache_rejects_different_data_protocol(tmp_path, field, value):
    metadata = _metadata()
    metadata[field] = value
    (tmp_path / "cache_meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=field):
        validate_paper_cache(tmp_path)


def test_public_cache_rejects_different_worker_order(tmp_path):
    metadata = _metadata()
    metadata["build"]["num_workers"] = 4
    (tmp_path / "cache_meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="eight-worker"):
        validate_paper_cache(tmp_path)


def test_archive_hash_check_remains_strict(tmp_path):
    (tmp_path / "cache_meta.json").write_text(json.dumps({"splits": {}}))
    with pytest.raises(ValueError, match="archived"):
        validate_paper_cache(tmp_path)

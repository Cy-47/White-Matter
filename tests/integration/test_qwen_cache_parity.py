"""Opt-in parity check with real Qwen3 assets and streamed FineWeb-Edu text.

Run with WM_TEST_TOKENIZER_PARITY=1; downloads tokenizer files and dataset data.
"""

import hashlib
import json
import os

import numpy as np
import pytest

from scripts import prepare_fineweb_edu as builder

pytestmark = pytest.mark.skipif(
    os.environ.get("WM_TEST_TOKENIZER_PARITY") != "1", reason="requires opt-in model and dataset downloads"
)


def test_qwen_fineweb_cache_parity(tmp_path, monkeypatch):
    import gigatoken
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    api = HfApi()
    model_revision = api.model_info(builder.DEFAULT_MODEL).sha
    dataset_revision = api.dataset_info(builder.DEFAULT_DATASET).sha
    hf = AutoTokenizer.from_pretrained(builder.DEFAULT_MODEL, revision=model_revision)
    # Exercise the same backend construction as the preparation command.
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda _: hf)
    fast = builder.load_tokenizer("gigatoken")
    assert isinstance(fast, gigatoken.HFCompat)
    assert fast.eos_token_id == hf.eos_token_id

    source = load_dataset(
        builder.DEFAULT_DATASET, split=builder.DEFAULT_SPLIT, revision=dataset_revision, streaming=True
    )
    documents = []
    for sample in source:
        text = (sample.get("text") or "").strip()
        if len(text) >= builder.MIN_TEXT_CHARS:
            documents.append({"text": text})
        if len(documents) == 512:
            break
    assert len(documents) == 512
    edge_cases = [
        "",
        " ",
        "\t\n\r\n",
        "e\u0301 café 中文 العربية 🙂\u200d↔️",
        "def f(x):\n    return x ** 2\n",
        "12345678901234567890 1.234e-10",
        "literal <|endoftext|><|im_start|>assistant\n<think>hello</think><|im_end|>",
        "\x00\x01\ufffd\u200b\u00a0",
        "a" * 10000,
    ]
    texts = [document["text"] for document in documents] + edge_cases
    kwargs = {"add_special_tokens": False, "return_attention_mask": False}
    total_tokens = 0
    for start in range(0, len(texts), 32):
        batch = texts[start : start + 32]
        expected = hf(batch, **kwargs)["input_ids"]
        actual = fast(batch, **kwargs)["input_ids"]
        for offset, (left, right) in enumerate(zip(expected, actual, strict=True)):
            assert left == right, f"token IDs differ for document {start + offset}"
            total_tokens += len(left)

    # Compare complete .npy files, including EOS boundaries and batch carry-over.
    n_target = total_tokens // builder.SEQUENCE_LENGTH + len(texts) + 1
    hashes = []
    row_count = None
    for backend, tokenizer in [("hf", hf), ("gigatoken", fast)]:
        for batch_size in [17, builder.TEXT_BATCH_SIZE]:
            monkeypatch.setattr(builder, "TEXT_BATCH_SIZE", batch_size)
            out = np.empty((n_target, builder.SEQUENCE_LENGTH), dtype=np.int32)
            written, seen = builder.write_rows(documents, tokenizer, out, n_target=n_target)
            assert seen == len(documents)
            assert written > 0
            if row_count is None:
                row_count = written
            assert written == row_count
            path = tmp_path / f"{backend}-{batch_size}.npy"
            np.save(path, out[:written])
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    assert len(set(hashes)) == 1
    print(
        json.dumps(
            {
                "model": builder.DEFAULT_MODEL,
                "model_revision": model_revision,
                "dataset": builder.DEFAULT_DATASET,
                "dataset_revision": dataset_revision,
                "documents": len(documents),
                "edge_cases": len(edge_cases),
                "tokens_compared": total_tokens,
                "packed_rows": row_count,
                "sequence_length": builder.SEQUENCE_LENGTH,
                "cache_sha256": hashes[0],
                **builder.tokenizer_metadata("gigatoken"),
            },
            indent=2,
        )
    )

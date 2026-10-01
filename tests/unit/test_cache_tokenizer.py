"""Exercise real tokenizer backends without downloading model artifacts."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import prepare_fineweb_edu as builder


def test_missing_gigatoken_fails_explicitly(monkeypatch):
    monkeypatch.setitem(sys.modules, "gigatoken", None)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: SimpleNamespace(eos_token_id=99))),
    )
    with pytest.raises(ImportError, match="training,data,gigatoken"):
        builder.load_tokenizer("gigatoken")


def test_real_backends_preserve_tokens_and_packed_rows(monkeypatch):
    pytest.importorskip("gigatoken")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    texts = [
        "Hello world! A short document.",
        "def f(x):\n    return x + 1\n",
        "中文 العربية café 🙂 e\u0301",
        "literal <|endoftext|> and <|im_start|> tokens",
        "  spaces\tand\nnewlines  ",
        "",
    ]
    raw = tokenizers.Tokenizer(tokenizers.models.BPE())
    raw.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = tokenizers.decoders.ByteLevel()
    raw.train_from_iterator(
        texts,
        tokenizers.trainers.BpeTrainer(
            vocab_size=320,
            initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<|endoftext|>", "<|im_start|>"],
        ),
    )
    hf = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<|endoftext|>")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda _: hf)
    fast = builder.load_tokenizer("gigatoken")
    kwargs = {"add_special_tokens": False, "return_attention_mask": False}
    assert fast(texts, **kwargs)["input_ids"] == hf(texts, **kwargs)["input_ids"]
    assert fast.eos_token_id == hf.eos_token_id
    monkeypatch.setattr(builder, "SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(builder, "TEXT_BATCH_SIZE", 2)
    monkeypatch.setattr(builder, "MIN_TEXT_CHARS", 0)
    rows = []
    for tokenizer in (hf, fast):
        out = np.zeros((100, 8), dtype=np.int32)
        written, _ = builder.write_rows([{"text": text} for text in texts], tokenizer, out, n_target=100)
        rows.append(out[:written])
    assert len(rows[0]) > 0
    np.testing.assert_array_equal(*rows)
    assert builder.tokenizer_metadata("gigatoken")["versions"]["gigatoken"]


@pytest.mark.parametrize("installed", [False, True])
def test_auto_selects_available_backend(monkeypatch, installed):
    monkeypatch.setitem(sys.modules, "gigatoken", SimpleNamespace() if installed else None)
    assert builder.resolve_tokenizer_backend("auto") == ("gigatoken" if installed else "hf")
    assert builder.resolve_tokenizer_backend("hf") == "hf"
    assert builder.resolve_tokenizer_backend("gigatoken") == "gigatoken"


def test_auto_does_not_hide_broken_installation(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "gigatoken":
            raise ModuleNotFoundError("missing dependency", name="awkward")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        builder.resolve_tokenizer_backend("auto")

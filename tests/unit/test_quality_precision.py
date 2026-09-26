"""Quality evaluation preserves checkpoint weights and the paper's head precision."""

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM

from evals import heldout
from evals.loading import load_complete_model
from white_matter.models.vanilla import VanillaConfig


def test_cuda_heldout_preserves_fp32_checkpoint(tmp_path, monkeypatch):
    model = AutoModelForCausalLM.from_config(VanillaConfig(
        vocab_size=31, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, eos_token_id=30,
    ))
    model.save_pretrained(tmp_path)

    class Loaded(Exception):
        pass

    def inspect_load(path, *, dtype):
        loaded = load_complete_model(path, dtype=dtype)
        for expected, actual in zip(model.parameters(), loaded.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        raise Loaded

    # Exercise CUDA's loading decision without requiring a GPU allocation.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(heldout, "load_complete_model", inspect_load)
    monkeypatch.setattr("sys.argv", ["heldout", "--model", str(tmp_path),
                                    "--data-dir", str(tmp_path), "--output", str(tmp_path / "out.json")])
    with pytest.raises(Loaded):
        heldout.main()


def test_heldout_head_uses_model_autocast(monkeypatch):
    weight = torch.nn.Parameter(torch.randn(11, 8))
    model = SimpleNamespace(parameters=lambda: iter([weight]))
    monkeypatch.setattr(heldout, "model_autocast_context", lambda device: torch.autocast("cpu", dtype=torch.bfloat16),
                        raising=False)

    def score(model, loader, **kwargs):
        assert torch.nn.functional.linear(torch.ones(2, 8), weight).dtype == torch.bfloat16
        return 1.0, 2

    monkeypatch.setattr(heldout, "evaluate_tokens", score)
    assert heldout.evaluate(model, []) == (1.0, 2)

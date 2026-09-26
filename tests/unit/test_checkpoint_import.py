"""Checkpoint conversion preserves model outputs and rejects incomplete weights."""

import pytest
import torch
from transformers import AutoModelForCausalLM

from scripts import import_paper_eval_checkpoints as importer
from white_matter.models.vanilla import VanillaConfig


def _source(tmp_path, monkeypatch):
    dimensions = dict(vocab_size=31, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, eos_token_id=30)
    monkeypatch.setitem(importer.SPECS, "vanilla_24l", (VanillaConfig, dimensions, {}))
    model = AutoModelForCausalLM.from_config(VanillaConfig(**dimensions)).eval()
    source = tmp_path / "research.pt"
    torch.save(model.state_dict(), source)
    return model, source


def test_explicit_path_conversion_preserves_model(tmp_path, monkeypatch):
    original, source = _source(tmp_path, monkeypatch)
    output = tmp_path / "export"
    importer.convert("vanilla_24l", source, output)
    loaded = AutoModelForCausalLM.from_pretrained(output).eval()
    with torch.no_grad():
        ids = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(loaded(ids).logits, original(ids).logits, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        importer.convert("vanilla_24l", source, output)


def test_conversion_rejects_incomplete_checkpoint(tmp_path, monkeypatch):
    model, source = _source(tmp_path, monkeypatch)
    state = model.state_dict()
    del state["model.norm.weight"]
    torch.save(state, source)
    with pytest.raises(ValueError, match="missing"):
        importer.convert("vanilla_24l", source, tmp_path / "export")
    assert not (tmp_path / "export").exists()

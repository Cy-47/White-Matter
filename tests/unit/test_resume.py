import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from training.checkpoint import (
    load_training_checkpoint,
    restore_rng_states,
    save_training_checkpoint,
    validate_resume_training_identity,
)
from training.data import _resume_sequential_loader
from training.forward import TrainingForward
from training.optim import build_optimizers, step_optimizers
from white_matter import WhiteMatterConfig, WhiteMatterForCausalLM


def test_resume_reproduces_next_update_and_all_rng_streams(tmp_path):
    torch.manual_seed(81)
    cfg = WhiteMatterConfig(
        vocab_size=101,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        eos_token_id=100, document_separator_token_id=100,
        num_kv_channels=2,
        cyclic_groups=2,
        num_passes=3,
    )
    model = WhiteMatterForCausalLM(cfg).train()
    rng = np.random.default_rng(82)

    def optimizers(model):
        return build_optimizers(
            model,
            base_lr=3e-4,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            muon_momentum=0.95,
            muon_ns_steps=5,
            device="cpu",
        )

    def step(model, opts):
        model.zero_grad(set_to_none=True)
        ids = torch.randint(0, 100, (1, 8))
        ids[:, 3] = 100
        loss = TrainingForward(model)(model.get_input_embeddings()(ids), 3, 2, token_ids=ids)
        loss.backward()
        step_optimizers(opts)
        return loss.detach()

    opts = optimizers(model)
    step(model, opts)
    path = tmp_path / "ckpt_full.pt"
    save_training_checkpoint(
        path,
        step=1,
        model=model,
        opt_state=opts.adamw.state_dict(),
        opt_muon_state=opts.muon.state_dict(),
        rng=rng,
        n_skipped=0,
        rolling={},
        cfg_name="test",
        seed=81,
        training_config_sha256="config",
        training_source_sha256="source",
    )
    draws = (rng.random(), np.random.random(), random.random())
    expected_loss = step(model, opts)
    checkpoint = load_training_checkpoint(path)
    assert checkpoint["step"] == 1
    validate_resume_training_identity(checkpoint, current_config_sha256="config", current_source_sha256="source")
    for key in ("config", "source"):
        with pytest.raises(ValueError, match=f"{key} mismatch"):
            validate_resume_training_identity(
                checkpoint,
                current_config_sha256="changed" if key == "config" else "config",
                current_source_sha256="changed" if key == "source" else "source",
            )
    restored = WhiteMatterForCausalLM(cfg).train()
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored_opts = optimizers(restored)
    restored_opts.adamw.load_state_dict(checkpoint["opt"])
    restored_opts.muon.load_state_dict(checkpoint["opt_muon"])
    restore_rng_states(checkpoint, rng)
    assert (rng.random(), np.random.random(), random.random()) == draws
    torch.testing.assert_close(step(restored, restored_opts), expected_loss, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(dict(restored.named_parameters())[name], parameter, rtol=0, atol=0)


def test_resume_data_offset_preserves_model_rng():
    loader = DataLoader(torch.arange(12), batch_size=2)
    before = torch.get_rng_state().clone()
    iterator, epoch, skipped, batches = _resume_sequential_loader(loader, total_batches=8)
    assert (epoch, skipped, batches) == (1, 2, 6)
    torch.testing.assert_close(next(iterator), torch.tensor([4, 5]))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("declared_format", [None, "unknown"])
def test_resume_rejects_unsupported_checkpoint_format(tmp_path, declared_format):
    path = tmp_path / "ckpt_full.pt"
    torch.save({"model_checkpoint_format": declared_format}, path)
    with pytest.raises(ValueError, match="unsupported model_checkpoint_format"):
        load_training_checkpoint(path)


def test_source_identity_covers_both_library_and_runner(tmp_path, monkeypatch):
    import white_matter
    from training import checkpoint as provenance

    library = tmp_path / "white_matter"
    runner = tmp_path / "training"
    for owner in (library, runner):
        owner.mkdir()
        (owner / "__init__.py").write_text("")
        (owner / "implementation.py").write_text("value = 1\n")
    monkeypatch.setattr(white_matter, "__file__", str(library / "__init__.py"))
    monkeypatch.setattr(provenance, "__file__", str(runner / "provenance.py"))
    original = provenance.training_source_sha256()
    checkpoint = {"training_config_sha256": "config", "training_source_sha256": original}
    for owner in (library, runner):
        path = owner / "implementation.py"
        path.write_text("value = 2\n")
        changed = provenance.training_source_sha256()
        assert changed != original
        with pytest.raises(ValueError, match="source mismatch"):
            validate_resume_training_identity(checkpoint, current_config_sha256="config", current_source_sha256=changed)
        path.write_text("value = 1\n")
        assert provenance.training_source_sha256() == original

"""Protect existing trajectories before a training process initializes CUDA."""

from argparse import Namespace

import pytest

from training.checkpoint import resolve_training_resume
from training.recipes import load_recipe


@pytest.mark.parametrize("artifact", ["metrics.jsonl", "latest", "final", "ckpt_full.pt.tmp"])
@pytest.mark.parametrize("external_resume", [False, True])
def test_training_rejects_orphaned_run_artifacts(tmp_path, artifact, external_resume, monkeypatch):
    from training import engine

    output = tmp_path / "run"
    output.mkdir()
    target = output / artifact
    if artifact in {"latest", "final"}:
        target.mkdir()
    else:
        target.write_text("original trajectory\n")
    external = tmp_path / "external.pt"
    external.write_text("checkpoint")

    def unexpected_setup():
        pytest.fail("training initialized distributed execution before checking its output")

    monkeypatch.setattr(engine, "setup_distributed", unexpected_setup)
    recipe = load_recipe("recipes/paper/white_matter_k8.yaml")
    with pytest.raises(FileExistsError, match="run artifacts"):
        engine.train(
            recipe,
            Namespace(
                output_dir=output,
                data_dir=tmp_path / "data",
                resume_from_checkpoint=external if external_resume else None,
            ),
        )
    if target.is_file():
        assert target.read_text() == "original trajectory\n"


def test_resume_allows_existing_trajectory_only_with_its_checkpoint(tmp_path):
    output = tmp_path / "run"
    assert resolve_training_resume(output, None) == output / "ckpt_full.pt"
    output.mkdir()
    assert resolve_training_resume(output, None) == output / "ckpt_full.pt"
    checkpoint = output / "ckpt_full.pt"
    checkpoint.write_text("checkpoint")
    (output / "metrics.jsonl").write_text("existing metrics")
    assert resolve_training_resume(output, None) == checkpoint
    assert resolve_training_resume(output, checkpoint) == checkpoint
    external = tmp_path / "external.pt"
    external.write_text("other checkpoint")
    with pytest.raises(ValueError, match="already has"):
        resolve_training_resume(output, external)
    assert resolve_training_resume(tmp_path / "new_run", external) == external
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_training_resume(tmp_path / "new_run", tmp_path / "missing.pt")


def test_resume_recognizes_symlinked_checkpoint(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    target = tmp_path / "saved.pt"
    target.write_text("checkpoint")
    checkpoint = output / "ckpt_full.pt"
    checkpoint.symlink_to(target)

    for requested in (None, checkpoint, target):
        assert resolve_training_resume(output, requested).resolve() == target

    other = tmp_path / "other.pt"
    other.write_text("different checkpoint")
    with pytest.raises(ValueError, match="already has"):
        resolve_training_resume(output, other)

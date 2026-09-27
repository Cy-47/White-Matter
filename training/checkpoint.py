"""Atomic checkpoints of model, optimizers, RNG streams and training position."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

HF_FULL_STATE_DICT_FORMAT = "hf_full_state_dict_v2"
RANKED_RNG_STATE_FORMAT = "white_matter_ranked_rng_state_v1"


def resolve_training_resume(output_dir: Path, resume_from: Path | None) -> Path:
    """Reject existing run artifacts unless continuing that directory's checkpoint."""
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint = output_dir / "ckpt_full.pt"
    resume = Path(resume_from).expanduser().resolve() if resume_from is not None else checkpoint
    if resume_from is not None and not resume.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"results path is not a directory: {output_dir}")
    if checkpoint.exists():
        if not checkpoint.is_file():
            raise ValueError(f"resume checkpoint is not a file: {checkpoint}")
        if resume.resolve() != checkpoint.resolve():
            raise ValueError("output directory already has ckpt_full.pt; choose a new output directory")
    else:
        artifacts = [
            name for name in ("metrics.jsonl", "latest", "final", "ckpt_full.pt.tmp") if (output_dir / name).exists()
        ]
        if artifacts:
            raise FileExistsError(
                f"output directory contains run artifacts without a resume checkpoint: {', '.join(artifacts)}; "
                "choose a new output directory"
            )
    return resume


def capture_local_rng_state(
    rng: np.random.Generator,
    *,
    cuda_device: int | None = None,
) -> dict[str, Any]:
    """Capture every RNG stream owned by the current training rank."""
    cuda_state = None
    if torch.cuda.is_available():
        if cuda_device is None:
            cuda_device = torch.cuda.current_device()
        cuda_state = torch.cuda.get_rng_state(cuda_device)
    return {
        "np_rng": rng.bit_generator.state,
        "np_global": np.random.get_state(),  # noqa: NPY002 - preserve third-party global RNG state.
        "py_random": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_state,
    }


def gather_rng_states_for_rank0(
    rng: np.random.Generator,
    *,
    rank: int,
    world_size: int,
    cuda_device: int | None = None,
) -> list[dict[str, Any]] | None:
    """Gather one complete local RNG snapshot per rank for a checkpoint.

    Every distributed rank must call this function in the same order. Only rank
    zero receives the gathered list; single-process training returns a one-item
    list without a collective.
    """
    if type(rank) is not int or type(world_size) is not int:
        raise TypeError("RNG checkpoint rank and world_size must be integers")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"invalid RNG checkpoint topology: rank={rank} world_size={world_size}")
    local_state = capture_local_rng_state(rng, cuda_device=cuda_device)
    if world_size == 1:
        return [local_state]
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("distributed RNG checkpointing requires an initialized process group")
    actual_world_size = dist.get_world_size()
    actual_rank = dist.get_rank()
    if (actual_rank, actual_world_size) != (rank, world_size):
        raise RuntimeError(
            "RNG checkpoint topology differs from the process group: "
            f"configured=({rank}, {world_size}) actual=({actual_rank}, {actual_world_size})"
        )
    gathered: list[dict[str, Any] | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local_state, gathered, dst=0)
    if gathered is None:
        return None
    if any(state is None for state in gathered):
        raise RuntimeError("rank-zero RNG gather returned an incomplete rank list")
    return [state for state in gathered if state is not None]


def save_training_checkpoint(
    path,
    *,
    step,
    model,
    opt_state: Mapping[str, Any],
    rng,
    n_skipped,
    rolling,
    cfg_name,
    seed,
    training_config_sha256: str,
    training_source_sha256: str,
    rng_states_by_rank: list[dict[str, Any]] | None = None,
    opt_muon_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if rng_states_by_rank is None:
        rng_states_by_rank = [capture_local_rng_state(rng)]
    payload = {
        **(extra or {}),
        "step": int(step),  # next step to run on resume
        # Trainer-state wrapper around the same canonical full model state used
        # by evaluation checkpoints.
        "model": model.state_dict(),
        "model_checkpoint_format": HF_FULL_STATE_DICT_FORMAT,
        "opt": dict(opt_state),
        **({"opt_muon": opt_muon_state} if opt_muon_state is not None else {}),
        "rng_state_format": RANKED_RNG_STATE_FORMAT,
        "rng_states_by_rank": rng_states_by_rank,
        "n_skipped": int(n_skipped),
        "rolling": dict(rolling),
        "cfg_name": cfg_name,
        "seed": int(seed),
        "training_config_sha256": training_config_sha256,
        "training_source_sha256": training_source_sha256,
    }
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on POSIX


def load_training_checkpoint(path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_checkpoint_format") != HF_FULL_STATE_DICT_FORMAT:
        raise ValueError(f"unsupported model_checkpoint_format {checkpoint.get('model_checkpoint_format')!r}")
    return checkpoint


def validate_resume_training_identity(
    checkpoint: Mapping[str, Any],
    *,
    current_config_sha256: str,
    current_source_sha256: str,
) -> None:
    """Require exact config and source matches before restoring trainer state."""

    for kind, current in (("config", current_config_sha256), ("source", current_source_sha256)):
        saved = checkpoint.get(f"training_{kind}_sha256")
        if not isinstance(saved, str) or saved != current:
            raise ValueError(f"resume training {kind} mismatch: checkpoint={saved!r} current={current}")


def restore_rng_states(
    checkpoint: dict[str, Any],
    rng: np.random.Generator,
    *,
    rank: int = 0,
    world_size: int = 1,
    cuda_device: int | None = None,
) -> None:
    """Restore the saved rank's RNG streams after checking checkpoint topology."""
    if checkpoint.get("rng_state_format") != RANKED_RNG_STATE_FORMAT:
        raise ValueError(f"unsupported checkpoint RNG format {checkpoint.get('rng_state_format')!r}")
    ranked = checkpoint["rng_states_by_rank"]
    if len(ranked) != world_size:
        raise ValueError(f"checkpoint RNG world_size={len(ranked)} does not match current world_size={world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid current RNG rank {rank} for world_size={world_size}")
    state = ranked[rank]
    rng.bit_generator.state = state["np_rng"]
    np.random.set_state(state["np_global"])  # noqa: NPY002 - restore the checkpointed global RNG.
    random.setstate(state["py_random"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if torch.cuda.is_available() and cuda_state is not None:
        if cuda_device is None:
            cuda_device = torch.cuda.current_device()
        torch.cuda.set_rng_state(cuda_state, cuda_device)


def training_source_sha256(*, model_config: Any | None = None) -> str:
    import white_matter

    owners = {"white_matter": Path(white_matter.__file__).resolve().parent, "training": Path(__file__).resolve().parent}
    if model_config is not None:
        module = importlib.import_module(type(model_config).__module__)
        module_file = Path(module.__file__).resolve()
        if not module_file.is_relative_to(owners["white_matter"]):
            # Study-local model implementations must be part of the resume
            # identity just as package model implementations are.
            owners[f"external_model/{model_config.model_type}"] = module_file.parent
    digest = hashlib.sha256()
    for owner, directory in sorted(owners.items()):
        for path in sorted(directory.rglob("*.py")):
            digest.update(f"{owner}/{path.relative_to(directory).as_posix()}".encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def resolved_config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()

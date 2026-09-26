"""Shared data and scoring protocol for the paper's 20k-step ablations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


# Metadata fingerprints for the archived paper cache.
_PAPER_METADATA_SHA256 = {
    "cache_meta.json": "bc97ff095a4df6d2d9f11e6e3ed0e8f9f33d7f7d6c25b927ed34a091a1ce8e36",
    "tokenize_meta.json": "e8f77b1d39d60a107a8d3337fc2b7ede447f3078bb46dbef9bc56c8acd0bd936",
}
PAPER_SEQUENCE_LENGTH = 2048
PAPER_TEST_SEQUENCES = 5000
PAPER_TEST_TARGETS = PAPER_TEST_SEQUENCES * (PAPER_SEQUENCE_LENGTH - 1)


PAPER_SMALL_MODEL = {
    "vocab_size": 151_936,
    "hidden_size": 512,
    "intermediate_size": 1_536,
    "num_hidden_layers": 16,
    "num_attention_heads": 6,
    "num_key_value_heads": 3,
    "head_dim": 96,
    "max_position_embeddings": 4_096,
    "rope_theta": 1_000_000.0,
    "rms_norm_eps": 1.0e-6,
    "eos_token_id": 151_643,
    "document_separator_token_id": 151_643,
    "pad_token_id": None,
    "residual_dtype": "fp32",
}


def validate_paper_training_recipe(recipe, expected_model) -> None:
    """Pin architecture, data source, and optimizer independently of YAML files."""
    from training.recipes import DataRecipe, OptimizerRecipe, model_recipe_keys

    if recipe.loss_backend != "torch" or recipe.ar_cuda_graph:
        raise ValueError("paper study training backend differs from the fixed protocol")
    if recipe.data != DataRecipe("Qwen/Qwen3-0.6B-Base", PAPER_SEQUENCE_LENGTH, 151_643):
        raise ValueError("paper study data recipe differs from the fixed protocol")
    expected_optimizer = OptimizerRecipe(
        learning_rate=3.0e-4, weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.95,
        muon_momentum=0.95, muon_ns_steps=5, warmup_fraction=0.02,
        minimum_lr_fraction=0.1, max_gradient_norm=1.0,
    )
    if recipe.optimizer != expected_optimizer:
        raise ValueError("paper study optimizer differs from the fixed protocol")
    for key in model_recipe_keys(type(expected_model)):
        if getattr(recipe.model, key, None) != getattr(expected_model, key, None):
            raise ValueError(f"paper study model {key} differs from the fixed protocol")


def validate_paper_cache(cache_dir: str | Path) -> dict:
    """Accept archived data or the public eight-worker rebuild protocol.

    Metadata and array layout are checked; neither route hashes the token array.
    A rebuild is not a claim of byte-for-byte equality with archived tokens.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    meta = json.loads((cache_dir / "cache_meta.json").read_text())
    if meta.get("build", {}).get("tool") == "prepare_fineweb_edu.py":
        total = 9_772_625
        expected = {
            "n_total": total, "max_length": PAPER_SEQUENCE_LENGTH,
            "model": "Qwen/Qwen3-0.6B-Base",
            "dataset": "karpathy/fineweb-edu-100b-shuffle",
            "dataset_config": "", "dataset_split": "train", "text_field": "text",
            "packing": "eos_crossdoc", "eos_id": 151_643,
            "approx_train_tokens": 9_765_625 * PAPER_SEQUENCE_LENGTH,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"rebuilt paper cache {key} must equal {value!r}")
        targets = [total // 8 + int(worker < total % 8) for worker in range(8)]
        if meta["build"].get("num_workers") != 8 or meta["build"].get("per_worker_target") != targets:
            raise ValueError("rebuilt paper cache requires the eight-worker row order")
    else:
        for name, expected_hash in _PAPER_METADATA_SHA256.items():
            path = cache_dir / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected_hash:
                raise ValueError(f"{path} differs from the archived paper cache metadata")
    splits = meta["splits"]
    expected_splits = {"n_train": 9_765_625, "n_val": 2_000, "n_test": PAPER_TEST_SEQUENCES}
    if any(int(splits[key]) != value for key, value in expected_splits.items()):
        raise ValueError("cache split sizes differ from paper Figure 7b")
    tokenized = np.load(cache_dir / "tokenized.npy", mmap_mode="r")
    if tokenized.shape != (9_772_625, PAPER_SEQUENCE_LENGTH):
        raise ValueError(f"tokenized cache shape differs from paper Figure 7b: {tokenized.shape}")
    if tokenized.dtype != np.dtype("int32"):
        raise ValueError("paper cache tokenized.npy must contain int32 token IDs")
    return meta


def evaluate_fixed_passes(model, loader, *, num_passes: int) -> tuple[float, int]:
    """Score the fixed study schedule using the shared evaluator."""
    from evals.execution import evaluate

    if type(num_passes) is not int or num_passes < 1:
        raise ValueError("num_passes must be a positive integer")
    return evaluate(model, loader, num_passes=num_passes, loss_backend="cce")

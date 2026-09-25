"""Strict, typed training recipes.

Recipes are closed schemas: unknown keys fail at load time.
Only logging and checkpoint cadence may be overridden from the command line.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoConfig, PretrainedConfig

from white_matter.models import register_models

register_models()


def model_recipe_keys(config_class):
    return frozenset(
        {"model_type"}
        | {
            name
            for cls in config_class.__mro__
            if cls is not PretrainedConfig and issubclass(cls, PretrainedConfig)
            for name, parameter in inspect.signature(cls.__init__).parameters.items()
            if name != "self" and parameter.kind is inspect.Parameter.KEYWORD_ONLY
        }
    )


@dataclass(frozen=True)
class OptimizerRecipe:
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    muon_momentum: float
    muon_ns_steps: int
    warmup_fraction: float
    minimum_lr_fraction: float
    max_gradient_norm: float
    distributed_muon: bool = False

    def __post_init__(self) -> None:
        if type(self.distributed_muon) is not bool:
            raise ValueError("optimizer.distributed_muon must be boolean")
        positive = {
            "learning_rate": self.learning_rate,
            "muon_momentum": self.muon_momentum,
            "max_gradient_norm": self.max_gradient_norm,
        }
        for name, value in positive.items():
            if type(value) is not float or value <= 0:
                raise ValueError(f"optimizer.{name} must be a positive float")
        if type(self.weight_decay) is not float or self.weight_decay < 0:
            raise ValueError("optimizer.weight_decay must be a non-negative float")
        for name, value in {
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "warmup_fraction": self.warmup_fraction,
            "minimum_lr_fraction": self.minimum_lr_fraction,
        }.items():
            if type(value) is not float or not 0 <= value < 1:
                raise ValueError(f"optimizer.{name} must be a float in [0, 1)")
        if type(self.muon_ns_steps) is not int or self.muon_ns_steps <= 0:
            raise ValueError("optimizer.muon_ns_steps must be a positive integer")


@dataclass(frozen=True)
class DataRecipe:
    tokenizer: str
    sequence_length: int
    eos_token_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer, str) or not self.tokenizer:
            raise ValueError("data.tokenizer must be a non-empty string")
        if type(self.sequence_length) is not int or self.sequence_length <= 1:
            raise ValueError("data.sequence_length must be an integer greater than one")
        if type(self.eos_token_id) is not int or self.eos_token_id < 0:
            raise ValueError("data.eos_token_id must be a non-negative integer")


@dataclass(frozen=True)
class TrainingRecipe:
    name: str
    model: PretrainedConfig
    data: DataRecipe
    optimizer: OptimizerRecipe
    steps: int
    global_batch_size: int
    gradient_accumulation_steps: int
    seed: int
    no_gradient_passes: int | None = None
    gradient_passes: int | None = None
    loss_backend: str = "torch"
    ar_cuda_graph: bool = False

    def __post_init__(self) -> None:
        if self.loss_backend not in {"torch", "cce"}:
            raise ValueError("loss_backend must be 'torch' or 'cce'")
        if type(self.ar_cuda_graph) is not bool:
            raise ValueError("ar_cuda_graph must be boolean")
        if self.ar_cuda_graph and self.model.execution_mode != "autoregressive":
            raise ValueError("ar_cuda_graph requires autoregressive execution")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("recipe.name must be a non-empty string")
        for name, value in {
            "steps": self.steps,
            "global_batch_size": self.global_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"recipe.{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("recipe.seed must be a non-negative integer")
        if self.global_batch_size % self.gradient_accumulation_steps:
            raise ValueError("global_batch_size must be divisible by gradient_accumulation_steps")
        if self.data.eos_token_id != self.model.eos_token_id:
            raise ValueError("data and model eos_token_id values must match")

        # WhiteMatter's exact-AR control retains its nominal pass metadata,
        # even though the trainer executes its single exact token sweep.
        iterative = hasattr(self.model, "num_passes")
        if iterative:
            if type(self.no_gradient_passes) is not int or self.no_gradient_passes < 0:
                raise ValueError("iterative models require non-negative no_gradient_passes")
            if type(self.gradient_passes) is not int or self.gradient_passes <= 0:
                raise ValueError("iterative models require positive gradient_passes")
            if self.no_gradient_passes + self.gradient_passes != self.model.num_passes:
                raise ValueError("no_gradient_passes + gradient_passes must equal model.num_passes")
        elif self.no_gradient_passes is not None or self.gradient_passes is not None:
            raise ValueError("single-pass models must not define iterative pass counts")

    @property
    def sha256(self) -> str:
        payload = asdict(self)
        payload["model"] = {name: getattr(self.model, name) for name in model_recipe_keys(type(self.model))}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _strict_dataclass(cls, raw: Any, *, where: str):
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping")
    allowed = {field.name for field in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown {where} keys: {sorted(unknown)}")
    missing = {
        field.name for field in fields(cls) if field.default is MISSING and field.default_factory is MISSING
    } - set(raw)
    if missing:
        raise ValueError(f"missing {where} keys: {sorted(missing)}")
    return cls(**raw)


def load_recipe(path: str | Path) -> TrainingRecipe:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    allowed = {field.name for field in fields(TrainingRecipe)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown recipe keys: {sorted(unknown)}")

    model_raw = raw.pop("model", None)
    data_raw = raw.pop("data", None)
    optimizer_raw = raw.pop("optimizer", None)
    if not isinstance(model_raw, dict):
        raise ValueError("recipe.model must be a mapping")
    model_type = model_raw.pop("model_type", None)
    if not isinstance(model_type, str):
        raise ValueError("recipe.model.model_type must be a model family name")
    config_class = type(AutoConfig.for_model(model_type))
    unknown_model_keys = set(model_raw) - model_recipe_keys(config_class)
    if unknown_model_keys:
        raise ValueError(f"unknown model keys: {sorted(unknown_model_keys)}")
    model = config_class(**model_raw)
    data = _strict_dataclass(DataRecipe, data_raw, where="data")
    optimizer = _strict_dataclass(OptimizerRecipe, optimizer_raw, where="optimizer")
    return TrainingRecipe(model=model, data=data, optimizer=optimizer, **raw)


def per_rank_batch_size(recipe: TrainingRecipe, world_size: int) -> int:
    divisor = world_size * recipe.gradient_accumulation_steps
    if world_size < 1 or recipe.global_batch_size % divisor:
        raise ValueError("global_batch_size must be divisible by world_size * gradient_accumulation_steps")
    return recipe.global_batch_size // divisor

"""Reusable feedback operators, components, and optional Hugging Face models."""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"
_FAMILIES = {"WhiteMatter": "white_matter", "LCKV": "lckv", "Vanilla": "vanilla", "FusedKV": "fusedkv"}
__all__ = [prefix + suffix for prefix in _FAMILIES for suffix in ("Config", "PreTrainedModel", "Model", "ForCausalLM")]


def __getattr__(name: str) -> Any:
    for prefix, family in _FAMILIES.items():
        if name in __all__ and name.startswith(prefix):
            value = getattr(import_module(f"white_matter.models.{family}"), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

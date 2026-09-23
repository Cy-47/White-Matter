"""Cyclic causal attention over explicitly indexed query slots."""

from .functional import cyclic_attention
from .metadata import CyclicAttentionMetadata, prepare_cyclic_attention_metadata

__all__ = ["CyclicAttentionMetadata", "cyclic_attention", "prepare_cyclic_attention_metadata"]

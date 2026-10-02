"""Pre-norm decoder layer that reads from the shared K/V channels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from white_matter.layers import FeedbackAttention
from white_matter.modules.checkpointing import checkpoint_pointwise
from white_matter.modules.mlp import FeedForward
from white_matter.ops import CyclicAttentionMetadata, StrictCausalMetadata


class FeedbackDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention: FeedbackAttention,
        mlp: FeedForward,
        *,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        q_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        decode_key_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        metadata: CyclicAttentionMetadata | StrictCausalMetadata | None = None,
        *,
        query_stride: int | None = None,
        query_offset: int = 0,
        cache_seqlens: torch.Tensor | None = None,
        committed_prefix: bool = False,
        jacobi: bool = False,
        prefix_length: int = 0,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            key_states,
            value_states,
            q_position_embeddings,
            decode_key_mask=decode_key_mask,
            document_ids=document_ids,
            metadata=metadata,
            query_stride=query_stride,
            query_offset=query_offset,
            cache_seqlens=cache_seqlens,
            committed_prefix=committed_prefix,
            jacobi=jacobi,
            prefix_length=prefix_length,
        )
        mlp_input = self.post_attention_layernorm(hidden_states)
        mlp_output = checkpoint_pointwise(self.mlp, mlp_input) if torch.is_grad_enabled() else self.mlp(mlp_input)
        return hidden_states + mlp_output


def run_feedback_layers(
    layers: nn.ModuleList,
    hidden: torch.Tensor,
    keys: Sequence[torch.Tensor],
    values: Sequence[torch.Tensor],
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    *,
    return_output: bool = True,
    include_top_output: bool = False,
    **attention_kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Retain layer inputs and optionally the final output for the next pool."""
    states = []
    for index, layer in enumerate(layers):
        states.append(hidden)
        if not return_output and not include_top_output and index == len(layers) - 1:
            break
        channel = index % len(keys)
        hidden = layer(hidden, keys[channel], values[channel], position_embeddings, **attention_kwargs)
    if include_top_output:
        states.append(hidden)
    return hidden, states

"""Qwen-native FusedKV single-pass decoder.

The lower half of the decoder computes ordinary Qwen self-attention. The
upper half keeps its Q/O projections and MLP but
reconstructs K/V from layer 0 and the final lower-half layer.  This module uses
explicit tensors rather than mutable cross-layer globals so checkpointing,
compilation, and concurrent model instances cannot observe stale state.
"""

from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn as nn

from white_matter.models import _qwen3 as qwen3
from white_matter.modules import RotaryEmbedding
from white_matter.modules.rotary import rotate_half
from ..configuration_base import DecoderConfig

from ..modeling_base import DecoderForCausalLM, DecoderModel, DecoderPreTrainedModel
from ..decoder_utils import prepare_attention_inputs, prepare_decoder_inputs
from .configuration_fusedkv import FusedKVConfig


class FusedKVFusion(nn.Module):
    """Direct bottom/middle mixtures; key weights are shared across each RoPE pair."""

    def __init__(self, num_key_value_heads: int, head_dim: int) -> None:
        super().__init__()
        self.k_bottom = nn.Parameter(torch.empty(num_key_value_heads, head_dim // 2))
        self.k_middle = nn.Parameter(torch.empty(num_key_value_heads, head_dim // 2))
        self.v_bottom = nn.Parameter(torch.empty(num_key_value_heads, head_dim))
        self.v_middle = nn.Parameter(torch.empty(num_key_value_heads, head_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.parameters():
            nn.init.normal_(weight, mean=0.0, std=1.0)

    def forward(self, bottom_key, middle_key, bottom_value, middle_value):
        k_bottom = torch.cat((self.k_bottom, self.k_bottom), dim=-1)
        k_middle = torch.cat((self.k_middle, self.k_middle), dim=-1)
        # Round each FP32 weighted source to activation dtype before adding.
        return (
            (bottom_key.float() * k_bottom.float()[None, :, None, :]).to(bottom_key.dtype)
            + (middle_key.float() * k_middle.float()[None, :, None, :]).to(middle_key.dtype),
            (bottom_value.float() * self.v_bottom.float()[None, :, None, :]).to(bottom_value.dtype)
            + (middle_value.float() * self.v_middle.float()[None, :, None, :]).to(middle_value.dtype),
        )


class FusedKVReconstructionAttention(nn.Module):
    """Qwen attention with layer-local Q/O and externally reconstructed K/V."""

    def __init__(self, config: DecoderConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(query_shape)).transpose(1, 2)
        cos, sin = position_embeddings
        query_states = query_states * cos.unsqueeze(1) + rotate_half(query_states) * sin.unsqueeze(1)
        if key_states.dtype != query_states.dtype:
            key_states = key_states.to(query_states.dtype)
        if value_states.dtype != query_states.dtype:
            value_states = value_states.to(query_states.dtype)

        attention_output, _ = qwen3.attention_forward(
            self, query_states, key_states, value_states, attention_mask, **kwargs
        )
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention_output)


class FusedKVSourceLayer(qwen3.Qwen3DecoderLayer):
    def forward(self, hidden_states: torch.Tensor, **attention_kwargs):
        attention_output, key_states, value_states = self.self_attn(
            self.input_layernorm(hidden_states), return_kv=True, **attention_kwargs
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, key_states, value_states


class FusedKVReconstructionLayer(qwen3.Qwen3DecoderLayer):
    attention_class = FusedKVReconstructionAttention


class FusedKVDecoder(DecoderPreTrainedModel):
    """Single-pass baseline reconstructing upper-layer K/V from two sources."""

    _no_split_modules = ["FusedKVSourceLayer", "FusedKVReconstructionLayer"]

    def __init__(self, config: FusedKVConfig) -> None:
        num_hidden_layers = config.num_hidden_layers
        head_dim = config.head_dim
        num_key_value_heads = config.num_key_value_heads
        fusedkv_storage_layers = num_hidden_layers // 2
        super().__init__(config)

        self.source_layers = nn.ModuleList(
            [FusedKVSourceLayer(config, layer_idx) for layer_idx in range(fusedkv_storage_layers)]
        )
        self.reconstruction_layers = nn.ModuleList(
            [
                FusedKVReconstructionLayer(config, layer_idx)
                for layer_idx in range(fusedkv_storage_layers, num_hidden_layers)
            ]
        )
        self.fusions = nn.ModuleList([FusedKVFusion(num_key_value_heads, head_dim) for _ in self.reconstruction_layers])
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.post_init()
        # The generic HF initializer handles Linear/RMSNorm but intentionally
        # knows nothing about the paper's N(0,1) coefficient initialization.
        for fusion in self.fusions:
            fusion.reset_parameters()
        self._initialize_direct_equivalent()

    @torch.no_grad()
    def _initialize_direct_equivalent(self) -> None:
        """Collapse an N(0,1) iterative chain into independent direct weights.

        Appendix A.3's equivalent initialization samples the same auxiliary
        coefficients as iterative FusedKV, recursively expands every upper
        cache into bottom/middle source coefficients, and then trains those
        expanded coefficients independently.  The resulting forward graph is
        direct FusedKV; only its step-zero function matches the iterative arm.
        """
        for previous, fusion in pairwise(self.fusions):
            key_bottom = fusion.k_bottom * previous.k_bottom
            key_middle = fusion.k_bottom * previous.k_middle + fusion.k_middle
            value_bottom = fusion.v_bottom * previous.v_bottom + fusion.v_middle
            value_middle = fusion.v_bottom * previous.v_middle
            fusion.k_bottom.copy_(key_bottom)
            fusion.k_middle.copy_(key_middle)
            fusion.v_bottom.copy_(value_bottom)
            fusion.v_middle.copy_(value_middle)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        num_passes: int | None = None,
        num_gradient_passes: int | None = None,
        document_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        checkpoint_chunk_size: int = 0,
    ) -> torch.Tensor:
        inputs_embeds, attention_mask, document_ids = prepare_decoder_inputs(
            inputs_embeds, self.config, attention_mask, document_ids
        )
        if checkpoint_chunk_size:
            raise ValueError("checkpoint_chunk_size is supported only for WhiteMatter autoregressive execution")
        if num_passes not in {None, 1}:
            raise ValueError("FusedKV decoding is single-pass")
        attention_kwargs = prepare_attention_inputs(
            inputs_embeds,
            document_ids,
            self.rotary_emb,
            self.config._attn_implementation,
        )
        hidden_states = inputs_embeds
        bottom_key = bottom_value = None
        for layer_idx, layer in enumerate(self.source_layers):
            hidden_states, middle_key, middle_value = layer(hidden_states, **attention_kwargs)
            if layer_idx == 0:
                bottom_key, bottom_value = middle_key, middle_value
        assert bottom_key is not None and bottom_value is not None
        for layer, fusion in zip(self.reconstruction_layers, self.fusions, strict=True):
            key_states, value_states = fusion(bottom_key, middle_key, bottom_value, middle_value)
            hidden_states = layer(
                hidden_states,
                key_states=key_states,
                value_states=value_states,
                **attention_kwargs,
            )
        return hidden_states


class FusedKVPreTrainedModel(DecoderPreTrainedModel):
    config_class = FusedKVConfig


class FusedKVModel(DecoderModel, FusedKVPreTrainedModel):
    config_class = FusedKVConfig
    decoder_class = FusedKVDecoder


class FusedKVForCausalLM(DecoderForCausalLM, FusedKVPreTrainedModel):
    config_class = FusedKVConfig
    model_class = FusedKVModel

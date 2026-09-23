"""Qwen3 decoder components using PyTorch's native RMSNorm.

Derived from Hugging Face Transformers; this local subset is maintained here.
"""

# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
from white_matter.layers import WhiteMatterAttention
from white_matter.layers.backends import attention_forward as run_attention, pack_kv_cache
from white_matter.blocks.decoder_layer import FeedbackDecoderLayer
from white_matter.modules import GatedMLP
from white_matter.modules.rotary import apply_rotary_pos_emb

from .configuration_base import DecoderConfig


def attention_forward(module, query, key, value, attention_mask, **kwargs):
    """Use the configured HF attention backend with the decoder's numerical policy."""
    cached = kwargs.pop("cached_attention", False)
    implementation = module.config._attn_implementation
    lengths = kwargs.pop("cache_seqlens", None)
    if cached:
        if attention_mask is not None and implementation == "flash_attention_2" and query.shape[2] == 1:
            key, value, lengths = pack_kv_cache(key, value, attention_mask[:, 0, 0] == 0)
            attention_mask = None
        if attention_mask is not None:
            implementation, lengths = "sdpa", None
        return run_attention(
            query, key, value, attention_mask=attention_mask, scaling=module.scaling,
            implementation=implementation, cache_seqlens=lengths, num_splits=module.num_splits,
        ), None
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(implementation, eager_attention_forward)
    return interface(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=module.scaling,
        sliding_window=None,
        **kwargs,
    )


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DecoderConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.num_splits = 0  # FA decode tuning; zero keeps its automatic choice.

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        return_kv: bool = False,
        past_key_values=None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # Prefill attends to the actual input; static cache capacity is not context.
            cached = kwargs.get("cached_attention", False)
            cached_key, cached_value = past_key_values.update(key_states, value_states, self.layer_idx)
            if cached:
                key_states, value_states = cached_key, cached_value
                kwargs["cache_seqlens"] = past_key_values.lengths(self.layer_idx, hidden_states.shape[0])
            if self.config._attn_implementation == "flash_attention_2":
                # Match FA's cast before its logging branch, which breaks compilation.
                query_states, key_states = query_states.to(value_states.dtype), key_states.to(value_states.dtype)

        attn_output, _ = attention_forward(self, query_states, key_states, value_states, attention_mask, **kwargs)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return (attn_output, key_states, value_states) if return_kv else attn_output


class Qwen3DecoderLayer(nn.Module):
    attention_class = Qwen3Attention

    def __init__(self, config: DecoderConfig, layer_idx: int):
        super().__init__()

        self.self_attn = self.attention_class(config=config, layer_idx=layer_idx)

        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, **attention_kwargs) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), **attention_kwargs)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


def make_feedback_layers(config, num_layers, *, strict_causal=False):
    layers = []
    for i in range(num_layers):
        attention = WhiteMatterAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_implementation=config._attn_implementation,
            strict_causal=strict_causal,
        )
        mlp = GatedMLP(config.hidden_size, config.intermediate_size)
        layers.append(FeedbackDecoderLayer(config.hidden_size, attention, mlp, rms_norm_eps=config.rms_norm_eps))
    return layers

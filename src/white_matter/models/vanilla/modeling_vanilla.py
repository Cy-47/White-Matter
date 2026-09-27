"""Pure vanilla Qwen3 baseline used in the paper."""

from __future__ import annotations

import torch
import torch.nn as nn

from white_matter.models import _qwen3 as qwen3
from white_matter.modules import RotaryEmbedding
from white_matter.modules.precision import cast_residual

from ..cache import DecoderCache
from ..decoder_utils import prepare_attention_inputs, prepare_decoder_inputs, run_feedforward_layers
from ..modeling_base import DecoderForCausalLM, DecoderModel, DecoderPreTrainedModel
from .configuration_vanilla import VanillaConfig


class VanillaDecoder(DecoderPreTrainedModel):
    """Vanilla Qwen3 decoder with one causal pass through all layers."""

    _no_split_modules = ["Qwen3DecoderLayer"]

    def __init__(self, config: VanillaConfig) -> None:
        num_hidden_layers = config.num_hidden_layers
        super().__init__(config)
        self.cache_prefix_slots = (0,) * num_hidden_layers
        self.layers = nn.ModuleList([qwen3.Qwen3DecoderLayer(config, i) for i in range(num_hidden_layers)])
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.post_init()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        num_passes: int | None = None,
        num_gradient_passes: int | None = None,
        document_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        checkpoint_chunk_size: int = 0,
        past_key_values: DecoderCache | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if checkpoint_chunk_size:
            raise ValueError("checkpoint_chunk_size is supported only for WhiteMatter autoregressive execution")
        if num_passes not in {None, 1}:
            raise ValueError("vanilla decoding is single-pass")
        if past_key_values is None:
            inputs_embeds, attention_mask, document_ids = prepare_decoder_inputs(
                inputs_embeds, self.config, attention_mask, document_ids
            )
            position_ids = None
        else:
            inputs_embeds = cast_residual(inputs_embeds, residual_dtype=self.config.residual_dtype)
        args = prepare_attention_inputs(
            inputs_embeds,
            document_ids,
            self.rotary_emb,
            self.config._attn_implementation,
            position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        return run_feedforward_layers(self.layers, inputs_embeds, **args)


class VanillaPreTrainedModel(DecoderPreTrainedModel):
    config_class = VanillaConfig


class VanillaModel(DecoderModel, VanillaPreTrainedModel):
    config_class = VanillaConfig
    decoder_class = VanillaDecoder


class VanillaForCausalLM(DecoderForCausalLM, VanillaPreTrainedModel):
    config_class = VanillaConfig
    model_class = VanillaModel

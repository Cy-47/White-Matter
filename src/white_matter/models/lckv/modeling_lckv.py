"""LCKV decoder: feedforward layers before and after a feedback block.

The feedforward decoder layers compute their own K/V and run once per forward.
The middle block shares a KV source and runs the configured Jacobi passes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from white_matter._typing import eager_loop
from white_matter.blocks.lckv import LCKVBlock
from white_matter.models import _qwen3 as qwen3
from white_matter.modules import KVPool, RotaryEmbedding
from white_matter.modules.precision import cast_residual
from white_matter.modules.routing import FixedSourceMixer

from .._qwen3 import make_feedback_layers
from ..cache import DecoderCache
from ..decoder_utils import prepare_attention_inputs, prepare_decoder_inputs, run_feedforward_layers
from ..modeling_base import DecoderForCausalLM, DecoderModel, DecoderPreTrainedModel
from .configuration_lckv import LCKVConfig


def _reserve_router_draws(num_layers, hidden_size, *, std=None, device=None, dtype=None):
    for _ in range(2):
        weight = torch.empty(num_layers, num_layers * hidden_size, device=device, dtype=dtype)
        if std is None:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            bias = torch.empty(num_layers, device=device, dtype=dtype)
            nn.init.uniform_(bias, -1 / math.sqrt(num_layers * hidden_size), 1 / math.sqrt(num_layers * hidden_size))
        else:
            nn.init.normal_(weight, mean=0.0, std=std)


class LCKVDecoder(DecoderPreTrainedModel):
    """Feedforward pre-layers, the feedback block, and feedforward post-layers."""

    _no_split_modules = ["FeedbackDecoderLayer", "Qwen3DecoderLayer"]

    def __init__(self, config: LCKVConfig) -> None:
        num_pre_layers = config.num_pre_layers
        num_post_layers = config.num_post_layers
        num_passes = config.num_passes
        num_feedback_layers = config.num_hidden_layers - num_pre_layers - num_post_layers
        super().__init__(config)
        self.cache_prefix_slots = (0,) * (num_pre_layers + 1 + num_post_layers)

        # Feedforward layers run once and compute their own K/V.
        self.pre_layers = nn.ModuleList([qwen3.Qwen3DecoderLayer(config, i) for i in range(num_pre_layers)])
        # Self-contained Jacobi block.
        layers = make_feedback_layers(config, num_feedback_layers, strict_causal=True)
        _reserve_router_draws(num_feedback_layers, config.hidden_size)
        pool = KVPool(
            config.hidden_size,
            config.num_key_value_heads,
            config.head_dim,
            num_feedback_layers,
            1,
            rms_norm_eps=config.rms_norm_eps,
            initializer_range=config.initializer_range,
            mixer=FixedSourceMixer(num_feedback_layers),
        )
        self.block = LCKVBlock(layers, pool, RotaryEmbedding(config.head_dim, config.rope_theta), num_passes=num_passes)
        # Feedforward layers after the feedback region.
        self.post_layers = nn.ModuleList(
            [qwen3.Qwen3DecoderLayer(config, num_pre_layers + 1 + i) for i in range(num_post_layers)]
        )
        # Rotary table for feedforward decoder layers (unshifted positions [0..T-1]).
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if module is self.block.kv_pool:
            weight = module.k_proj_weight
            _reserve_router_draws(
                module.num_layers,
                module.hidden_size,
                std=self.config.initializer_range,
                device=weight.device,
                dtype=weight.dtype,
            )

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
        """Run bottom feedforward layers, Jacobi refinement, and top feedforward layers.

        All three stages remain inside the gradient tape. Bottom feedforward
        states do not depend on the refinement passes; top feedforward layers read
        the final middle-block output.
        """
        if checkpoint_chunk_size:
            raise ValueError("checkpoint_chunk_size is supported only for WhiteMatter autoregressive execution")
        if past_key_values is None:
            inputs_embeds, attention_mask, document_ids = prepare_decoder_inputs(
                inputs_embeds, self.config, attention_mask, document_ids
            )
        else:
            inputs_embeds = cast_residual(inputs_embeds, residual_dtype=self.config.residual_dtype)
        ordinary_args = (
            prepare_attention_inputs(
                inputs_embeds,
                document_ids,
                self.rotary_emb,
                self.config._attn_implementation,
                position_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )
            if self.pre_layers or self.post_layers
            else {}
        )
        x = run_feedforward_layers(self.pre_layers, inputs_embeds, **ordinary_args)
        if past_key_values is None:
            x = self.block(x, num_passes=num_passes, num_gradient_passes=num_gradient_passes, document_ids=document_ids)
        elif not past_key_values.get_seq_length() and self.config.prefill_mode == "jacobi":
            x, (keys, values) = self.block(
                x,
                num_passes=num_passes,
                num_gradient_passes=0,
                document_ids=document_ids,
                output_final_state=True,
            )
            past_key_values.update(keys[:, 0], values[:, 0], self.config.num_pre_layers)
        else:
            x = self._forward_cached(x, past_key_values, position_ids, document_ids)
        return run_feedforward_layers(self.post_layers, x, **ordinary_args)

    @eager_loop
    def _forward_cached(self, x, cache, position_ids, document_ids):
        """Run exact autoregressive LCKV and commit one shared KV per token."""
        owner = self.config.num_pre_layers
        storage = cache.layers[owner]
        outputs = []
        for t in range(x.shape[1]):
            token = x[:, t : t + 1]
            lengths = cache.lengths(owner, x.shape[0]) if cache.get_seq_length() + t else None
            mask = None
            if document_ids is not None and lengths is not None:
                previous = cache.document_ids[:, : cache.get_seq_length() + t]
                keep = previous == document_ids[:, t : t + 1]
                slots = torch.arange(storage.keys.shape[-2], device=x.device)
                keep = torch.nn.functional.pad(keep, (0, storage.keys.shape[-2] - keep.shape[-1]))
                keep = keep & (slots[None] < lengths[:, None])
                mask = x.new_zeros(keep[:, None, None].shape).masked_fill(~keep[:, None, None], float("-inf"))
            hidden, key, value = self._token_step(
                token,
                storage.keys if lengths is not None else None,
                storage.values if lengths is not None else None,
                position_ids[:, t : t + 1],
                mask,
                lengths,
                cache.capacity is not None,
            )
            cache.update(key[0], value[0], owner)
            outputs.append(hidden)
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)

    def _token_step(self, token, keys, values, position_ids, mask, lengths, static_cache):
        position = self.block.rotary_emb(token, position_ids)
        hidden, states = token, []
        for layer in self.block.layers:
            states.append(hidden)
            if keys is None:
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            else:
                hidden = layer(
                    hidden,
                    keys,
                    values,
                    position,
                    decode_key_mask=mask,
                    cache_seqlens=lengths,
                    static_cache=static_cache,
                )
        key, value = self.block.kv_pool.project_token(states, position)
        return hidden, key, value


class LCKVPreTrainedModel(DecoderPreTrainedModel):
    config_class = LCKVConfig


class LCKVModel(DecoderModel, LCKVPreTrainedModel):
    config_class = LCKVConfig
    decoder_class = LCKVDecoder


class LCKVForCausalLM(DecoderForCausalLM, LCKVPreTrainedModel):
    config_class = LCKVConfig
    model_class = LCKVModel

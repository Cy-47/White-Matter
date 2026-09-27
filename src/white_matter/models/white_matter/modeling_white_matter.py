"""Hugging Face backbones and causal language models for WhiteMatter."""

from __future__ import annotations

import torch
from torch import nn

from white_matter.blocks import WhiteMatterBlock
from white_matter.blocks._execution import resolve_passes
from white_matter.layers.backends import pack_kv_cache
from white_matter.models import _qwen3 as qwen3
from white_matter.models._qwen3 import make_feedback_layers
from white_matter.modules import KVPool, RotaryEmbedding
from white_matter.modules.documents import feedback_document_mask
from white_matter.modules.precision import cast_residual

from ..cache import DecoderCache
from ..decoder_utils import prepare_attention_inputs, prepare_decoder_inputs, run_feedforward_layers
from ..modeling_base import DecoderForCausalLM, DecoderModel, DecoderPreTrainedModel
from .configuration_white_matter import WhiteMatterConfig


class WhiteMatterDecoder(DecoderPreTrainedModel):
    """Own the feedback block and select the configured execution schedule.

    Decoder-level layers belong here; the block contains the feedback layers.
    """

    _no_split_modules = ["FeedbackDecoderLayer"]
    # Research models may replace the pool without duplicating the decoder,
    # attention schedule, or cache implementation.
    pool_class = KVPool
    block_class = WhiteMatterBlock

    def __init__(self, config: WhiteMatterConfig) -> None:
        super().__init__(config)
        self.cache_prefix_slots = (0,) * config.num_pre_layers + (1,) + (0,) * config.num_post_layers
        self.pre_layers = nn.ModuleList([qwen3.Qwen3DecoderLayer(config, i) for i in range(config.num_pre_layers)])
        # The block owns the feedback layers, KV pool, dummy token, and RoPE.
        depth = config.num_hidden_layers - config.num_pre_layers - config.num_post_layers
        layers = make_feedback_layers(config, depth)
        pool = self.pool_class(
            config.hidden_size,
            config.num_key_value_heads,
            config.head_dim,
            depth,
            config.num_kv_channels,
            rms_norm_eps=config.rms_norm_eps,
            initializer_range=config.initializer_range,
            router_prior=config.router_prior,
            router_layer_stride=config.router_layer_stride,
            router_dynamic=config.router_dynamic,
        )
        self.block = self.block_class(
            layers,
            pool,
            RotaryEmbedding(config.head_dim, config.rope_theta),
            num_passes=config.num_passes,
            checkpoint_jacobi_passes=config.checkpoint_jacobi_passes,
        )
        self.post_layers = nn.ModuleList(
            [qwen3.Qwen3DecoderLayer(config, config.num_pre_layers + 1 + i) for i in range(config.num_post_layers)]
        )
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)
        # Inherited init pass — covers nn.Linear / RMSNorm / MLP weights.
        self.post_init()
        # Re-init the parameters that don't fit `_init_weights`.
        self.block.kv_pool.mixer.reset_parameters()

    def _init_weights(self, module):
        """Initialize the feedback dummy token."""
        super()._init_weights(module)
        if module is self.block:
            # HF guards nn.init calls so loading never overwrites a learned dummy.
            nn.init.zeros_(module.dummy_token)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        num_passes: int | None = None,
        num_gradient_passes: int | None = None,
        cyclic_groups: int | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        checkpoint_chunk_size: int = 0,
        backward_batch_size: int = 0,
        split_state_vjp: bool = True,
        past_key_values: DecoderCache | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cache = past_key_values
        mode = self.config.execution_mode
        if cache is not None:
            mode = "autoregressive" if cache.get_seq_length() else (self.config.prefill_mode or mode)
            if mode in {"cyclic", "jacobi"} and self.config.prefill_mode is None:
                raise ValueError("set prefill_mode explicitly for iterative prefill followed by AR decoding")
        if cache is None and mode == "autoregressive" and not self.training and not torch.is_grad_enabled():
            cache = DecoderCache(self.cache_prefix_slots)
            document_ids, position_ids, attention_mask = cache.prepare(
                inputs_embeds,
                None,
                attention_mask,
                document_ids,
                None,
                position_ids,
            )
        if cache is None or mode in {"cyclic", "jacobi"}:
            num_passes, num_gradient_passes = resolve_passes(
                self.config.num_passes, num_passes, 0 if cache is not None else num_gradient_passes
            )
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
                past_key_values=cache,
                attention_mask=attention_mask,
            )
            if self.pre_layers or self.post_layers
            else {}
        )
        x = run_feedforward_layers(self.pre_layers, inputs_embeds, **ordinary_args)
        if mode == "autoregressive":
            x = (
                self._forward_cached(x, cache, position_ids, document_ids)
                if cache is not None
                else self.block.forward_autoregressive(
                    x,
                    attention_mask=attention_mask,
                    document_ids=document_ids,
                    checkpoint_chunk_size=checkpoint_chunk_size,
                    backward_batch_size=backward_batch_size,
                    split_state_vjp=split_state_vjp,
                )
            )
        elif mode == "jacobi":
            if checkpoint_chunk_size:
                raise ValueError("Jacobi execution does not support autoregressive checkpoint chunks")
            result = self.block.forward_jacobi(
                x,
                num_passes=num_passes,
                num_gradient_passes=num_gradient_passes,
                document_ids=document_ids,
                output_final_state=cache is not None,
            )
            if cache is None:
                x = result
            else:
                x, final_state = result
                cache.update(*(tensor.flatten(1, 2) for tensor in final_state), self.config.num_pre_layers)
        else:
            groups = self.config.cyclic_groups if cyclic_groups is None else cyclic_groups
            if type(groups) is not int or groups < 1:
                raise ValueError("cyclic_groups must be a positive integer")
            state = None
            if (
                cache is not None
                and cache.capacity is not None
                and x.is_cuda
                and not self.training
                and torch.is_autocast_enabled("cuda")
                and torch.get_autocast_dtype("cuda") == torch.bfloat16
                and attention_mask is None
                and document_ids is None
                and x.shape[1] > groups
            ):
                feedback = cache.layers[self.config.num_pre_layers]
                if not feedback.is_initialized:
                    prototype = x.new_empty(
                        (
                            x.shape[0],
                            self.config.num_kv_channels * self.config.num_key_value_heads,
                            0,
                            self.config.head_dim,
                        ),
                        dtype=torch.bfloat16,
                    )
                    feedback.lazy_initialization(prototype, prototype)
                state = tuple(
                    t.unflatten(1, (self.config.num_kv_channels, self.config.num_key_value_heads))
                    for t in (feedback.keys, feedback.values)
                )
            x, final_state = self.block(
                x,
                num_passes=num_passes,
                num_gradient_passes=num_gradient_passes,
                cyclic_groups=groups,
                attention_mask=attention_mask,
                document_ids=document_ids,
                output_final_state=cache is not None,
                kv_cache=state,
            )
            if cache is not None:
                if state is None:
                    cache.update(*(tensor.flatten(1, 2) for tensor in final_state), self.config.num_pre_layers)
                else:
                    feedback.cumulative_length.fill_(x.shape[1] + 1)
        return run_feedforward_layers(self.post_layers, x, **ordinary_args)

    def _forward_cached(self, x, cache, position_ids, document_ids):
        """AR prefill and decoding share one loop over committed KV storage."""
        seen, index = cache.get_seq_length(), self.config.num_pre_layers
        feedback = cache.layers[index]
        if not feedback.is_initialized or not seen:
            dummy = self.block._project_dummy(x.shape[0])
            cache.update(*(tensor.transpose(0, 1).flatten(1, 2) for tensor in dummy), index)
        outputs = []
        for t in range(x.shape[1]):
            state = tuple(
                tensor.unflatten(1, (self.config.num_kv_channels, self.config.num_key_value_heads))
                for tensor in (feedback.keys, feedback.values)
            )
            mask, lengths = None, cache.lengths(index, x.shape[0])
            if document_ids is not None:
                keep = feedback_document_mask(cache.document_ids[:, : seen + t], document_ids[:, t : t + 1])
                if x.is_cuda and self.config._attn_implementation == "flash_attention_2":
                    key, value, lengths = pack_kv_cache(*state, keep)
                    state = key, value
                else:
                    mask = x.new_zeros(keep.shape).masked_fill(~keep, torch.finfo(x.dtype).min)[:, None, None]
                    lengths = None
            output, key, value = self.block._run_token_layers(
                x[:, t : t + 1],
                state,
                self.block.rotary_emb(x, position_ids[:, t : t + 1] + 1),
                mask,
                cache_seqlens=lengths,
            )
            cache.update(key.transpose(0, 1).flatten(1, 2), value.transpose(0, 1).flatten(1, 2), index)
            outputs.append(output)
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)


class WhiteMatterPreTrainedModel(DecoderPreTrainedModel):
    config_class = WhiteMatterConfig


class WhiteMatterModel(DecoderModel, WhiteMatterPreTrainedModel):
    config_class = WhiteMatterConfig
    decoder_class = WhiteMatterDecoder


class WhiteMatterForCausalLM(DecoderForCausalLM, WhiteMatterPreTrainedModel):
    config_class = WhiteMatterConfig
    model_class = WhiteMatterModel

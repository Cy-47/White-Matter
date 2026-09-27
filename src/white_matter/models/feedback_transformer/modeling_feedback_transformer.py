"""Exact sequential Feedback Transformer prefill and cached decoding."""

from __future__ import annotations

import torch
from torch import nn

from white_matter.blocks.feedback_transformer import FeedbackMemory, FeedbackTransformerBlock
from white_matter.layers.backends import pack_kv_cache
from white_matter.models._qwen3 import make_feedback_layers
from white_matter.modules import RotaryEmbedding
from white_matter.modules.precision import cast_residual

from ..cache import DecoderCache
from ..decoder_utils import prepare_decoder_inputs
from ..modeling_base import DecoderForCausalLM, DecoderModel, DecoderPreTrainedModel
from .configuration_feedback_transformer import FeedbackTransformerConfig


class FeedbackTransformerDecoder(DecoderPreTrainedModel):
    _no_split_modules = ["FeedbackDecoderLayer"]
    supports_last_token_only = True

    def __init__(self, config: FeedbackTransformerConfig):
        super().__init__(config)
        self.cache_prefix_slots = (0,)
        layers = nn.ModuleList(make_feedback_layers(config, config.num_hidden_layers, strict_causal=True))
        memory = FeedbackMemory(
            config.hidden_size,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.rms_norm_eps,
        )
        self.block = FeedbackTransformerBlock(layers, memory, RotaryEmbedding(config.head_dim, config.rope_theta))
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
        last_token_only: bool = False,
    ) -> torch.Tensor:
        if num_passes not in {None, 1} or num_gradient_passes not in {None, 1}:
            raise ValueError("Feedback Transformer uses one exact autoregressive sweep")
        if checkpoint_chunk_size:
            raise ValueError("Feedback Transformer checkpointing is not implemented")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise NotImplementedError("Feedback Transformer currently requires unpadded inputs")
        if past_key_values is None:
            if last_token_only:
                raise ValueError("last_token_only requires cached inference")
            x, attention_mask, document_ids = prepare_decoder_inputs(
                inputs_embeds,
                self.config,
                attention_mask,
                document_ids,
            )
            return self.block.forward_reference(x, document_ids=document_ids, attention_mask=attention_mask)[0]
        x = cast_residual(inputs_embeds, residual_dtype=self.config.residual_dtype)
        return self._forward_cached(x, past_key_values, position_ids, document_ids, last_token_only)

    @torch.compiler.disable
    def _forward_cached(self, x, cache, position_ids, document_ids, last_token_only):
        """Write one K/V slot after each complete layer sweep."""
        storage = cache.layers[0]
        seen = cache.get_seq_length()
        outputs = [] if not last_token_only else None
        last_output = None
        for t in range(x.shape[1]):
            keys = values = lengths = mask = None
            if seen + t:
                keys, values = storage.keys, storage.values
                lengths = cache.lengths(0, x.shape[0])
                if document_ids is not None:
                    prior = cache.document_ids[:, : seen + t]
                    keep = prior == document_ids[:, t : t + 1]
                    if x.is_cuda and self.config._attn_implementation == "flash_attention_2":
                        keys, values, lengths = pack_kv_cache(keys, values, keep)
                    else:
                        slots = torch.arange(keys.shape[-2], device=x.device)
                        keep = torch.nn.functional.pad(keep, (0, keys.shape[-2] - keep.shape[-1]))
                        keep = keep & (slots[None] < lengths[:, None])
                        mask = x.new_zeros((x.shape[0], 1, 1, keys.shape[-2])).masked_fill(
                            ~keep[:, None, None], float("-inf")
                        )
                        lengths = None
            out, key, value = self.block.token_step(
                x[:, t : t + 1],
                keys,
                values,
                position_ids[:, t : t + 1],
                key_mask=mask,
                cache_lengths=lengths,
            )
            cache.update(key, value, 0)
            if outputs is not None:
                outputs.append(out)
            last_output = out
        if outputs is None:
            return last_output
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)


class FeedbackTransformerPreTrainedModel(DecoderPreTrainedModel):
    config_class = FeedbackTransformerConfig


class FeedbackTransformerModel(DecoderModel, FeedbackTransformerPreTrainedModel):
    config_class = FeedbackTransformerConfig
    decoder_class = FeedbackTransformerDecoder


class FeedbackTransformerForCausalLM(DecoderForCausalLM, FeedbackTransformerPreTrainedModel):
    config_class = FeedbackTransformerConfig
    model_class = FeedbackTransformerModel

    def compile(self, *args, **kwargs):
        # The cached sweep itself stays in Python so prefill traces one token
        # step, not a graph with a copy of every layer for every prompt token.
        block = self.model.decoder.block
        if not getattr(block, "_token_step_compiled", False):
            token_options = dict(kwargs)
            token_options["fullgraph"] = True
            block.token_step = torch.compile(block.token_step, *args, **token_options)
            block._token_step_compiled = True
        return super().compile(*args, **kwargs)

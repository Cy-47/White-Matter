"""Hugging Face backbones and causal language models for WhiteMatter."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from white_matter.modules.documents import document_ids_from_eos
from white_matter.modules.precision import model_autocast_context

from .cache import DecoderCache
from .configuration_base import DecoderConfig


class DecoderPreTrainedModel(PreTrainedModel):
    # HF copies this onto shared subclasses; concrete families must redeclare it.
    config_class = DecoderConfig
    base_model_prefix = "model"
    _no_split_modules = [
        "FeedbackDecoderLayer",
        "Qwen3DecoderLayer",
        "FusedKVSourceLayer",
        "FusedKVReconstructionLayer",
    ]
    supports_gradient_checkpointing = False
    _supports_sdpa = True
    _supports_flash_attn = True
    # HF's BF16 loader rounds matrix operands once, retaining normalization and
    # feedback gains at their original precision (also for mixed-dtype exports).
    _keep_in_fp32_modules_strict = ["norm", "pre_mix", "post_mix", "dummy_token", "fusions"]

    def _init_weights(self, module):
        super()._init_weights(module)
        # Nonpersistent buffers must be recreated after HF's meta loading.
        if hasattr(module, "reset_runtime_buffers"):
            module.reset_runtime_buffers()


class DecoderModel(DecoderPreTrainedModel):
    """Token embeddings, architecture-specific decoder, and final normalization."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__(config)
        decoder = self.decoder_class(config)
        parameter = next(decoder.parameters())
        # Allocate embeddings once, then initialize in the reference RNG order.
        # The causal LM ties its output head to this same weight tensor.
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            dtype=parameter.dtype,
            device="meta",
        ).to_empty(device=parameter.device)
        self.decoder = decoder
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(
            device=parameter.device, dtype=parameter.dtype
        )
        with torch.no_grad():
            self.embed_tokens.weight.normal_(mean=0.0, std=0.02)
            if self.embed_tokens.padding_idx is not None:
                self.embed_tokens.weight[self.embed_tokens.padding_idx].zero_()
            self.norm.weight.fill_(1.0)
        for module in (self.embed_tokens, self.norm):
            module._is_hf_initialized = True
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        num_passes: int | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        last_token_only: bool = False,
        return_dict: bool | None = None,
    ) -> BaseModelOutputWithPast | tuple:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        use_cache = (past_key_values is not None or self.config.use_cache) if use_cache is None else use_cache
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if use_cache:
            if not hasattr(self.decoder, "cache_prefix_slots"):
                raise NotImplementedError("this model family does not support cached generation")
            if torch.is_grad_enabled():
                raise ValueError("persistent caches are inference-only; use torch.no_grad() or inference_mode()")
            if past_key_values is None:
                past_key_values = DecoderCache(self.decoder.cache_prefix_slots)
            elif (
                not isinstance(past_key_values, DecoderCache)
                or past_key_values.prefix_slots != self.decoder.cache_prefix_slots
            ):
                raise TypeError("past_key_values must be a DecoderCache with this decoder's KV owners")
        if (
            position_ids is not None
            and use_cache
            and not past_key_values.get_seq_length()
            and getattr(self.config, "prefill_mode", None) in {"cyclic", "jacobi"}
        ):
            raise ValueError(
                "explicit position_ids require autoregressive prefill; iterative suffix prefill is not supported"
            )
        if position_ids is not None and not use_cache:
            raise NotImplementedError("explicit position_ids currently require cached inference")
        if self.config.output_hidden_states if output_hidden_states is None else output_hidden_states:
            raise NotImplementedError("per-layer hidden states are not exposed for iterative decoders")
        if self.config.output_attentions if output_attentions is None else output_attentions:
            raise NotImplementedError("attention weights are not exposed by the production attention kernels")
        if (
            not use_cache
            and input_ids is not None
            and document_ids is None
            and self.config.document_separator_token_id is not None
        ):
            document_ids = document_ids_from_eos(input_ids, self.config.document_separator_token_id)
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        with model_autocast_context(device):
            hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
            if hidden.ndim != 3 or hidden.shape[1] < 1:
                raise ValueError("inputs must have a nonempty batch/sequence/hidden layout")
            if use_cache:
                documents, positions, valid = past_key_values.prepare(
                    hidden,
                    input_ids,
                    attention_mask,
                    document_ids,
                    self.config.document_separator_token_id,
                    position_ids,
                )
                decoder_options = {"last_token_only": True} if last_token_only else {}
                hidden = self.decoder(
                    hidden,
                    attention_mask=valid,
                    document_ids=documents,
                    num_passes=num_passes,
                    past_key_values=past_key_values,
                    position_ids=positions,
                    **decoder_options,
                )
                past_key_values.advance(positions, valid)
            else:
                hidden = self.decoder(
                    hidden, attention_mask=attention_mask, document_ids=document_ids, num_passes=num_passes
                )
            hidden = self.norm(hidden.to(dtype=self.norm.weight.dtype))
        output = BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=past_key_values)
        return output if (self.config.return_dict if return_dict is None else return_dict) else output.to_tuple()


class DecoderForCausalLM(DecoderPreTrainedModel, GenerationMixin):
    """A standalone causal language model with tied input/output embeddings."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _keys_to_ignore_on_load_missing = ["lm_head.weight"]

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__(config)
        self.model = self.model_class(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device="meta")
        self.tie_weights()
        self.lm_head._is_hf_initialized = True
        self.post_init()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace embeddings while keeping the output head tied."""
        super().set_input_embeddings(value)
        self.tie_weights()

    def tie_weights(self, **_: Any) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        num_passes: int | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_dict: bool | None = None,
    ) -> CausalLMOutputWithPast | tuple:
        if isinstance(logits_to_keep, int) and logits_to_keep < 0:
            raise ValueError("logits_to_keep must be nonnegative")
        if labels is not None and (not isinstance(logits_to_keep, int) or logits_to_keep != 0):
            raise ValueError("labels require logits_to_keep=0 for the full next-token loss")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            document_ids=document_ids,
            num_passes=num_passes,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            last_token_only=(
                isinstance(logits_to_keep, int)
                and logits_to_keep == 1
                and (past_key_values is not None or use_cache is True)
                and getattr(self.model.decoder, "supports_last_token_only", False)
            ),
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        positions = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        with model_autocast_context(hidden.device):
            logits = self.lm_head(hidden[:, positions])
        loss = None
        if labels is not None:
            if labels.shape != hidden.shape[:2]:
                raise ValueError("labels must match the input batch and sequence dimensions")
            loss = nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].to(logits.device).reshape(-1)
            )
        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)
        return output if (self.config.return_dict if return_dict is None else return_dict) else output.to_tuple()

    def allocate_inference_cache(self, max_cache_len: int | None = None) -> DecoderCache:
        """Allocate native HF storage per KV owner; dimensions are inferred at prefill."""
        if not hasattr(self.model.decoder, "cache_prefix_slots"):
            raise NotImplementedError("this model family does not support cached generation")
        return DecoderCache(self.model.decoder.cache_prefix_slots, max_cache_len)

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    ):
        if not generation_config.use_cache:
            return
        if generation_config.cache_implementation not in {None, "dynamic", "static"}:
            raise NotImplementedError("supported cache implementations: dynamic, static")
        if model_kwargs.get("past_key_values") is None:
            capacity = max_cache_length if generation_config.cache_implementation == "static" else None
            model_kwargs["past_key_values"] = self.allocate_inference_cache(capacity)

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        # Cache/document positions replace HF's global offsets; explicit IDs still pass through.
        return None

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_passes: int | None = None,
        use_cache: bool = False,
        past_key_values: Any = None,
        next_sequence_length: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if kwargs.get("inputs_embeds") is not None or kwargs.get("document_ids") is not None:
            raise NotImplementedError("generation requires input_ids; pass document_ids through forward instead")
        kwargs.setdefault("logits_to_keep", 1)
        return super().prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            next_sequence_length=next_sequence_length if past_key_values is not None else None,
            num_passes=num_passes,
            use_cache=use_cache,
            **kwargs,
        )

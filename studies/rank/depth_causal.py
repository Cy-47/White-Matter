"""Study-local depth-causal full-rank ablation from paper Figure 7b."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from white_matter.blocks import WhiteMatterBlock
from white_matter.blocks._execution.metadata import prepare_feedback_metadata
from white_matter.models.decoder_utils import prepare_attention_inputs, prepare_decoder_inputs, run_feedforward_layers
from white_matter.models.white_matter.configuration_white_matter import WhiteMatterConfig
from white_matter.models.white_matter.modeling_white_matter import (
    WhiteMatterDecoder,
    WhiteMatterForCausalLM,
    WhiteMatterModel,
)
from white_matter.modules.kv_pool import KVPool
from white_matter.modules.rotary import rotate_half


class DepthCausalKVPool(KVPool):
    """Project basis ell as soon as source layers 0 through ell exist."""

    def premix_source(self, hidden: torch.Tensor, dummy: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        full = torch.cat((dummy.expand(hidden.shape[0], 1, -1).to(hidden.dtype), hidden), dim=1)
        normed = F.rms_norm(full, (self.hidden_size,), None, self.rms_norm_eps)
        return (
            normed * self.pre_mix_k_weight[layer].to(normed.dtype),
            normed * self.pre_mix_v_weight[layer].to(normed.dtype),
        )

    def _route_one(self, stacked: torch.Tensor, layer: int, router) -> torch.Tensor:
        if stacked.shape[2] != layer + 1:
            raise ValueError("depth-causal routing requires exactly the available source prefix")
        # At each depth the stride is anchored at that depth. Left zero padding
        # keeps the router's input width equal to the ordinary full-rank model.
        indices = torch.arange(layer % router.layer_stride, layer + 1, router.layer_stride, device=stacked.device)
        selected = stacked.index_select(2, indices)
        missing = router.num_sources - selected.shape[2]
        if missing:
            selected = torch.cat((selected.new_zeros(*selected.shape[:2], missing, self.hidden_size), selected), dim=2)
        context = selected.reshape(*selected.shape[:2], router.num_sources * self.hidden_size)
        start = layer * self.num_layers
        logits = F.linear(
            context,
            router.linear.weight[start : start + self.num_layers],
            router.linear.bias[start : start + self.num_layers],
        )
        return logits[..., : layer + 1].to(stacked.dtype)

    def project_causal_basis(
        self,
        sources_k: list[torch.Tensor],
        sources_v: list[torch.Tensor],
        layer: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(sources_k) != layer + 1 or len(sources_v) != layer + 1:
            raise ValueError("depth-causal basis requires source layers 0 through its layer")
        stacked_k, stacked_v = torch.stack(sources_k, dim=2), torch.stack(sources_v, dim=2)
        router_k, router_v = self.mixer.k_router, self.mixer.v_router
        weights_k = self._route_one(stacked_k, layer, router_k)
        weights_v = self._route_one(stacked_v, layer, router_v)
        mixed_k = torch.einsum("btld,btl->btd", stacked_k, weights_k)
        mixed_v = torch.einsum("btld,btl->btd", stacked_v, weights_v)
        mixed_k = F.rms_norm(mixed_k, (self.hidden_size,), None, self.mix_norm_eps)
        mixed_v = F.rms_norm(mixed_v, (self.hidden_size,), None, self.mix_norm_eps)
        mixed_k = mixed_k * self.post_mix["k_gain"][layer].to(mixed_k.dtype)
        mixed_v = mixed_v * self.post_mix["v_gain"][layer].to(mixed_v.dtype)
        batch, length = mixed_k.shape[:2]
        shape = (batch, length, self.num_key_value_heads, self.head_dim)
        key = F.linear(mixed_k, self.k_proj_weight[layer]).view(shape).transpose(1, 2)
        value = F.linear(mixed_v, self.v_proj_weight[layer]).view(shape).transpose(1, 2)
        key = F.rms_norm(key, (self.head_dim,), None, self.rms_norm_eps)
        key = key * self.k_norm_weight[layer].to(key.dtype)
        cos, sin = position_embeddings
        cos, sin = cos[:, None].to(key.dtype), sin[:, None].to(key.dtype)
        key = key * cos + rotate_half(key) * sin
        return key.to(value.dtype), value


class DepthCausalBlock(WhiteMatterBlock):
    def forward_depth_causal(self, x: torch.Tensor, *, document_ids: torch.Tensor | None = None) -> torch.Tensor:
        q_pos, k_pos = self._prepare_rope(x, document_ids)
        metadata = None
        if document_ids is not None:
            slots = torch.arange(x.shape[1], device=x.device)
            metadata = prepare_feedback_metadata(document_ids, [slots], x.shape[1])[0]
        sources_k: list[torch.Tensor] = []
        sources_v: list[torch.Tensor] = []
        hidden = x
        for layer_idx, layer in enumerate(self.layers):
            source_k, source_v = self.kv_pool.premix_source(hidden, self.dummy_token, layer_idx)
            sources_k.append(source_k)
            sources_v.append(source_v)
            key, value = self.kv_pool.project_causal_basis(sources_k, sources_v, layer_idx, k_pos)
            hidden = layer(
                hidden,
                key[..., :-1, :].contiguous(),
                value[..., :-1, :].contiguous(),
                q_pos,
                document_ids=document_ids,
                metadata=metadata,
            )
        return hidden


class DepthCausalConfig(WhiteMatterConfig):
    model_type = "white_matter_depth_causal"

    def __init__(self, **kwargs) -> None:
        mode = kwargs.pop("execution_mode", "depth_causal")
        if mode != "depth_causal":
            raise ValueError("depth-causal study requires depth_causal execution")
        kwargs.setdefault("include_top_output", False)
        kwargs.setdefault("use_dummy_token", True)
        if kwargs["include_top_output"]:
            raise ValueError("depth-causal study requires include_top_output=False")
        kwargs.setdefault("num_passes", 1)
        kwargs.setdefault("router_prior", "identity:0.25")
        kwargs.setdefault("num_kv_channels", kwargs.get("num_hidden_layers", 16))
        super().__init__(execution_mode="cyclic", **kwargs)
        if self.num_passes != 1 or self.num_kv_channels != self.num_hidden_layers:
            raise ValueError("depth-causal study requires one pass and one channel per layer")
        if self.router_prior != "identity:0.25" or not self.router_dynamic:
            raise ValueError("depth-causal study requires a dynamic identity-prior router")
        if self.num_pre_layers or self.num_post_layers:
            raise ValueError("depth-causal study requires all layers in the feedback block")
        self.execution_mode = "depth_causal"


class DepthCausalDecoder(WhiteMatterDecoder):
    pool_class = DepthCausalKVPool
    block_class = DepthCausalBlock

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        num_passes: int | None = None,
        num_gradient_passes: int | None = None,
        document_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        checkpoint_chunk_size: int = 0,
        past_key_values=None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if past_key_values is not None or position_ids is not None:
            raise NotImplementedError("depth-causal study does not implement cached generation")
        if num_passes not in {None, 1} or num_gradient_passes not in {None, 1}:
            raise ValueError("depth-causal execution is one forward pass")
        if checkpoint_chunk_size or kwargs:
            raise ValueError("depth-causal study does not use iterative checkpoint options")
        x, attention_mask, document_ids = prepare_decoder_inputs(
            inputs_embeds,
            self.config,
            attention_mask,
            document_ids,
        )
        ordinary_args = (
            prepare_attention_inputs(
                x,
                document_ids,
                self.rotary_emb,
                self.config._attn_implementation,
            )
            if self.pre_layers or self.post_layers
            else {}
        )
        x = run_feedforward_layers(self.pre_layers, x, **ordinary_args)
        x = self.block.forward_depth_causal(x, document_ids=document_ids)
        return run_feedforward_layers(self.post_layers, x, **ordinary_args)


class DepthCausalModel(WhiteMatterModel):
    config_class = DepthCausalConfig
    decoder_class = DepthCausalDecoder


class DepthCausalForCausalLM(WhiteMatterForCausalLM):
    config_class = DepthCausalConfig
    model_class = DepthCausalModel


_registered = False


def register_model() -> None:
    global _registered
    if _registered:
        return
    AutoConfig.register(DepthCausalConfig.model_type, DepthCausalConfig)
    AutoModel.register(DepthCausalConfig, DepthCausalModel)
    AutoModelForCausalLM.register(DepthCausalConfig, DepthCausalForCausalLM)
    _registered = True

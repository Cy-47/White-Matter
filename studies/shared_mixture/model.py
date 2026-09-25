"""Study-local full-cache WhiteMatter with one source mixture per token.

All feedback layers read distinct KV pairs. The sixteen pairs at paper scale
come from one routed, RMS-normalized hidden vector, with independent K/V
projection matrices and post-mix gains for each pair.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from white_matter.models.white_matter.configuration_white_matter import WhiteMatterConfig
from white_matter.models.white_matter.modeling_white_matter import (
    WhiteMatterDecoder,
    WhiteMatterForCausalLM,
    WhiteMatterModel,
)
from white_matter.modules.kv_pool import KVPool
from white_matter.modules.routing import Router


class SharedMixtureMixer(nn.Module):
    """One content-dependent signed mixture of every source layer."""

    def __init__(self, num_layers: int, hidden_size: int, *, layer_stride: int, router_prior: str) -> None:
        super().__init__()
        self.router = Router(
            num_layers, hidden_size, 1,
            router_prior=router_prior, layer_stride=layer_stride,
        )

    def reset_parameters(self) -> None:
        self.router.reset_parameters()

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        weights = self.router(stacked)
        return torch.einsum("btld,btkl->btkd", stacked, weights)


class SharedMixtureKVPool(KVPool):
    """One pre-projection source vector, projected into k distinct KV pairs."""

    def __init__(
        self,
        hidden_size: int,
        num_key_value_heads: int,
        head_dim: int,
        num_layers: int,
        num_kv_channels: int,
        *,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        router_prior: str = "cyclic:0.25",
        router_layer_stride: int = 1,
        router_dynamic: bool = True,
    ) -> None:
        if router_dynamic is not True:
            raise ValueError("shared-mixture study requires a dynamic router")
        mixer = SharedMixtureMixer(
            num_layers, hidden_size, layer_stride=router_layer_stride, router_prior=router_prior,
        )
        super().__init__(
            hidden_size, num_key_value_heads, head_dim, num_layers, num_kv_channels,
            rms_norm_eps=rms_norm_eps, initializer_range=initializer_range,
            router_prior=router_prior, router_layer_stride=router_layer_stride,
            mixer=mixer,
        )
        # The source entering K and V is literally the same tensor. Keeping the
        # ordinary pool's two pre-mix parameters would make two mixtures or
        # leave a trainable parameter unused in production training.
        del self.pre_mix_k_weight
        del self.pre_mix_v_weight
        self.pre_mix_weight = nn.Parameter(torch.ones(num_layers, hidden_size))

    def _project(self, stacked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence_length = stacked.shape[:2]
        normed = F.rms_norm(stacked, (self.hidden_size,), None, self.rms_norm_eps)
        source = normed * self.pre_mix_weight[None, None].to(normed.dtype)
        mixed = self.mixer(source)
        mixed = F.rms_norm(mixed, (self.hidden_size,), None, self.mix_norm_eps)
        # The shared mixture is expanded only when independent channel gains
        # and projections are applied; routing itself computes one vector.
        keys_in = mixed * self.post_mix["k_gain"][None, None].to(mixed.dtype)
        values_in = mixed * self.post_mix["v_gain"][None, None].to(mixed.dtype)
        shape = (self.num_kv_channels, batch * sequence_length, self.hidden_size)
        keys = torch.bmm(
            keys_in.permute(2, 0, 1, 3).reshape(shape),
            self.k_proj_weight.transpose(1, 2),
        )
        values = torch.bmm(
            values_in.permute(2, 0, 1, 3).reshape(shape),
            self.v_proj_weight.transpose(1, 2),
        )
        channel_shape = (self.num_kv_channels, batch, sequence_length, self.num_key_value_heads, self.head_dim)
        return (
            keys.view(channel_shape).permute(1, 0, 3, 2, 4).contiguous(),
            values.view(channel_shape).permute(1, 0, 3, 2, 4).contiguous(),
        )


class SharedMixtureConfig(WhiteMatterConfig):
    model_type = "white_matter_shared_mixture"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault(
            "num_kv_channels",
            kwargs.get("num_hidden_layers", 16) - kwargs.get("num_pre_layers", 0) - kwargs.get("num_post_layers", 0),
        )
        super().__init__(**kwargs)
        depth = self.num_hidden_layers - self.num_pre_layers - self.num_post_layers
        if self.num_kv_channels != depth:
            raise ValueError("shared-mixture study requires one KV pair per feedback layer")
        if self.router_prior != "cyclic:0.25":
            raise ValueError("shared-mixture study uses the equal-source cyclic:0.25 prior")
        if self.router_dynamic is not True:
            raise ValueError("shared-mixture study requires a dynamic router")


class SharedMixtureDecoder(WhiteMatterDecoder):
    pool_class = SharedMixtureKVPool


class SharedMixtureModel(WhiteMatterModel):
    config_class = SharedMixtureConfig
    decoder_class = SharedMixtureDecoder


class SharedMixtureForCausalLM(WhiteMatterForCausalLM):
    config_class = SharedMixtureConfig
    model_class = SharedMixtureModel


_registered = False


def register_model() -> None:
    """Register the study family before loading recipes or HF checkpoints."""
    global _registered
    if _registered:
        return
    AutoConfig.register(SharedMixtureConfig.model_type, SharedMixtureConfig)
    AutoModel.register(SharedMixtureConfig, SharedMixtureModel)
    AutoModelForCausalLM.register(SharedMixtureConfig, SharedMixtureForCausalLM)
    _registered = True

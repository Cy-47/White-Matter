"""Cross-layer KV construction."""

from collections.abc import Callable
from typing import Protocol, cast

import torch
import torch.nn.functional as F
from torch import nn

from .rotary import rotate_half
from .routing import FixedSourceMixer, Router, _RoutedMixer
from .checkpointing import checkpoint_pointwise


def _premix(
    normed: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ((normed * k_weight).bfloat16(), (normed * v_weight).bfloat16())


def _projection_input(normed: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    mixed = (normed * gain.to(normed.dtype)).bfloat16()
    return mixed.permute(2, 0, 1, 3).flatten(1, 2)


# Compile only inference pointwise work, leaving RMS reductions and GEMMs intact.
_premix_inference = torch.compile(_premix, fullgraph=True, dynamic=True)
_projection_input_inference = torch.compile(_projection_input, fullgraph=True, dynamic=True)


class KVMixer(Protocol):
    """Token-local layer mixing; token chunks must be independently projectable."""

    __call__: Callable[..., tuple[torch.Tensor, torch.Tensor]]

    def forward(self, stacked_K: torch.Tensor, stacked_V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class KVPool(nn.Module):
    """Shared rank-k K/V channels with the routing used by all paper recipes."""

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
        mixer: KVMixer | None = None,
    ) -> None:
        super().__init__()
        if not 1 <= num_kv_channels <= num_layers:
            raise ValueError(f"num_kv_channels must be in [1, {num_layers}], got {num_kv_channels}")
        self.num_layers = num_layers
        self.num_kv_channels = num_kv_channels
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.mix_norm_eps = 1.0e-4
        self.rms_norm_eps = float(rms_norm_eps)

        self.mixer = (
            mixer
            if mixer is not None
            else _RoutedMixer(
                *(
                    Router(
                        num_layers,
                        self.hidden_size,
                        num_kv_channels,
                        router_prior=router_prior,
                        layer_stride=router_layer_stride,
                        dynamic=router_dynamic,
                    )
                    for _ in range(2)
                )
            )
        )
        self.post_mix = nn.ParameterDict({
            name: nn.Parameter(torch.ones(num_kv_channels, hidden_size)) for name in ("k_gain", "v_gain")
        })

        self.pre_mix_k_weight = nn.Parameter(torch.ones(num_layers, self.hidden_size))
        self.pre_mix_v_weight = nn.Parameter(torch.ones(num_layers, self.hidden_size))

        kv_dim = self.num_key_value_heads * self.head_dim
        init_std = float(initializer_range)
        self.k_proj_weight = nn.Parameter(torch.empty(num_kv_channels, kv_dim, self.hidden_size))
        self.v_proj_weight = nn.Parameter(torch.empty(num_kv_channels, kv_dim, self.hidden_size))
        nn.init.normal_(self.k_proj_weight, mean=0.0, std=init_std)
        nn.init.normal_(self.v_proj_weight, mean=0.0, std=init_std)
        self.k_norm_weight = nn.Parameter(torch.ones(num_kv_channels, self.head_dim))

    def _project(self, stacked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Mix (B,T,L,D) source-layer inputs into (B,T,k,D) channels before projection.
        fixed_source = (not self.training and not torch.is_grad_enabled()
                        and type(self.mixer) is FixedSourceMixer)
        if fixed_source:
            # LCKV's other source coefficients are exactly zero. Slice before
            # normalization and K/V premixing to avoid L full-size temporaries.
            stacked = stacked[:, :, -1:, :]
        normed = F.rms_norm(stacked, (self.hidden_size,), None, self.rms_norm_eps)
        shape = (1, 1, stacked.shape[2], self.hidden_size)
        k_weight = self.pre_mix_k_weight[-1:] if fixed_source else self.pre_mix_k_weight
        v_weight = self.pre_mix_v_weight[-1:] if fixed_source else self.pre_mix_v_weight
        fused = (not self.training and not torch.is_grad_enabled() and stacked.is_cuda
                 and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
                 and type(self.mixer) is _RoutedMixer)
        if fused:
            # Cast these small weights before fusion: Inductor otherwise removes
            # their BF16 round-trip when multiplying FP32 normalized activations.
            stacked_K, stacked_V = _premix_inference(
                normed, k_weight.view(*shape).to(stacked.dtype),
                v_weight.view(*shape).to(stacked.dtype),
            )
        else:
            stacked_K = normed * k_weight.view(*shape).to(stacked.dtype)
            stacked_V = normed * v_weight.view(*shape).to(stacked.dtype)
        del normed
        if fixed_source:
            logits = cast(FixedSourceMixer, self.mixer).logits[:, -1:]
            h_K = torch.einsum("btld,kl->btkd", stacked_K, logits.to(stacked_K.dtype))
            h_V = torch.einsum("btld,kl->btkd", stacked_V, logits.to(stacked_V.dtype))
        else:
            h_K, h_V = self.mixer(stacked_K, stacked_V)
        del stacked_K, stacked_V
        h_K = F.rms_norm(h_K, (self.hidden_size,), None, self.mix_norm_eps)
        h_V = F.rms_norm(h_V, (self.hidden_size,), None, self.mix_norm_eps)
        batch, sequence_length = h_K.shape[:2]
        if fused:
            h_K = _projection_input_inference(h_K, self.post_mix["k_gain"][None, None])
            h_V = _projection_input_inference(h_V, self.post_mix["v_gain"][None, None])
        else:
            h_K = h_K * self.post_mix["k_gain"][None, None].to(h_K.dtype)
            h_V = h_V * self.post_mix["v_gain"][None, None].to(h_V.dtype)
        # Each channel has its own projection, shared across batch and token positions.
        K_by_channel = torch.bmm(
            h_K if fused else h_K.permute(2, 0, 1, 3).reshape(self.num_kv_channels, batch * sequence_length, self.hidden_size),
            self.k_proj_weight.transpose(1, 2),
        )
        V_by_channel = torch.bmm(
            h_V if fused else h_V.permute(2, 0, 1, 3).reshape(self.num_kv_channels, batch * sequence_length, self.hidden_size),
            self.v_proj_weight.transpose(1, 2),
        )
        channel_shape = (self.num_kv_channels, batch, sequence_length, self.num_key_value_heads, self.head_dim)
        return (
            K_by_channel.view(channel_shape).permute(1, 0, 3, 2, 4).contiguous(),
            V_by_channel.view(channel_shape).permute(1, 0, 3, 2, 4).contiguous(),
        )

    def _apply_k_norm_rope(
        self,
        K_channels: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        normed = F.rms_norm(K_channels, (self.head_dim,), None, self.rms_norm_eps)
        weights = self.k_norm_weight.view(1, self.num_kv_channels, 1, 1, self.head_dim).to(K_channels.dtype)
        K_channels = normed * weights
        cos, sin = position_embeddings
        cos = cos[:, None, None, :, :].to(dtype=K_channels.dtype)
        sin = sin[:, None, None, :, :].to(dtype=K_channels.dtype)
        return (K_channels * cos) + (rotate_half(K_channels) * sin)

    def project_sequence(
        self,
        stacked: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        dummy_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Routing/projection is token-local. Bound inference temporaries before
        # materializing the layer stack or dummy row; preserve training GEMM shapes.
        chunk_size = 512
        if not self.training and not torch.is_grad_enabled() and stacked.shape[1] > chunk_size:
            output = None
            offset = int(dummy_token is not None)
            for start in range(0, stacked.shape[1], chunk_size):
                end = min(start + chunk_size, stacked.shape[1])
                rope_start = start + offset if start else 0
                rope = tuple(t[:, rope_start : end + offset] for t in position_embeddings)
                chunk = self.project_sequence(
                    stacked[:, start:end], (rope[0], rope[1]), dummy_token=dummy_token if start == 0 else None,
                )
                if output is None:
                    shape = (*chunk[0].shape[:3], stacked.shape[1] + offset, chunk[0].shape[-1])
                    output = chunk[0].new_empty(shape), chunk[1].new_empty(shape)
                for target, source in zip(output, chunk, strict=True):
                    target[..., rope_start:end + offset, :].copy_(source)
                del chunk, source
            assert output is not None
            return output
        if dummy_token is not None:
            dummy = dummy_token.expand(stacked.shape[0], 1, -1)
            dummy = dummy.to(stacked.dtype).unsqueeze(2).expand(-1, -1, self.num_layers, -1)
            stacked = torch.cat((dummy, stacked), dim=1)
        if torch.is_grad_enabled() and stacked.requires_grad:
            K_channels, V_channels = checkpoint_pointwise(self._project, stacked)
        else:
            K_channels, V_channels = self._project(stacked)
        # Attention consumes BF16 keys. Cast once before storage instead of
        # retaining promoted keys and casting them again at every layer read.
        K_channels = self._apply_k_norm_rope(K_channels, position_embeddings).to(V_channels.dtype)
        return K_channels, V_channels

    def project_token(
        self,
        layer_hidden_inputs: list[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        K_channels, V_channels = self._project(torch.stack(layer_hidden_inputs, dim=2))
        K_channels = self._apply_k_norm_rope(K_channels, position_embeddings).to(V_channels.dtype)
        return K_channels.transpose(0, 1).contiguous(), V_channels.transpose(0, 1).contiguous()

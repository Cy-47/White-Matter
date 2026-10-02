"""Qwen3-shaped attention with Q computed locally; channels K, V supplied externally.

Mirrors `Qwen3Attention.forward`'s dispatch path, but K-norm and RoPE on the
K side are already applied at storage time inside ``KVPool``. The block selects the channel read by each layer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

import torch
import torch.nn as nn

from white_matter.modules.rotary import rotate_half
from white_matter.ops import (
    CyclicAttentionMetadata,
    StrictCausalMetadata,
    cyclic_attention,
    prepare_strict_causal_metadata,
    strict_causal_attention,
)

from .backends import attention_forward


class FeedbackAttention(Protocol):
    strict_causal: bool
    use_dummy_token: bool | None
    __call__: Callable[..., torch.Tensor]

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T_q, D')
        key_states: torch.Tensor,  # (B,H,N,d); the block selects the KV channel
        value_states: torch.Tensor,  # Same layout as key_states; keys already include norm/RoPE.
        q_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        decode_key_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        metadata: CyclicAttentionMetadata | StrictCausalMetadata | None = None,
        *,
        query_stride: int | None = None,
        query_offset: int = 0,
        cache_seqlens: torch.Tensor | None = None,
        committed_prefix: bool = False,
        jacobi: bool = False,
        prefix_length: int = 0,
    ) -> torch.Tensor: ...


class WhiteMatterAttention(nn.Module):
    """Qwen3 attention reading externally supplied keys and values."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        *,
        rms_norm_eps: float = 1e-6,
        attention_implementation: str = "sdpa",
        strict_causal: bool = False,
        num_splits: int = 0,
    ) -> None:
        super().__init__()
        self.attention_implementation = attention_implementation
        self.strict_causal = strict_causal
        self.use_dummy_token: bool | None = None
        self.num_splits = num_splits  # FA decode tuning; zero keeps its automatic choice.
        self.head_dim = head_dim
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            hidden_size,
            num_attention_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            num_attention_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T_q, D')
        key_states: torch.Tensor,  # (B,H,N,d); the block selects the KV channel
        value_states: torch.Tensor,  # Same layout as key_states; keys already include norm/RoPE.
        q_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        decode_key_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        metadata: CyclicAttentionMetadata | StrictCausalMetadata | None = None,
        *,
        query_stride: int | None = None,
        query_offset: int = 0,
        cache_seqlens: torch.Tensor | None = None,
        committed_prefix: bool = False,
        jacobi: bool = False,
        prefix_length: int = 0,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        projection_dtype = query_states.dtype
        query_states = self.q_norm(query_states).transpose(1, 2)
        q_cos, q_sin = q_position_embeddings
        query_states = query_states * q_cos.unsqueeze(1) + rotate_half(query_states) * q_sin.unsqueeze(1)
        # Persistent KV already has the attention dtype; avoid copying the prefix.
        if query_stride is None and cache_seqlens is None and not jacobi:
            key_states, value_states = key_states.to(query_states.dtype), value_states.to(query_states.dtype)
        if self.use_dummy_token is not None and (jacobi or committed_prefix):
            lengths = None if cache_seqlens is None else cache_seqlens - int(self.use_dummy_token)
            start = (
                prefix_length
                if jacobi
                else (key_states.shape[-2] - int(self.use_dummy_token) if lengths is None else lengths)
            )
            keep = None if decode_key_mask is None else decode_key_mask[:, 0, 0]
            if keep is not None and keep.dtype != torch.bool:
                keep = keep == 0
            output = strict_causal_attention(
                query_states,
                key_states,
                value_states,
                query_start=start,
                kv_lengths=lengths,
                use_dummy_token=self.use_dummy_token,
                metadata=metadata if isinstance(metadata, StrictCausalMetadata) else None,
                key_mask=keep,
                softmax_scale=self.scaling,
                backend=("reference" if getattr(self, "_force_jacobi_reference", False) else "auto")
                if jacobi
                else ("flash_attention_2" if self.attention_implementation == "flash_attention_2" else "reference"),
                num_splits=self.num_splits,
            ).to(projection_dtype)
        elif self.strict_causal and not (committed_prefix or cache_seqlens is not None or decode_key_mask is not None):
            schedule = metadata if isinstance(metadata, StrictCausalMetadata) else None
            if schedule is None and document_ids is not None:
                schedule = prepare_strict_causal_metadata(document_ids, query_states.shape[-2])
            output = strict_causal_attention(
                query_states,
                key_states,
                value_states,
                metadata=schedule,
                softmax_scale=self.scaling,
                backend="auto" if self.attention_implementation == "flash_attention_2" else "reference",
            )
        elif not self.strict_causal and (
            query_stride is not None or (metadata is not None and decode_key_mask is None)
        ):
            # Cyclic and depth-causal readers use explicit dummy-shifted bounds.
            backend: Literal["tilelang", "reference"] = (
                "tilelang"
                if query_stride is not None
                and query_states.is_cuda
                and not getattr(self, "_force_cyclic_reference", False)
                else "reference"
            )
            dtype = torch.bfloat16 if backend == "tilelang" else query_states.dtype
            output = (
                cyclic_attention(
                    query_states.to(dtype),
                    key_states.to(dtype),
                    value_states.to(dtype),
                    query_stride=1 if query_stride is None else query_stride,
                    query_offset=query_offset,
                    strict_past=self.use_dummy_token is not None and not self.use_dummy_token,
                    metadata=metadata if isinstance(metadata, CyclicAttentionMetadata) else None,
                    backend=backend,
                )
                .to(projection_dtype)
                .transpose(1, 2)
            )
        else:
            # Strict-causal sequential readers receive only committed past keys.
            output = attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask=decode_key_mask,
                scaling=self.scaling,
                implementation="sdpa" if decode_key_mask is not None else self.attention_implementation,
                is_causal=not self.strict_causal and decode_key_mask is None,
                cache_seqlens=None if self.strict_causal and decode_key_mask is not None else cache_seqlens,
                num_splits=self.num_splits,
            )
        return self.o_proj(output.reshape(*input_shape, -1).contiguous())

"""Qwen3-shaped attention with Q computed locally; channels K, V supplied externally.

Mirrors `Qwen3Attention.forward`'s dispatch path, but K-norm and RoPE on the
K side are already applied at storage time inside ``KVPool``. The block selects the channel read by each layer.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from importlib.util import find_spec
from typing import Literal, Protocol

import torch

from white_matter.modules.documents import document_cu_seqlens, document_start_mask
import torch.nn as nn
import torch.nn.functional as F

from white_matter._typing import compiler_disable
from white_matter.modules.rotary import rotate_half
from white_matter.ops import CyclicAttentionMetadata, cyclic_attention

from .backends import attention_forward


@lru_cache(maxsize=1)
def _flash_attn_available() -> bool:
    # Keep the base package importable without probing optional dependencies.
    return find_spec("flash_attn") is not None


class FeedbackAttention(Protocol):
    strict_causal: bool
    __call__: Callable[..., torch.Tensor]

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T_q, D')
        key_states: torch.Tensor,  # (B,H,N,d); the block selects the KV channel
        value_states: torch.Tensor,  # Same layout as key_states; keys already include norm/RoPE.
        q_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        decode_key_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        metadata: CyclicAttentionMetadata | None = None,
        *,
        query_stride: int | None = None,
        query_offset: int = 0,
        cache_seqlens: torch.Tensor | None = None,
        committed_prefix: bool = False,
        jacobi: bool = False,
    ) -> torch.Tensor: ...


def _selected_cu_seqlens(document_starts: torch.Tensor, keep_flat: torch.Tensor) -> torch.Tensor:
    """Varlen boundaries after pruning, with globally distinct document IDs."""
    # Row starts are document starts too. A cumulative count avoids assumptions
    # about the caller's segment labels and cannot merge different batch rows.
    segments = document_starts.reshape(-1).cumsum(0)[keep_flat]
    is_start = torch.ones_like(segments, dtype=torch.bool)
    is_start[1:] = segments[1:] != segments[:-1]
    starts = torch.nonzero(is_start, as_tuple=False).view(-1)
    end = starts.new_full((1,), segments.numel())
    return torch.cat((starts, end)).to(torch.int32)


@compiler_disable
def _lckv_flash_attention(
    Q: torch.Tensor,  # (B, Hq, T, d) — RoPE'd queries
    K: torch.Tensor,  # (B, Hkv, T, d) — RoPE'd/K-normed channels (k=1 read)
    V: torch.Tensor,  # (B, Hkv, T, d)
    document_ids: torch.Tensor | None,  # (B, T) per-token document ids, or None
    scale: float,
) -> torch.Tensor:
    """Strictly earlier-token LCKV attention, independently per document.

    Drop each document's first query and last K/V, run causal flash-varlen,
    then scatter into a zero output. First tokens receive only the residual.
    Returns (B,T,Hq,d); no dummy token is needed.
    """
    from flash_attn import flash_attn_varlen_func

    B, Hq, T, d = Q.shape
    Hkv = K.shape[1]
    device = Q.device
    if document_ids is None:  # whole sequence == one document
        document_ids = torch.zeros(B, T, dtype=torch.long, device=device)

    # flash_attn only accepts fp16/bf16. Under bf16 autocast Q/K/V are already
    # bf16 (no-op cast); an fp32 caller casts to
    # bf16 for the kernel and restores the input dtype on the scattered output.
    compute_dtype = Q.dtype if Q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    q = Q.transpose(1, 2).reshape(B * T, Hq, d).to(compute_dtype)  # (B*T, Hq, d)
    k = K.transpose(1, 2).reshape(B * T, Hkv, d).to(compute_dtype)
    v = V.transpose(1, 2).reshape(B * T, Hkv, d).to(compute_dtype)

    is_first = document_start_mask(document_ids)
    is_last = torch.ones_like(document_ids, dtype=torch.bool)
    is_last[:, :-1] = document_ids[:, 1:] != document_ids[:, :-1]
    q_keep = (~is_first).reshape(-1)  # drop each doc's first token (queries)
    k_keep = (~is_last).reshape(-1)  # drop each doc's last token (keys/values)

    q_sel = q[q_keep].contiguous()
    k_sel = k[k_keep].contiguous()
    v_sel = v[k_keep].contiguous()
    # q and k each drop exactly one token per document, so per-doc counts match
    # and the two cu_seqlens are identical; build once.
    cu = _selected_cu_seqlens(is_first, q_keep)

    out_sel = flash_attn_varlen_func(
        q_sel,
        k_sel,
        v_sel,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=T,
        max_seqlen_k=T,  # safe static upper bound (no sync)
        softmax_scale=scale,
        causal=True,
    )  # (Nq, Hq, d)

    out = Q.new_zeros(B * T, Hq, d)
    out[q_keep] = out_sel.to(out.dtype)
    return out.reshape(B, T, Hq, d)


def _lckv_sdpa_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    document_ids: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Portable LCKV attention over strictly earlier same-document tokens."""

    B, Hq, T, _ = Q.shape
    Hkv = K.shape[1]
    if Hq % Hkv:
        raise ValueError(f"query heads ({Hq}) must be divisible by KV heads ({Hkv})")
    if document_ids is None:
        document_ids = torch.zeros(B, T, dtype=torch.long, device=Q.device)
    if document_ids.shape != (B, T):
        raise ValueError(f"document_ids must have shape {(B, T)}, got {tuple(document_ids.shape)}")

    positions = torch.arange(T, device=Q.device)
    strictly_earlier = positions.view(1, T, 1) > positions.view(1, 1, T)
    same_document = document_ids.unsqueeze(2) == document_ids.unsqueeze(1)
    keep = (same_document & strictly_earlier).unsqueeze(1)
    if Hq != Hkv:
        repeats = Hq // Hkv
        K = K.repeat_interleave(repeats, dim=1)
        V = V.repeat_interleave(repeats, dim=1)
    out = F.scaled_dot_product_attention(
        Q,
        K,
        V,
        attn_mask=keep,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return out.transpose(1, 2)


def _jacobi_document_keys(
    key: torch.Tensor, value: torch.Tensor, document_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace each document's first shifted key with the shared dummy key."""
    batch, _, length, _ = key.shape
    if document_ids.shape != (batch, length):
        raise ValueError("Jacobi document IDs must match the query sequence")
    slots = torch.arange(length, device=key.device).expand(batch, length)
    slots = slots.masked_fill(document_start_mask(document_ids), 0)
    indices = slots[:, None, :, None].expand_as(key)
    return key.gather(2, indices), value.gather(2, indices), document_cu_seqlens(document_ids)


@compiler_disable
def _jacobi_flash_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    document_ids: torch.Tensor | None, scale: float,
) -> torch.Tensor:
    """Causal FlashAttention over dummy-shifted Jacobi keys, per document."""
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    dtype = query.dtype if query.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    q = query.transpose(1, 2).to(dtype)
    if document_ids is None:
        k, v = (tensor.transpose(1, 2).to(dtype) for tensor in (key, value))
        return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True).to(query.dtype)
    key, value, cu = _jacobi_document_keys(key, value, document_ids)
    k, v = (tensor.transpose(1, 2).to(dtype).flatten(0, 1) for tensor in (key, value))
    output = flash_attn_varlen_func(
        q.flatten(0, 1), k, v,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=query.shape[-2], max_seqlen_k=query.shape[-2],
        dropout_p=0.0, softmax_scale=scale, causal=True,
    )
    return output.view(query.shape[0], query.shape[-2], query.shape[1], query.shape[-1]).to(query.dtype)


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
        metadata: CyclicAttentionMetadata | None = None,
        *,
        query_stride: int | None = None,
        query_offset: int = 0,
        cache_seqlens: torch.Tensor | None = None,
        committed_prefix: bool = False,
        jacobi: bool = False,
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
        if self.strict_causal and (committed_prefix or cache_seqlens is not None or decode_key_mask is not None):
            # Sequential readers receive only earlier tokens. Prefix visibility
            # is independent of whether storage is cached or differentiable.
            output = attention_forward(
                query_states, key_states, value_states,
                attention_mask=decode_key_mask,
                scaling=self.scaling,
                implementation="sdpa" if decode_key_mask is not None else self.attention_implementation,
                is_causal=False,
                cache_seqlens=None if decode_key_mask is not None else cache_seqlens,
                num_splits=self.num_splits,
            )
        elif self.strict_causal:
            attention = (
                _lckv_flash_attention
                if query_states.is_cuda and self.attention_implementation == "flash_attention_2"
                else _lckv_sdpa_attention
            )
            output = attention(query_states, key_states, value_states, document_ids, self.scaling)
        elif jacobi:
            # Jacobi's shifted keys have a dedicated FlashAttention schedule;
            # ordinary decoder attention settings govern only other paths.
            if query_states.is_cuda and self.head_dim <= 256 and _flash_attn_available() and not getattr(
                self, "_force_jacobi_reference", False
            ):
                output = _jacobi_flash_attention(
                    query_states, key_states, value_states, document_ids, self.scaling,
                ).to(projection_dtype)
            else:
                output = cyclic_attention(
                    query_states, key_states.to(query_states.dtype), value_states.to(query_states.dtype),
                    query_stride=1, metadata=metadata, backend="reference",
                ).to(projection_dtype).transpose(1, 2)
        elif query_stride is not None or (metadata is not None and decode_key_mask is None):
            # Cyclic and depth-causal readers use explicit dummy-shifted bounds.
            backend: Literal["tilelang", "reference"] = (
                "tilelang"
                if query_stride is not None and query_states.is_cuda and not getattr(self, "_force_cyclic_reference", False)
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
                    metadata=metadata,
                    backend=backend,
                )
                .to(projection_dtype)
                .transpose(1, 2)
            )
        else:
            output = attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask=decode_key_mask,
                scaling=self.scaling,
                implementation="sdpa" if decode_key_mask is not None else self.attention_implementation,
                is_causal=decode_key_mask is None,
                cache_seqlens=cache_seqlens,
                num_splits=self.num_splits,
            )
        return self.o_proj(output.reshape(*input_shape, -1).contiguous())

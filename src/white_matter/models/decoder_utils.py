"""Tensor preparation shared by concrete HF decoders."""

import torch

from white_matter.modules.documents import (
    block_causal_additive_mask,
    document_cu_seqlens,
    document_position_ids,
    plain_causal_additive_mask,
)
from white_matter.modules.precision import cast_residual


def segment_documents_and_padding(document_ids, attention_mask, hidden_states):
    shape = hidden_states.shape[:2]
    if document_ids is not None:
        if document_ids.shape != shape:
            raise ValueError("document_ids must match the input batch and sequence dimensions")
        document_ids = document_ids.to(device=hidden_states.device)
    if attention_mask is None:
        return document_ids
    if attention_mask.shape != shape:
        raise ValueError("attention_mask must match the input batch and sequence dimensions")
    valid = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    if document_ids is None and bool(valid.all()):
        return None
    transitions = torch.zeros_like(valid, dtype=torch.long)
    boundaries = valid[:, 1:] != valid[:, :-1]
    if document_ids is not None:
        boundaries = boundaries | (document_ids[:, 1:] != document_ids[:, :-1])
    transitions[:, 1:] = boundaries.long()
    return transitions.cumsum(dim=1)


def prepare_decoder_inputs(hidden_states, config, attention_mask=None, document_ids=None):
    hidden_states = cast_residual(hidden_states, residual_dtype=config.residual_dtype)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=hidden_states.device)
    document_ids = segment_documents_and_padding(document_ids, attention_mask, hidden_states)
    return hidden_states, attention_mask, document_ids


def run_feedforward_layers(layers, x, **attention_kwargs):
    for layer in layers:
        x = layer(x, **attention_kwargs)
    return x


def prepare_attention_inputs(
    x: torch.Tensor,
    document_ids: torch.Tensor | None,
    rotary_emb,
    attn_impl: str,
    position_ids: torch.Tensor | None = None,
    *,
    past_key_values=None,
    attention_mask=None,
):
    """Prepare ordinary-layer arguments once for pre/post layers or a full decoder.

    FlashAttention receives varlen boundaries; SDPA/eager receive an additive
    document mask. Cached continuation additionally masks the committed prefix.
    """
    B, T, _ = x.shape
    device = x.device
    kwargs = {}
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
        seen = past_key_values.get_seq_length()
        if seen:
            mask = None
            if document_ids is not None:
                keys = torch.arange(seen + T, device=device)
                queries = torch.arange(seen, seen + T, device=device)
                keep = (document_ids[:, :, None] == past_key_values.document_ids[:, None, :]) & (
                    queries[:, None] >= keys
                )
                mask = x.new_zeros(keep.shape).masked_fill(~keep, torch.finfo(x.dtype).min)[:, None]
            return dict(
                position_embeddings=rotary_emb(x, position_ids), attention_mask=mask, cached_attention=True, **kwargs
            )
        document_ids = segment_documents_and_padding(document_ids, attention_mask, x)
    if document_ids is None:
        if position_ids is None:
            position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        attention_mask = (
            plain_causal_additive_mask(B, T, dtype=x.dtype, device=device) if attn_impl == "eager" else None
        )
        return dict(position_embeddings=rotary_emb(x, position_ids), attention_mask=attention_mask, **kwargs)
    if position_ids is None:
        position_ids = document_position_ids(document_ids)
    pos_emb = rotary_emb(x, position_ids)
    if attn_impl == "flash_attention_2":
        cu = document_cu_seqlens(document_ids)
        kwargs.update(
            {
                "cu_seq_lens_q": cu,
                "cu_seq_lens_k": cu,
                "max_length_q": T,
                "max_length_k": T,
            }
        )
        return dict(position_embeddings=pos_emb, attention_mask=None, **kwargs)
    return dict(
        position_embeddings=pos_emb, attention_mask=block_causal_additive_mask(document_ids, dtype=x.dtype), **kwargs
    )

"""Document boundaries, reset positions, and causal masks for packed sequences."""

import torch


def document_ids_from_eos(input_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Count preceding EOS tokens; each EOS belongs to the document it closes."""
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError(f"eos_id must be a nonnegative integer, got {eos_id!r}")
    is_eos = (input_ids == eos_id).long()
    return is_eos.cumsum(dim=1) - is_eos


def document_start_mask(document_ids: torch.Tensor) -> torch.Tensor:
    """Mark each row's first token and every subsequent document transition."""
    starts = torch.ones_like(document_ids, dtype=torch.bool)
    starts[:, 1:] = document_ids[:, 1:] != document_ids[:, :-1]
    return starts


def document_position_ids(document_ids: torch.Tensor) -> torch.Tensor:
    """Reset RoPE at each document: [0,0,0,1,1,2] maps to [0,1,2,0,1,0]."""
    positions = torch.arange(document_ids.shape[1], device=document_ids.device).expand_as(document_ids)
    starts = torch.where(document_start_mask(document_ids), positions, 0).cummax(dim=1).values
    return positions - starts


def document_cu_seqlens(document_ids: torch.Tensor) -> torch.Tensor:
    """FlashAttention boundaries for flattened (B*T) tokens, including row starts."""
    starts = document_start_mask(document_ids).flatten().nonzero().flatten()
    return torch.cat((starts, starts.new_full((1,), document_ids.numel()))).to(torch.int32)


def plain_causal_additive_mask(
    batch_size: int,
    sequence_length: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Standard (B,1,T,T) additive causal mask."""
    allowed = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device).tril_()
    mask = torch.zeros(batch_size, 1, sequence_length, sequence_length, dtype=dtype, device=device)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)


def block_causal_additive_mask(document_ids: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Standard causal attention restricted to each document, in (B,1,T,T) layout."""
    batch, length = document_ids.shape
    same = document_ids.unsqueeze(2) == document_ids.unsqueeze(1)
    causal = torch.ones(length, length, dtype=torch.bool, device=document_ids.device).tril_()
    mask = torch.zeros(batch, 1, length, length, dtype=dtype, device=document_ids.device)
    return mask.masked_fill_(~(same & causal).unsqueeze(1), torch.finfo(dtype).min)


def feedback_document_mask(past: torch.Tensor, current: torch.Tensor, *, use_dummy_token: bool = True) -> torch.Tensor:
    """Visible committed tokens, including the always-visible leading dummy."""
    same = (past == current) & (current >= 0)
    return torch.cat((torch.ones_like(current, dtype=torch.bool), same), dim=1) if use_dummy_token else same

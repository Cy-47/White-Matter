"""Explicit document inputs saved by attention autograd and recomputation."""

from typing import NamedTuple

import torch


class CyclicAttentionMetadata(NamedTuple):
    query_document_ids: torch.Tensor
    key_document_ids: torch.Tensor
    # Per-query lower key bound and per-key exclusive upper query bound.
    # The shared dummy is handled separately from these document bounds.
    query_key_start: torch.Tensor
    key_query_end: torch.Tensor


def prepare_cyclic_attention_metadata(
    query_document_ids: torch.Tensor, key_document_ids: torch.Tensor
) -> CyclicAttentionMetadata:
    """Prepare bounds outside compiled loops from sorted per-row segment IDs.

    Both inputs are integer matrices (batch, length). Keys must start with one
    globally visible dummy slot, ID -1. Remaining IDs are nonnegative and
    nondecreasing; queries are nonnegative and nondecreasing. IDs must fit int32
    even when input tensors use int64. Slot-to-token
    mapping and document numbering belong to the caller. Bounds must be rebuilt
    when these inputs change and must not be modified between forward/backward.
    """
    if query_document_ids.ndim != 2 or key_document_ids.ndim != 2:
        raise ValueError("segment IDs must have shape (batch, length)")
    if (
        query_document_ids.shape[0] != key_document_ids.shape[0]
        or min(*query_document_ids.shape, key_document_ids.shape[1]) < 1
    ):
        raise ValueError("segment IDs require matching nonempty batches and sequences")
    if query_document_ids.device != key_document_ids.device:
        raise ValueError("segment IDs must be on the same device")
    for ids in (query_document_ids, key_document_ids):
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("segment IDs must be int32 or int64")
        if ids.dtype == torch.int64 and bool((ids > torch.iinfo(torch.int32).max).any()):
            raise ValueError("segment IDs must fit int32 (maximum 2147483647)")
        if bool((ids[:, 1:] < ids[:, :-1]).any()):
            raise ValueError("segment IDs must be nondecreasing within each row")
    if bool((query_document_ids < 0).any()) or bool((key_document_ids[:, 1:] < 0).any()):
        raise ValueError("only the leading key dummy may have a negative segment ID")
    if not bool((key_document_ids[:, 0] == -1).all()):
        raise ValueError("document attention requires a leading dummy key with segment ID -1")
    q = query_document_ids.to(torch.int32).contiguous()
    k = key_document_ids.to(torch.int32).contiguous()
    start = torch.searchsorted(k, q).to(torch.int32)
    end = torch.searchsorted(q, k, right=True).to(torch.int32)
    end[:, 0] = q.shape[1]
    return CyclicAttentionMetadata(q, k, start, end)

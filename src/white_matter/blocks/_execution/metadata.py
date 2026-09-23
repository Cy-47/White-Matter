"""Map packed documents to dummy-shifted feedback keys and cyclic queries."""

from collections.abc import Sequence

import torch

from white_matter.modules.documents import document_start_mask

from white_matter.ops import CyclicAttentionMetadata


def prepare_feedback_metadata(
    document_ids: torch.Tensor,
    query_positions: Sequence[torch.Tensor],
    key_length: int,
) -> tuple[CyclicAttentionMetadata, ...]:
    """Build metadata for query groups ordered by cyclic residue.

    Key slot zero is the shared dummy (-1); slot s+1 stores token s. Queries
    in group r occupy r, r+stride, ... . Jacobi uses one group with stride 1.
    Document starts/ends and the shared key segments are computed outside the
    layer/pass loops. All returned tensors remain explicit attention inputs.
    """
    stride = len(query_positions)
    if not stride:
        raise ValueError("cyclic query groups must be nonempty")
    batch, length = document_ids.shape
    if key_length > length + 1:
        raise ValueError(f"key_length={key_length} exceeds dummy-shifted token extent {length + 1}")
    positions = torch.arange(length, device=document_ids.device).expand(batch, length)
    is_start = document_start_mask(document_ids)
    token_start = torch.where(is_start, positions, 0).cummax(dim=1).values
    after_marker = torch.full_like(positions, length)
    after_marker[:, :-1] = torch.where(is_start[:, 1:], positions[:, 1:], length)
    token_end = after_marker.flip((1,)).cummin(dim=1).values.flip((1,))

    segments = document_ids.to(torch.int32)
    key_document_ids = torch.cat((segments.new_full((batch, 1), -1), segments), dim=1)[:, :key_length].contiguous()
    metadata = []
    for residue, slots in enumerate(query_positions):
        query_document_ids = segments.index_select(1, slots).contiguous()
        query_key_start = token_start.index_select(1, slots).to(torch.int32).add_(1).contiguous()
        query_end = (
            torch.div((token_end - residue).clamp_min(0) + stride - 1, stride, rounding_mode="floor")
            .clamp_max(slots.numel())
            .to(torch.int32)
        )
        key_query_end = torch.cat((segments.new_full((batch, 1), slots.numel()), query_end), dim=1)[
            :, :key_length
        ].contiguous()
        metadata.append(CyclicAttentionMetadata(query_document_ids, key_document_ids, query_key_start, key_query_end))
    return tuple(metadata)

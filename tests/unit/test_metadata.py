"""Feedback slot mapping agrees with the independent segment-search constructor."""

import pytest
import torch

from white_matter.blocks._execution.metadata import prepare_feedback_metadata
from white_matter.ops import prepare_cyclic_attention_metadata


@pytest.mark.parametrize("stride", [1, 2, 4])
@pytest.mark.parametrize("length", [17, 33])
def test_feedback_metadata_matches_segment_search(stride, length):
    torch.manual_seed(length)
    segments = torch.randint(0, 2, (2, length)).cumsum(1)
    # Cover singleton documents and different boundaries in each batch row.
    segments[0] = torch.arange(length)
    residues = range(stride)
    positions = [torch.arange(r, length, stride) for r in residues]
    actual = prepare_feedback_metadata(segments, positions, length)
    key_document_ids = torch.cat((segments.new_full((2, 1), -1), segments[:, :-1]), dim=1)
    for slots, residue in zip(positions, residues, strict=True):
        expected = prepare_cyclic_attention_metadata(segments[:, slots], key_document_ids)
        torch.testing.assert_close(actual[residue], expected, rtol=0, atol=0)


def test_feedback_metadata_rejects_empty_groups():
    with pytest.raises(ValueError, match="nonempty"):
        prepare_feedback_metadata(torch.zeros(1, 8, dtype=torch.long), [], 8)

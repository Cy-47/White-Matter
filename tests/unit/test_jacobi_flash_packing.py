"""Packed-document FlashAttention inputs retain Jacobi's dummy-key semantics."""

import torch

from white_matter.blocks._execution.metadata import prepare_feedback_metadata
from white_matter.ops import cyclic_attention, prepare_strict_causal_metadata, strict_causal_attention


def test_packed_jacobi_causal_segments_match_reference_outputs_and_gradients():
    torch.manual_seed(1827)
    documents = torch.tensor([[0, 0, 1, 1, 1, 2], [0, 1, 1, 2, 3, 3]])
    query = torch.randn(2, 4, 6, 8, requires_grad=True)
    key = torch.randn(2, 2, 7, 8, requires_grad=True)
    value = torch.randn(2, 2, 7, 8, requires_grad=True)
    schedule = prepare_strict_causal_metadata(documents, 6, use_dummy_token=True)
    assert schedule.cu_queries.tolist() == [0, 2, 5, 6, 7, 9, 10, 12]
    packed_result = strict_causal_attention(
        query, key, value, use_dummy_token=True, metadata=schedule, backend="reference"
    ).transpose(1, 2)
    slots = torch.arange(documents.shape[1])
    metadata = prepare_feedback_metadata(documents, [slots], documents.shape[1])[0]
    reference = cyclic_attention(query, key[..., :6, :], value[..., :6, :], query_stride=1, metadata=metadata)
    torch.testing.assert_close(packed_result, reference, rtol=1e-5, atol=1e-6)
    probe = torch.randn_like(reference)
    packed_grad = torch.autograd.grad((packed_result * probe).sum(), (query, key, value), retain_graph=True)
    reference_grad = torch.autograd.grad((reference * probe).sum(), (query, key, value))
    for actual, expected in zip(packed_grad, reference_grad, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

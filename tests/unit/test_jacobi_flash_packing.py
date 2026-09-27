"""Packed-document FlashAttention inputs retain Jacobi's dummy-key semantics."""

import torch
import torch.nn.functional as F

from white_matter.blocks._execution.metadata import prepare_feedback_metadata
from white_matter.layers.white_matter import _jacobi_document_keys
from white_matter.ops import cyclic_attention


def test_packed_jacobi_causal_segments_match_reference_outputs_and_gradients():
    torch.manual_seed(1827)
    documents = torch.tensor([[0, 0, 1, 1, 1, 2], [0, 1, 1, 2, 3, 3]])
    query = torch.randn(2, 4, 6, 8, requires_grad=True)
    key = torch.randn(2, 2, 6, 8, requires_grad=True)
    value = torch.randn(2, 2, 6, 8, requires_grad=True)
    packed_key, packed_value, cu = _jacobi_document_keys(key, value, documents)
    assert cu.tolist() == [0, 2, 5, 6, 7, 9, 10, 12]
    output = []
    for start, stop in zip(cu[:-1].tolist(), cu[1:].tolist(), strict=True):
        row, lo, hi = start // 6, start % 6, stop % 6 or 6
        q = query[row : row + 1, :, lo:hi]
        k = packed_key[row : row + 1, :, lo:hi]
        v = packed_value[row : row + 1, :, lo:hi]
        output.append(F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True))
    packed_result = (
        torch.cat(output, dim=2)
        .reshape(
            query.shape[1],
            *documents.shape,
            query.shape[-1],
        )
        .permute(1, 0, 2, 3)
    )
    slots = torch.arange(documents.shape[1])
    metadata = prepare_feedback_metadata(documents, [slots], documents.shape[1])[0]
    reference = cyclic_attention(query, key, value, query_stride=1, metadata=metadata)
    torch.testing.assert_close(packed_result, reference, rtol=1e-5, atol=1e-6)
    probe = torch.randn_like(reference)
    packed_grad = torch.autograd.grad((packed_result * probe).sum(), (query, key, value), retain_graph=True)
    reference_grad = torch.autograd.grad((reference * probe).sum(), (query, key, value))
    for actual, expected in zip(packed_grad, reference_grad, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

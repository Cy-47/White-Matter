"""The standalone CUDA operator must consume the same metadata as the reference."""

import pytest
import torch

from white_matter.ops import cyclic_attention, prepare_cyclic_attention_metadata

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("head_dim", [64, 96, 128])
@pytest.mark.parametrize("query_length", [1, 33])
def test_backward_preprocessing_matches_fp64(transposed, head_dim, query_length):
    from white_matter.ops.cyclic_attention._tilelang.backward_preprocess import backward_preprocess

    torch.manual_seed(81)
    shape = (2, 3, query_length, head_dim)
    dout = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    out = torch.randn(
        (2, query_length, 3, head_dim) if transposed else shape, device="cuda", dtype=torch.bfloat16
    )
    if transposed:
        out = out.transpose(1, 2)
    actual = backward_preprocess(dout, out)
    expected = (dout.double() * out.double()).sum(-1)
    assert actual.dtype == torch.float32 and actual.is_contiguous()
    # Bound FP32 reduction roundoff against the sum of absolute products,
    # including rows where positive and negative terms nearly cancel.
    bound = torch.finfo(torch.float32).eps * head_dim * (dout.double() * out.double()).abs().sum(-1)
    assert torch.all((actual.double() - expected).abs() <= bound)


@pytest.mark.parametrize("gqa_ratio", [1, 2, 3])
@pytest.mark.parametrize("documents", [False, True])
@pytest.mark.parametrize("head_dim", [64, 96, 128])
def test_operator_outputs_and_all_input_gradients(documents, head_dim, gqa_ratio):
    import tilelang  # noqa: F401

    torch.manual_seed(23)
    q = torch.randn(2, 2 * gqa_ratio, 33, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 2, 132, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    metadata = None
    if documents:
        seg = torch.stack((torch.arange(132, device="cuda") // 19, torch.arange(132, device="cuda") // 31))
        metadata = prepare_cyclic_attention_metadata(
            seg[:, 1::4], torch.cat((seg.new_full((2, 1), -1), seg[:, :-1]), 1)
        )
    probe = torch.randn_like(q)
    expected = cyclic_attention(q, k, v, query_stride=4, query_offset=1, metadata=metadata)
    compiled_attention = torch.compile(cyclic_attention, fullgraph=True)
    actual = compiled_attention(q, k, v, query_stride=4, query_offset=1, metadata=metadata, backend="tilelang")
    # Projection-ready storage avoids a transpose copy after every reader layer.
    assert actual.transpose(1, 2).is_contiguous()
    expected_grads = torch.autograd.grad(expected, (q, k, v), probe)
    actual_grads = torch.autograd.grad(actual, (q, k, v), probe)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2e-2)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=1e-1, atol=2e-2)


@pytest.mark.parametrize("documents", [False, True])
def test_equal_scores_preserve_uniform_attention_across_tiles(documents):
    q = torch.full((1, 2, 1, 128), 13., device="cuda", dtype=torch.bfloat16)
    k = torch.full((1, 1, 4096, 128), 29., device="cuda", dtype=torch.bfloat16)
    v = torch.ones_like(k)
    v[:, :, 2048:].neg_()
    metadata = None
    if documents:
        qseg = torch.zeros((1, 1), device="cuda", dtype=torch.long)
        kseg = torch.zeros((1, 4096), device="cuda", dtype=torch.long)
        kseg[:, 0] = -1
        metadata = prepare_cyclic_attention_metadata(qseg, kseg)
    # All scores are equal and all keys visible: the signed values average to
    # exactly zero. Rescaling by anything other than one biases earlier tiles.
    actual = cyclic_attention(q, k, v, query_stride=4096, query_offset=4095,
                              metadata=metadata, backend="tilelang")
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)


@pytest.mark.parametrize('element_stride', [1, 2])
def test_native_cache_views_match_contiguous_attention_and_gradients(element_stride):
    torch.compiler.reset()
    torch.manual_seed(29)
    q = torch.randn(2, 4, 33, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    storage = [torch.randn(4, 3, 2, 145, 128 * element_stride, device='cuda', dtype=torch.bfloat16)
               for _ in range(2)]
    # Batch slice, channel selection, unused capacity and optional strided elements.
    k, v = [t[1:3, 1, :, :132, ::element_stride].detach().requires_grad_() for t in storage]
    assert not k.is_contiguous()
    attention = torch.compile(cyclic_attention, fullgraph=True)
    options = dict(query_stride=4, query_offset=1, backend='tilelang')
    expected = attention(q, k.contiguous(), v.contiguous(), **options)
    actual = attention(q, k, v, **options)
    probe = torch.randn_like(q)
    expected_grads = torch.autograd.grad(expected, (q, k, v), probe)
    actual_grads = torch.autograd.grad(actual, (q, k, v), probe)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=0, atol=0)

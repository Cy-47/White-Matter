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
    out = torch.randn((2, query_length, 3, head_dim) if transposed else shape, device="cuda", dtype=torch.bfloat16)
    if transposed:
        out = out.transpose(1, 2)
    actual = backward_preprocess(dout, out)
    expected = (dout.double() * out.double()).sum(-1)
    assert actual.dtype == torch.float32
    assert actual.is_contiguous()
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
    q = torch.full((1, 2, 1, 128), 13.0, device="cuda", dtype=torch.bfloat16)
    k = torch.full((1, 1, 4096, 128), 29.0, device="cuda", dtype=torch.bfloat16)
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
    actual = cyclic_attention(q, k, v, query_stride=4096, query_offset=4095, metadata=metadata, backend="tilelang")
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)


@pytest.mark.parametrize("element_stride", [1, 2])
def test_native_cache_views_match_contiguous_attention_and_gradients(element_stride):
    torch.compiler.reset()
    torch.manual_seed(29)
    q = torch.randn(2, 4, 33, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    storage = [torch.randn(4, 3, 2, 145, 128 * element_stride, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    # Batch slice, channel selection, unused capacity and optional strided elements.
    k, v = [t[1:3, 1, :, :132, ::element_stride].detach().requires_grad_() for t in storage]
    assert not k.is_contiguous()
    attention = torch.compile(cyclic_attention, fullgraph=True)
    options = {"query_stride": 4, "query_offset": 1, "backend": "tilelang"}
    expected = attention(q, k.contiguous(), v.contiguous(), **options)
    actual = attention(q, k, v, **options)
    probe = torch.randn_like(q)
    expected_grads = torch.autograd.grad(expected, (q, k, v), probe)
    actual_grads = torch.autograd.grad(actual, (q, k, v), probe)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=0, atol=0)


@pytest.mark.parametrize("documents", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_dynamic_shapes_reuse_kernels_and_preserve_gradients(documents, compiled):
    from white_matter.ops.cyclic_attention._tilelang.registration import _get_kernel

    torch.manual_seed(47)
    _get_kernel.cache_clear()
    attention = torch.compile(cyclic_attention, fullgraph=True, dynamic=True) if compiled else cyclic_attention
    # Change batch, both lengths, cache capacity, and residue independently of
    # head/tile configuration. Include partial tiles and unequal K/V strides.
    for batch, qlen, kvlen, padding, offset in ((2, 17, 71, 5, 1), (3, 65, 268, 19, 2), (2, 19, 83, 7, 0)):
        q = torch.randn(batch, 4, qlen, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k, v = [
            torch.randn(batch, 2, kvlen + extra, 64, device="cuda", dtype=q.dtype)[:, :, :kvlen]
            .detach()
            .requires_grad_()
            for extra in (padding, padding + 3)
        ]
        metadata = None
        if documents:
            qseg = ((offset + 4 * torch.arange(qlen, device="cuda")) // 23).expand(batch, -1).contiguous()
            kseg = ((torch.arange(kvlen, device="cuda") - 1) // 23).expand(batch, -1).contiguous()
            metadata = prepare_cyclic_attention_metadata(qseg, kseg)
        options = {"query_stride": 4, "query_offset": offset, "metadata": metadata}
        expected = cyclic_attention(q, k, v, **options)
        actual = attention(q, k, v, backend="tilelang", **options)
        probe = torch.randn_like(q)
        expected_grads = torch.autograd.grad(expected, (q, k, v), probe)
        actual_grads = torch.autograd.grad(actual, (q, k, v), probe)
        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2e-2)
        torch.testing.assert_close(actual_grads, expected_grads, rtol=1e-1, atol=2e-2)
        # One forward, one dQ and one dKV specialization for the whole sweep.
        assert _get_kernel.cache_info().misses == 3


@pytest.mark.parametrize("documents", [False, True])
def test_invisible_trailing_key_tiles_have_zero_gradients(documents):
    torch.manual_seed(127)
    q = torch.randn(1, 4, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 2, 513, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    metadata = None
    if documents:
        qseg = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
        kseg = torch.zeros((1, 513), device="cuda", dtype=torch.int32)
        kseg[:, 0] = -1
        metadata = prepare_cyclic_attention_metadata(qseg, kseg)
    options = {"query_stride": 4, "query_offset": 0, "metadata": metadata}
    expected = cyclic_attention(q, k, v, **options)
    actual = cyclic_attention(q, k, v, backend="tilelang", **options)
    probe = torch.randn_like(q)
    expected_grads = torch.autograd.grad(expected, (q, k, v), probe)
    actual_grads = torch.autograd.grad(actual, (q, k, v), probe)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2e-2)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=1e-1, atol=2e-2)
    # Only slot zero is visible. Later KV tiles have query-loop start >= end,
    # including the final partial tile; none may contribute to either gradient.
    for gradient in actual_grads[1:]:
        torch.testing.assert_close(gradient[:, :, 1:], torch.zeros_like(gradient[:, :, 1:]), rtol=0, atol=0)


@pytest.mark.parametrize("misaligned_key", [False, True])
@pytest.mark.parametrize("head_dim", [64, 96, 128])
def test_misaligned_cache_pointer_is_normalized(misaligned_key, head_dim):
    from white_matter.ops.cyclic_attention._tilelang.registration import _get_kernel

    torch.manual_seed(61)
    q = torch.randn(2, 4, 17, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    shape = (2, 2, 71, head_dim)
    size = 2 * 2 * 71 * head_dim
    aligned = torch.randn(shape, device="cuda", dtype=q.dtype, requires_grad=True)
    misaligned = torch.randn(size + 1, device="cuda", dtype=q.dtype)[1:].view(shape).requires_grad_()
    assert misaligned.data_ptr() % 16 != 0
    assert all(s % 8 == 0 for s in misaligned.stride()[:3])
    k, v = (misaligned, aligned) if misaligned_key else (aligned, misaligned)
    options = {"query_stride": 4, "query_offset": 1}
    expected = cyclic_attention(q, k, v, **options)
    _get_kernel.cache_clear()
    actual = cyclic_attention(q, k, v, backend="tilelang", **options)
    probe = torch.randn_like(q)
    expected_grads = torch.autograd.grad(expected, (q, k, v), probe)
    actual_grads = torch.autograd.grad(actual, (q, k, v), probe)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2e-2)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=1e-1, atol=2e-2)
    assert _get_kernel.cache_info().misses == 3
    # Normalized and already-aligned inputs reuse the same compiled kernels.
    cyclic_attention(q, k.clone(), v.clone(), backend="tilelang", **options)
    assert _get_kernel.cache_info().misses == 3


@pytest.mark.parametrize("documents", [False, True])
def test_single_query_with_misaligned_storage_preserves_gradients(documents):
    torch.manual_seed(67)
    shape = (2, 4, 1, 128)
    q = torch.randn(2 * 4 * 128 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(shape).requires_grad_()
    assert q.transpose(1, 2).is_contiguous()
    assert q.data_ptr() % 16 != 0
    k = torch.randn(2, 2, 17, 128, device="cuda", dtype=q.dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    metadata = None
    if documents:
        qseg = torch.zeros((2, 1), device="cuda", dtype=torch.int32)
        kseg = torch.zeros((2, 17), device="cuda", dtype=torch.int32)
        kseg[:, 0] = -1
        metadata = prepare_cyclic_attention_metadata(qseg, kseg)
    options = {"query_stride": 4, "query_offset": 1, "metadata": metadata}
    expected = cyclic_attention(q, k, v, **options)
    actual = cyclic_attention(q, k, v, backend="tilelang", **options)
    probe = torch.randn_like(q)
    expected_grads = torch.autograd.grad(expected, (q, k, v), probe)
    actual_grads = torch.autograd.grad(actual, (q, k, v), probe)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2e-2)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=1e-1, atol=2e-2)


@pytest.mark.parametrize("head_dim", [64, 96, 128])
@pytest.mark.parametrize("future_documents", [False, True])
def test_document_attention_skips_prior_tiles_and_retains_dummy(head_dim, future_documents):
    torch.manual_seed(193)
    batch, query_length, key_length, stride = 2, 129, 1033, 8
    q = torch.randn(batch, 4, query_length, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, 2, key_length, head_dim, device="cuda", dtype=q.dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    # Different document boundaries per batch cross query-tile boundaries;
    # later tiles can skip several KV tiles while still visiting dummy tile 0.
    key_ids = torch.stack(
        (torch.arange(key_length, device="cuda") // 257, torch.arange(key_length, device="cuda") // 321)
    )
    key_ids[:, 0] = -1
    query_ids = key_ids[:, : query_length * stride : stride].clamp_min(0)
    if future_documents:
        # These documents have no causally visible keys. For the second batch
        # the first tile starts beyond its dense causal loop end.
        query_ids = query_ids + 2
    metadata = prepare_cyclic_attention_metadata(query_ids, key_ids)
    options = {"query_stride": stride, "metadata": metadata}
    expected = cyclic_attention(q, k, v, **options)
    actual = cyclic_attention(q, k, v, backend="tilelang", **options)
    probe = torch.randn_like(q)
    expected_grads = torch.autograd.grad(expected, (q, k, v), probe)
    actual_grads = torch.autograd.grad(actual, (q, k, v), probe)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2e-2)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=1e-1, atol=2e-2)
    if future_documents:
        expected_dummy = v[:, :, :1].repeat_interleave(2, dim=1).expand_as(q)
        torch.testing.assert_close(actual, expected_dummy, rtol=0, atol=0)
        torch.testing.assert_close(actual_grads[0], torch.zeros_like(q), rtol=0, atol=2e-2)
        torch.testing.assert_close(actual_grads[1], torch.zeros_like(k), rtol=0, atol=2e-2)
        torch.testing.assert_close(actual_grads[2][:, :, 1:], torch.zeros_like(v[:, :, 1:]), rtol=0, atol=0)


@pytest.mark.parametrize("documents", [False, True])
@pytest.mark.parametrize("query_length", [1, 17, 33])
def test_short_queries_ignore_future_cache_tiles(documents, query_length):
    torch.manual_seed(107)
    query_stride, query_offset, kv_length = 8, 3, 2048
    q = torch.randn(2, 4, query_length, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 2, kv_length, 128, device="cuda", dtype=q.dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    metadata = None
    if documents:
        positions = torch.arange(kv_length, device="cuda")
        key_ids = torch.stack((positions // 97, positions // 113)).to(torch.int32)
        key_ids[:, 0] = -1
        query_positions = torch.arange(query_length, device="cuda") * query_stride + query_offset
        metadata = prepare_cyclic_attention_metadata(key_ids[:, query_positions], key_ids)
    options = {"query_stride": query_stride, "query_offset": query_offset, "metadata": metadata}
    expected = cyclic_attention(q, k, v, **options)
    actual = cyclic_attention(q, k, v, backend="tilelang", **options)
    probe = torch.randn_like(q)
    expected_grads = torch.autograd.grad(expected, (q, k, v), probe)
    actual_grads = torch.autograd.grad(actual, (q, k, v), probe)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2e-2)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=1e-1, atol=2e-2)
    first_future_key = query_offset + query_stride * (query_length - 1) + 1
    for gradient in actual_grads[1:]:
        assert torch.count_nonzero(gradient[:, :, first_future_key:]) == 0

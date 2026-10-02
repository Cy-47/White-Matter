"""One attention contract for Jacobi, cached prefill and single-token decode."""

import pytest
import torch

from white_matter.ops import prepare_strict_causal_metadata, strict_causal_attention

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]
    ),
]


def oracle(q, k, v, starts, dummy, docs=None, lengths=None):
    """Explicit per-query softmax, independent of FA alignment and packing."""
    batch, heads, count, dim = q.shape
    outputs = []
    for b in range(batch):
        rows = []
        for i in range(count):
            pos = int(starts[b]) + i
            slots = list(range(min(pos, k.shape[-2] - int(dummy), int(lengths[b]) if lengths is not None else pos)))
            if docs is not None:
                start = pos
                while start and (docs[b, start - 1] < 0 or docs[b, start - 1] == docs[b, pos]):
                    start -= 1
                slots = [s for s in slots if s >= start and docs[b, s] >= 0 and docs[b, pos] >= 0]
            indices = ([0] if dummy and (docs is None or docs[b, pos] >= 0) else []) + [s + int(dummy) for s in slots]
            if indices:
                kk = k[b, :, indices].repeat_interleave(heads // k.shape[1], 0)
                vv = v[b, :, indices].repeat_interleave(heads // k.shape[1], 0)
                weights = ((q[b, :, i, None].float() * kk.float()).sum(-1) * dim**-0.5).softmax(-1)
                rows.append((weights[..., None] * vv.float()).sum(-2).to(q.dtype))
            else:
                rows.append(q[b, :, i] * 0 + (k[b, :, :0].sum() + v[b, :, :0].sum()))
        outputs.append(torch.stack(rows))
    return torch.stack(outputs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dummy", [False, True])
@pytest.mark.parametrize(("prefix", "count"), [(0, 1), (0, 7), (3, 5), (4, 1)])
@pytest.mark.parametrize("packed", [False, True])
def test_attention_outputs_and_gradients(device, dummy, prefix, count, packed):
    torch.manual_seed(29)
    dtype = torch.bfloat16 if device == "cuda" else torch.float64
    q = torch.randn(2, 4, count, 64, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 2, prefix + count + int(dummy), 64, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    starts = torch.full((2,), prefix, device=device)
    docs = None
    if packed:
        docs = torch.arange(prefix + count, device=device)[None].expand(2, -1).clone() // 3
        docs[1] = torch.arange(prefix + count, device=device) // 2
    metadata = (
        None if docs is None else prepare_strict_causal_metadata(docs, count, query_start=prefix, use_dummy_token=dummy)
    )
    actual = strict_causal_attention(q, k, v, query_start=prefix, use_dummy_token=dummy, metadata=metadata)
    expected = oracle(q, k, v, starts, dummy, docs)
    tolerance = {"rtol": 0.03, "atol": 0.025} if device == "cuda" else {"rtol": 2e-6, "atol": 2e-6}
    torch.testing.assert_close(actual, expected, **tolerance)
    probe = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * probe).sum(), (q, k, v), retain_graph=True)
    ge = torch.autograd.grad((expected * probe).sum(), (q, k, v))
    for a, e in zip(ga, ge, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, e, **tolerance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dummy", [False, True])
@torch.no_grad()
def test_ragged_decode_and_unused_capacity(device, dummy):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    q = torch.randn(3, 4, 1, 64, device=device, dtype=dtype)
    k = torch.randn(3, 2, 9 + int(dummy), 64, device=device, dtype=dtype)
    v = torch.randn_like(k)
    lengths = torch.tensor([0, 3, 7], device=device, dtype=torch.int32)
    for b, length in enumerate(lengths.tolist()):
        k[b, :, length + int(dummy) :] = float("nan")
        v[b, :, length + int(dummy) :] = float("nan")
    actual = strict_causal_attention(q, k, v, query_start=lengths, kv_lengths=lengths, use_dummy_token=dummy)
    expected = oracle(q, k, v, lengths, dummy)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.02)


@pytest.mark.parametrize("dummy", [False, True])
def test_ragged_prefill_and_padding(dummy):
    starts = torch.tensor([0, 3])
    docs = torch.tensor([[0, 0, 1, -1, -1, -1, -1], [0, 0, 0, 0, 1, 1, -1]])
    q = torch.randn(2, 4, 4, 8)
    k = torch.randn(2, 2, 7 + int(dummy), 8)
    v = torch.randn_like(k)
    metadata = prepare_strict_causal_metadata(docs, 4, query_start=starts, use_dummy_token=dummy)
    actual = strict_causal_attention(q, k, v, query_start=starts, use_dummy_token=dummy, metadata=metadata)
    torch.testing.assert_close(actual, oracle(q, k, v, starts, dummy, docs), rtol=1e-5, atol=1e-6)


def test_current_and_future_keys_have_zero_gradient():
    q = torch.randn(1, 2, 1, 8, requires_grad=True)
    k = torch.randn(1, 1, 7, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    strict_causal_attention(q, k, v, query_start=3).sum().backward()
    assert k.grad[..., 3:, :].count_nonzero() == v.grad[..., 3:, :].count_nonzero() == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dummy", [False, True])
def test_ragged_prefixes_with_truncated_history(device, dummy):
    q = torch.randn(2, 4, 4, 64, device=device, requires_grad=True)
    k = torch.randn(2, 2, 8 + int(dummy), 64, device=device, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    starts = torch.tensor([0, 3], device=device)
    lengths = torch.tensor([1, 4], device=device)
    actual = strict_causal_attention(q, k, v, query_start=starts, kv_lengths=lengths, use_dummy_token=dummy)
    expected = oracle(q, k, v, starts, dummy, lengths=lengths)
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.02)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dummy", [False, True])
def test_document_metadata_with_unstored_decode_query(device, dummy):
    q = torch.randn(2, 4, 1, 64, device=device)
    k = torch.randn(2, 2, 3 + int(dummy), 64, device=device)
    v = torch.randn_like(k)
    docs = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1]], device=device)
    metadata = prepare_strict_causal_metadata(docs, 1, query_start=3, use_dummy_token=dummy)
    actual = strict_causal_attention(q, k, v, query_start=3, use_dummy_token=dummy, metadata=metadata)
    expected = oracle(q, k, v, torch.tensor([3, 3]), dummy, docs)
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.02)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dummy", [False, True])
def test_cached_padding_preserves_document_history(device, dummy):
    docs = torch.tensor([[0, 0, -2, -2, 0, 0, -2, 0], [0, -2, 1, -2, 1, 0, 0, -2]], device=device)
    q = torch.randn(2, 4, 4, 64, device=device, requires_grad=True)
    k = torch.randn(2, 2, 8 + int(dummy), 64, device=device, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    metadata = prepare_strict_causal_metadata(docs, 4, query_start=4, use_dummy_token=dummy)
    actual = strict_causal_attention(q, k, v, metadata=metadata, use_dummy_token=dummy)
    expected = oracle(q, k, v, torch.tensor([4, 4]), dummy, docs)
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.02)
    probe = torch.randn_like(actual)
    gradients = [torch.autograd.grad((out * probe).sum(), (q, k, v), retain_graph=True) for out in (actual, expected)]
    torch.testing.assert_close(*gradients, rtol=0.04, atol=0.02)


@pytest.mark.parametrize("device", DEVICES)
def test_auto_falls_back_for_large_heads(device):
    q = torch.randn(1, 2, 3, 320, device=device)
    k = torch.randn(1, 1, 3, 320, device=device)
    v = torch.randn_like(k)
    actual = strict_causal_attention(q, k, v)
    expected = strict_causal_attention(q, k, v, backend="reference")
    torch.testing.assert_close(actual, expected)


def test_empty_document_metadata():
    metadata = prepare_strict_causal_metadata(torch.empty(2, 0, dtype=torch.long), 0)
    assert metadata.query_indices.numel() == metadata.key_indices.numel() == 0
    assert metadata.cu_queries.tolist() == metadata.cu_keys.tolist() == [0]

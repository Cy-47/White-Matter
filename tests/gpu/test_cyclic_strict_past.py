"""No-dummy cyclic kernels preserve strict-past outputs and gradients."""

import pytest
import torch

from white_matter.blocks._execution.metadata import prepare_feedback_metadata
from white_matter.ops import cyclic_attention

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("stride", [1, 4])
@pytest.mark.parametrize("dummy", [False, True])
def test_cyclic_strict_past_cuda(packed, stride, dummy):
    torch.manual_seed(71)
    length = 137
    docs = torch.arange(length, device="cuda")[None].expand(2, -1).clone() // 11
    docs[1] = torch.arange(length, device="cuda") // 3
    groups = [torch.arange(r, length, stride, device="cuda") for r in range(stride)]
    metadata = prepare_feedback_metadata(docs, groups, length, use_dummy_token=dummy) if packed else None
    for residue, slots in enumerate(groups):
        q = torch.randn(2, 4, slots.numel(), 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(2, 2, length, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        options = {
            "query_stride": stride,
            "query_offset": residue,
            "strict_past": not dummy,
            "metadata": None if metadata is None else metadata[residue],
        }
        actual = cyclic_attention(q, k, v, backend="tilelang", **options)
        expected = cyclic_attention(q, k, v, backend="reference", **options)
        torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.025)
        probe = torch.randn_like(actual)
        ga = torch.autograd.grad((actual * probe).sum(), (q, k, v), retain_graph=True)
        ge = torch.autograd.grad((expected * probe).sum(), (q, k, v))
        for a, e in zip(ga, ge, strict=True):
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a, e, rtol=0.06, atol=0.04)


@pytest.mark.parametrize("mode", ["autoregressive", "jacobi", "cyclic"])
@pytest.mark.parametrize("dummy", [False, True])
@torch.no_grad()
def test_model_prefill_and_decode_cuda(mode, dummy):
    from white_matter import WhiteMatterConfig, WhiteMatterForCausalLM

    config = WhiteMatterConfig(
        vocab_size=101,
        eos_token_id=100,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_kv_channels=2,
        use_dummy_token=dummy,
        execution_mode=mode,
        prefill_mode=mode,
        document_separator_token_id=None,
        cyclic_groups=2,
        num_passes=3,
    )
    config._attn_implementation = "flash_attention_2"
    model = WhiteMatterForCausalLM(config).cuda().eval()
    ids = torch.randint(1, 100, (2, 9), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prefix = model(ids[:, :5], use_cache=True)
        result = model(ids[:, 5:8], past_key_values=prefix.past_key_values, use_cache=True)
        result = model(ids[:, 8:], past_key_values=result.past_key_values, use_cache=True)
    assert torch.isfinite(result.logits).all()
    assert result.past_key_values.layers[0].keys.shape[-2] == 9 + int(dummy)


@pytest.mark.parametrize("dummy", [False, True])
@torch.no_grad()
def test_decode_fullgraph(dummy):
    from white_matter.ops import strict_causal_attention

    q = torch.randn(2, 4, 1, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 2, 17, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lengths = torch.tensor([0, 11], dtype=torch.int32, device="cuda")
    options = {"query_start": lengths, "kv_lengths": lengths, "use_dummy_token": dummy, "backend": "flash_attention_2"}
    expected = strict_causal_attention(q, k, v, **options)
    compiled = torch.compile(strict_causal_attention, fullgraph=True)
    torch.testing.assert_close(compiled(q, k, v, **options), expected)

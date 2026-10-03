"""Compiled defaults, visible eager opt-outs, and scoped compiler settings."""

import argparse

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from training.compile import _packed_ar_backend, compile_evaluation
from white_matter.compilation import add_compile_argument, execution_policy
from white_matter.ops import strict_causal_attention
from white_matter.ops.flash_attention import can_use_torch_flash, torch_flash_decode


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_flash_capability_check_inside_nested_region(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    @torch.compiler.nested_compile_region
    def region(x):
        query = x.sin()
        return query + int(can_use_torch_flash(query, query, query))

    def forward(x):
        return region(x.cos())

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    x = torch.randn(2, 1, 2, 8, dtype=dtype, device=device)
    compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), forward(x))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("static_cache", [False, True])
@pytest.mark.parametrize("dim", [8, 63])
@torch.inference_mode()
def test_native_decode_cache_reuses_graphs(device, batch, strided, static_cache, dim, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.compiler.reset()
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def native_reference(q, k, v, cu_q, cu_k, max_q, max_k, dropout, causal, debug, *, scale, seqused_k):
        slots = torch.arange(max_k, device=k.device)
        indices = cu_k[:-1, None] + slots
        keep = slots[None, :] < seqused_k[:, None]
        output = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(2),
            k[indices].transpose(1, 2),
            v[indices].transpose(1, 2),
            attn_mask=keep[:, None, None, :],
            scale=scale,
            enable_gqa=True,
        )
        return (output.squeeze(2),)

    # Initialize compiler dependencies before replacing an ATen operator.
    from torch._dynamo.backends import debugging  # noqa: F401

    if device == "cpu":
        # Exercise the native packing and compiler path without a CUDA kernel.
        monkeypatch.setattr(torch.ops.aten, "_flash_attention_forward", native_reference)
    compiled = torch.compile(torch_flash_decode, backend=backend, dynamic=True, fullgraph=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    q = torch.randn(batch, 4, 1, dim, device=device, dtype=dtype)
    with execution_policy():
        for length in range(5, 17):
            capacity = 20 if static_cache else length
            k, v = (
                torch.randn(batch, 4 if strided else 2, capacity, dim, device=device, dtype=dtype) for _ in range(2)
            )
            if strided:
                k, v = k[:, :2], v[:, :2]
            lengths = length - torch.arange(batch, device=device, dtype=torch.int32)
            keep = torch.arange(capacity, device=device)[None, :] < lengths[:, None]
            expected = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=keep[:, None, None, :], scale=0.5, enable_gqa=True
            ).transpose(1, 2)
            actual = compiled(q, k, v, lengths, 0.5, static_cache=static_cache)
            torch.testing.assert_close(actual, expected, atol=0.03 if device == "cuda" else 1e-6, rtol=0.03)
    assert len(graphs) <= 2
    if static_cache:
        for graph in graphs:
            assert sum(node.target is torch.arange for node in graph.graph.nodes) == int(device == "cpu")
    torch.compiler.reset()


@pytest.mark.parametrize("name", ["--compile", "--compiled"])
def test_cli_compiles_by_default(name):
    parser = argparse.ArgumentParser()
    add_compile_argument(parser, name)
    key = name.removeprefix("--")
    assert vars(parser.parse_args([]))[key] is True
    assert vars(parser.parse_args(["--no-" + key]))[key] is False


def test_eager_warning_and_stance_are_scoped():
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    @torch.compile(backend=backend)
    def f(x):
        return x.sin() + 1

    with pytest.warns(RuntimeWarning, match="Eager execution selected.*internal helpers"), execution_policy(False):
        torch.testing.assert_close(f(torch.zeros(2)), torch.ones(2))
    assert not graphs
    with execution_policy():
        torch.testing.assert_close(f(torch.zeros(2)), torch.ones(2))
    assert len(graphs) == 1


def test_packed_ar_backend_warns_and_preserves_gradients():
    x = torch.randn(4, requires_grad=True)
    compiled = torch.compile(lambda x: x.sin(), backend=_packed_ar_backend)
    with pytest.warns(RuntimeWarning, match="aot_eager.*tensor kernels remain eager"):
        compiled(x).sum().backward()
    torch.testing.assert_close(x.grad, x.detach().cos())


def test_compiler_errors_propagate_and_settings_restore():
    def broken_backend(*args):
        raise RuntimeError("compiler failure")

    @torch.compile(backend=broken_backend)
    def f(x):
        return x + 1

    with torch._dynamo.config.patch(suppress_errors=True, fail_on_recompile_limit_hit=False):
        with execution_policy():
            assert torch._dynamo.config.fail_on_recompile_limit_hit
        with pytest.raises(Exception, match="compiler failure"), execution_policy():
            f(torch.ones(2))
        assert torch._dynamo.config.suppress_errors
        assert not torch._dynamo.config.fail_on_recompile_limit_hit


@torch.inference_mode()
def test_growing_reference_cache_reuses_compiled_graphs():
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    @torch.compile(backend=backend, dynamic=True, fullgraph=True)
    def attend(q, k, v):
        return strict_causal_attention(q, k, v, query_start=k.shape[-2], backend="reference")

    q = torch.randn(2, 4, 1, 8)
    with execution_policy():
        for length in range(5, 17):
            k, v = torch.randn(2, 2, length, 8), torch.randn(2, 2, length, 8)
            expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True).transpose(1, 2)
            torch.testing.assert_close(attend(q, k, v), expected)
    assert len(graphs) <= 2


@pytest.mark.parametrize(
    ("family", "cached"),
    [
        (family, cached)
        for family in ("vanilla", "white_matter", "lckv", "feedback_transformer")
        for cached in (False, True)
    ]
    + [("fusedkv", False)],
)
@torch.inference_mode()
def test_evaluation_compiles_tensor_work_and_preserves_outputs(family, cached, monkeypatch):
    torch.compiler.reset()
    config = AutoConfig.for_model(
        family,
        vocab_size=31,
        eos_token_id=30,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_kv_channels=2,
        num_passes=2,
        cyclic_groups=2,
        document_separator_token_id=None,
        prefill_mode="autoregressive",
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(ids, use_cache=cached).logits
    compile_fn = torch.compile
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def capture(fn, **kwargs):
        return compile_fn(fn, backend=backend, dynamic=kwargs.get("dynamic"), fullgraph=kwargs.get("fullgraph", False))

    monkeypatch.setattr(torch, "compile", capture)
    with execution_policy():
        compile_evaluation(model)
        actual = model(ids, use_cache=cached).logits
    torch.testing.assert_close(actual, expected)
    assert graphs
    torch.compiler.reset()

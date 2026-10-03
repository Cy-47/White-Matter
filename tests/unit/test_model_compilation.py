"""Complete model capture, repeated pass reuse, and cached evaluation regression."""

import copy

import pytest
import torch
from torch._dynamo.backends.registry import lookup_backend
from transformers import AutoConfig, AutoModelForCausalLM

from training.compile import compile_evaluation, compile_training_forward
from training.forward import TrainingForward
from white_matter.compilation import execution_policy


def tiny_model(family, **kwargs):
    config = AutoConfig.for_model(
        family,
        vocab_size=31,
        eos_token_id=30,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=kwargs.pop("num_hidden_layers", 2),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_kv_channels=2,
        num_passes=4,
        cyclic_groups=2,
        document_separator_token_id=None,
        **kwargs,
    )
    return AutoModelForCausalLM.from_config(config)


@pytest.fixture(autouse=True)
def reset_compiler():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.parametrize("family", ["vanilla", "fusedkv", "white_matter", "lckv"])
@torch.inference_mode()
def test_complete_forward_is_one_graph(family):
    model = tiny_model(family).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(ids, use_cache=False).logits
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return lookup_backend("aot_eager")(graph, inputs)

    model.compile(backend=backend, fullgraph=True)
    with execution_policy():
        actual = model(ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
    assert len(graphs) == 1
    # The captured entry includes both token embedding and vocabulary projection.
    code = graphs[0].code
    assert "embedding" in code
    assert "logits" in code
    assert "torch._C._nn.linear" in code
    assert all(getattr(layer, "_compiled_call_impl", None) is None for layer in model.model.decoder.modules())


@pytest.mark.parametrize("family", ["white_matter", "lckv"])
@pytest.mark.parametrize("gradient_passes", [2, 4])
def test_jacobi_capture_preserves_all_gradients(family, gradient_passes):
    model = tiny_model(family, execution_mode="jacobi").train()
    reference = copy.deepcopy(model)
    ids = torch.tensor([[1, 2, 3, 4]])
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return lookup_backend("aot_eager")(graph, inputs)

    expected = TrainingForward(reference)(None, 4, gradient_passes, token_ids=ids)
    expected.backward()
    runner = torch.compile(TrainingForward(model), backend=backend, fullgraph=True)
    with execution_policy():
        actual = runner(None, 4, gradient_passes, token_ids=ids)
        actual.backward()
    torch.testing.assert_close(actual, expected)
    for (name, param), (_, ref) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        assert (param.grad is None) == (ref.grad is None), name
        if param.grad is not None:
            torch.testing.assert_close(param.grad, ref.grad, atol=1e-6, rtol=1e-4, msg=name)
    assert len(graphs) == 1
    regions = [node for node in graphs[0].graph.nodes if "invoke_subgraph" in str(node.target)]
    assert not regions  # Jacobi training is compiled inline for BF16 gradient parity.


@pytest.mark.parametrize("family", ["white_matter", "lckv"])
@torch.inference_mode()
def test_jacobi_inference_reuses_pass_regions(family):
    model = tiny_model(family, execution_mode="jacobi").eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(ids, use_cache=False).logits
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return lookup_backend("aot_eager")(graph, inputs)

    model.compile(backend=backend, fullgraph=True)
    with execution_policy():
        actual = model(ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
    assert len(graphs) == 1
    regions = [node for node in graphs[0].graph.nodes if "invoke_subgraph" in str(node.target)]
    assert len(regions) == 4
    assert len({node.args[1] for node in regions}) < len(regions)


@torch.inference_mode()
def test_sixteen_layer_cached_evaluation_survives_prefill_decode_and_reset(monkeypatch):
    model = tiny_model("vanilla", num_hidden_layers=16).eval()
    reference = copy.deepcopy(model)
    original_compile = torch.compile

    def capture(fn, **kwargs):
        return original_compile(fn, backend="aot_eager", dynamic=kwargs.get("dynamic"))

    monkeypatch.setattr(torch, "compile", capture)
    with execution_policy():
        compile_evaluation(model)
        for _ in range(2):
            actual_cache = expected_cache = None
            for tokens in ([[1, 2, 3, 4]], [[5]], [[6]]):
                ids = torch.tensor(tokens)
                actual = model(ids, use_cache=True, past_key_values=actual_cache)
                expected = reference(ids, use_cache=True, past_key_values=expected_cache)
                torch.testing.assert_close(actual.logits, expected.logits)
                actual_cache, expected_cache = actual.past_key_values, expected.past_key_values


def test_training_entry_captures_embeddings_and_preserves_their_gradients(monkeypatch):
    model = tiny_model("vanilla").train()
    reference = copy.deepcopy(model)
    ids = torch.tensor([[1, 2, 3, 4]])
    original_compile = torch.compile
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return lookup_backend("aot_eager")(graph, inputs)

    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: original_compile(fn, backend=backend, fullgraph=True))
    runner = compile_training_forward(TrainingForward(model))
    with execution_policy():
        actual = runner(None, 1, 1, token_ids=ids)
        actual.backward()
    expected = TrainingForward(reference)(reference.model.embed_tokens(ids), 1, 1, token_ids=ids)
    expected.backward()
    torch.testing.assert_close(actual, expected)
    assert len(graphs) == 1
    assert "embedding" in graphs[0].code
    for (name, param), (_, ref) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        torch.testing.assert_close(param.grad, ref.grad, atol=1e-6, rtol=1e-4, msg=name)


@pytest.mark.parametrize(
    ("family", "mode"), [("white_matter", "jacobi"), ("white_matter", "cyclic"), ("lckv", "jacobi")]
)
@pytest.mark.parametrize("packed", [False, True])
@torch.inference_mode()
def test_eager_pass_loop_reuses_graphs_across_long_sweeps(family, mode, packed, monkeypatch):
    model = tiny_model(family, execution_mode=mode).eval()
    reference = copy.deepcopy(model)
    ids = torch.tensor([[1, 2, 3, 4]])
    documents = torch.tensor([[0, 0, 1, 1]]) if packed else None
    original_compile = torch.compile
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def capture(fn, **kwargs):
        kwargs.pop("options", None)
        return original_compile(fn, backend=backend, **kwargs)

    monkeypatch.setattr(torch, "compile", capture)
    with execution_policy(), torch._dynamo.config.patch(recompile_limit=8):
        compile_evaluation(model, eager_pass_loop=True)
        for passes in (1, 2):
            model.model(ids, num_passes=passes, document_ids=documents, use_cache=False)
        count = len(graphs)
        assert count > 0
        assert any("invoke_subgraph" in graph.code or "scaled_dot_product_attention" in graph.code for graph in graphs)
        for passes in (*range(3, 13), 65, 96, 128):
            actual = model.model(ids, num_passes=passes, document_ids=documents, use_cache=False).last_hidden_state
            expected = reference.model(
                ids, num_passes=passes, document_ids=documents, use_cache=False
            ).last_hidden_state
            torch.testing.assert_close(actual, expected)
            assert len(graphs) == count


@pytest.mark.parametrize("family", ["white_matter", "lckv", "feedback_transformer"])
@torch.inference_mode()
def test_ar_prefill_keeps_loop_eager_and_reuses_tensor_graphs(family):
    model = tiny_model(family, prefill_mode="autoregressive").eval()
    reference = copy.deepcopy(model)
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    model.compile(backend=backend, dynamic=True)
    with execution_policy():
        for length in (4, 8, 16):
            ids = torch.arange(1, length + 1)[None]
            actual = model(ids, use_cache=True)
            expected = reference(ids, use_cache=True)
            torch.testing.assert_close(actual.logits, expected.logits)
            token = torch.tensor([[17]])
            actual = model(token, use_cache=True, past_key_values=actual.past_key_values)
            expected = reference(token, use_cache=True, past_key_values=expected.past_key_values)
            torch.testing.assert_close(actual.logits, expected.logits)
            if length == 8:
                count = len(graphs)
            elif length == 16:
                assert len(graphs) == count
    assert any("scaled_dot_product_attention" in graph.code for graph in graphs)


@pytest.mark.parametrize("family", ["white_matter", "feedback_transformer"])
@pytest.mark.parametrize("packed", [False, True])
def test_ar_eager_loop_preserves_outputs_and_all_gradients(family, packed):
    model = tiny_model(family, execution_mode="autoregressive").train()
    reference = copy.deepcopy(model)
    documents = torch.tensor([[0, 0, 1, 1]]) if packed else None
    actual_input = torch.randn(1, 4, 16, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_(True)
    model.compile(backend="aot_eager", dynamic=True)
    with execution_policy():
        actual = model(inputs_embeds=actual_input, document_ids=documents, use_cache=False).logits
        actual.square().mean().backward()
    expected = reference(inputs_embeds=expected_input, document_ids=documents, use_cache=False).logits
    expected.square().mean().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_input.grad, expected_input.grad)
    for (name, param), (_, ref) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        assert (param.grad is None) == (ref.grad is None), name
        if param.grad is not None:
            torch.testing.assert_close(param.grad, ref.grad, atol=1e-6, rtol=1e-4, msg=name)

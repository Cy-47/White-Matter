"""Model Inductor capture and cached continuation with eager AR scheduling."""

import copy

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from training.compile import compile_evaluation
from white_matter.compilation import execution_policy

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


@pytest.mark.parametrize(
    ("family", "mode"),
    [
        ("vanilla", "cyclic"),
        ("fusedkv", "cyclic"),
        ("white_matter", "cyclic"),
        ("white_matter", "jacobi"),
        ("white_matter", "autoregressive"),
        ("lckv", "jacobi"),
        ("feedback_transformer", "autoregressive"),
    ],
)
@torch.inference_mode()
def test_compiled_model_matches_eager_and_cached_continuation(family, mode):
    torch.compiler.reset()
    torch.manual_seed(71)
    config = AutoConfig.for_model(
        family,
        vocab_size=127,
        eos_token_id=126,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=16 if family == "vanilla" else 4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_kv_channels=2,
        num_passes=3,
        cyclic_groups=2,
        execution_mode=mode,
        prefill_mode=mode if family in {"white_matter", "lckv"} else "autoregressive",
        document_separator_token_id=None,
    )
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config).cuda().eval()
    reference = copy.deepcopy(model)
    ids = torch.randint(0, 126, (2, 8), device="cuda")
    model.compile(fullgraph=family in {"vanilla", "fusedkv"}, options={"emulate_precision_casts": True})
    cached = family != "fusedkv"
    with execution_policy():
        actual = model(ids, use_cache=cached)
        expected = reference(ids, use_cache=cached)
        torch.testing.assert_close(actual.logits, expected.logits, atol=0.03, rtol=0.03)
        if cached:
            for _ in range(2):
                token = torch.randint(0, 126, (2, 1), device="cuda")
                actual = model(token, use_cache=True, past_key_values=actual.past_key_values)
                expected = reference(token, use_cache=True, past_key_values=expected.past_key_values)
                torch.testing.assert_close(actual.logits, expected.logits, atol=0.03, rtol=0.03)
    torch.compiler.reset()


@pytest.mark.parametrize("mode", ["cyclic", "jacobi"])
@pytest.mark.parametrize("packed", [False, True])
@torch.inference_mode()
def test_compiled_passes_with_eager_scheduling_match_reference(mode, packed, monkeypatch):
    torch.compiler.reset()
    torch.manual_seed(71)
    config = AutoConfig.for_model(
        "white_matter",
        vocab_size=127,
        eos_token_id=126,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_kv_channels=2,
        cyclic_groups=2,
        execution_mode=mode,
        document_separator_token_id=None,
    )
    model = AutoModelForCausalLM.from_config(config).cuda().eval()
    reference = copy.deepcopy(model)
    ids = torch.randint(0, 126, (2, 8), device="cuda")
    documents = (torch.arange(8, device="cuda") // 4).expand(2, -1) if packed else None
    original_compile = torch.compile
    graphs = []

    def backend(graph, inputs, **kwargs):
        graphs.append(graph)
        return torch._inductor.compile(graph, inputs, **kwargs)

    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: original_compile(fn, backend=backend, **kwargs))
    with execution_policy(), torch._dynamo.config.patch(recompile_limit=8):
        compile_evaluation(model, eager_pass_loop=True)
        for passes in (1, 2):
            model.model(ids, num_passes=passes, document_ids=documents, use_cache=False)
        count = len(graphs)
        assert count > 0
        for passes in (*range(3, 13), 65, 96, 128):
            actual = model.model(ids, num_passes=passes, document_ids=documents, use_cache=False).last_hidden_state
            expected = reference.model(
                ids, num_passes=passes, document_ids=documents, use_cache=False
            ).last_hidden_state
            torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
            assert len(graphs) == count
    torch.compiler.reset()

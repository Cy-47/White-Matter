"""Compiled three-pass held-out scoring agrees with a direct CE reference."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from studies.shared_mixture.evaluate_heldout import evaluate_three_pass
from studies.shared_mixture.model import SharedMixtureConfig, register_model
from training.compile import compile_evaluation
from training.precision import attention_kernel_context, configure_precision
from white_matter.compilation import execution_policy
from white_matter.modules.precision import model_autocast_context

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


def test_compiled_packed_three_pass_score_matches_direct_cross_entropy(monkeypatch):
    pytest.importorskip("tilelang")
    register_model()
    configure_precision("cuda")
    torch.manual_seed(825)
    cfg = SharedMixtureConfig(
        vocab_size=257,
        hidden_size=192,
        intermediate_size=384,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=96,
        eos_token_id=256,
        document_separator_token_id=256,
        num_kv_channels=4,
        num_passes=3,
        cyclic_groups=8,
        router_layer_stride=2,
    )
    cfg._attn_implementation = "flash_attention_2"
    optimized = AutoModelForCausalLM.from_config(cfg).cuda().eval()
    reference = copy.deepcopy(optimized)
    captured_decoder = []
    original_compile = torch.compile

    def backend(graph, inputs, **kwargs):
        captured_decoder.append(any("kv_pool" in node.name for node in graph.graph.nodes))
        return torch._inductor.compile(graph, inputs, **kwargs)

    def capture(fn=None, **kwargs):
        kwargs["backend"] = backend
        return original_compile(fn, **kwargs)

    monkeypatch.setattr(torch, "compile", capture)
    ids = torch.randint(0, 256, (2, 32), device="cuda")
    ids[0, [7, 20]] = 256
    ids[1, [11, 24]] = 256
    loader = DataLoader([{"input_ids": row.cpu()} for row in ids], batch_size=2)
    with execution_policy():
        compile_evaluation(optimized)
        loss_sum, targets = evaluate_three_pass(optimized, loader)
    assert any(captured_decoder), "evaluation must compile the shared-mixture decoder"
    assert targets == 62
    with torch.inference_mode(), attention_kernel_context("cuda"), model_autocast_context("cuda"):
        logits = reference(ids, num_passes=3).logits[:, :-1]
        expected = F.cross_entropy(logits.float().reshape(-1, 257), ids[:, 1:].reshape(-1))
    assert loss_sum / targets == pytest.approx(float(expected), abs=5e-3)

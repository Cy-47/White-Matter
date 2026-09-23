"""Compiled training compared with an explicit eager reference."""

import copy
import logging

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from tests.numerics import assert_close, assert_gradient_maps_close
from training.compile import compile_feedback, compile_training_forward
from training.forward import TrainingForward
from training.losses import checkpointed_linear_cross_entropy, lm_cross_entropy_from_hidden
from training.precision import configure_precision
from white_matter.modules.documents import document_ids_from_eos
from white_matter.modules.precision import model_autocast_context

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


@pytest.mark.parametrize("architecture", ["white_matter", "autoregressive", "lckv", "vanilla", "fusedkv"])
def test_compiled_training_preserves_loss_and_all_gradients(architecture):
    configure_precision("cuda")
    torch.manual_seed(71)
    exact_ar = architecture == "autoregressive"
    extra = (
        dict(num_kv_channels=2, cyclic_groups=4, num_passes=3, router_layer_stride=2)
        if architecture in {"white_matter", "autoregressive"}
        else dict(num_kv_channels=1, num_passes=3, num_pre_layers=1, num_post_layers=1)
        if architecture == "lckv"
        else dict(num_kv_channels=None)
    )
    config = AutoConfig.for_model(
        "white_matter" if exact_ar else architecture,
        **({"execution_mode": "autoregressive"} if exact_ar else {}),
        vocab_size=257,
        hidden_size=192,
        intermediate_size=384,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=96,
        max_position_embeddings=256,
        eos_token_id=256, document_separator_token_id=256,
        **extra,
    )
    config._attn_implementation = "flash_attention_2"
    optimized = AutoModelForCausalLM.from_config(config).cuda().train()
    reference = copy.deepcopy(optimized)
    compile_feedback(optimized, mode="default", ar_dynamic=exact_ar)
    ids = torch.randint(0, 256, (2, 32 if exact_ar else 128), device="cuda")
    ids[0, [6, 19]] = 256
    ids[1, [3, 23]] = 256
    segments = document_ids_from_eos(ids, 256)
    wrapper = compile_training_forward(
        TrainingForward(optimized, checkpoint_chunk_size=16 if exact_ar else 0, external_ce=exact_ar)
    )

    def run(model, compiled):
        inputs = model.get_input_embeddings()(ids)
        inputs.retain_grad()
        with model_autocast_context("cuda"):
            if compiled:
                output = wrapper(
                    inputs,
                    1 if architecture in {"vanilla", "fusedkv"} else 3,
                    1 if architecture in {"vanilla", "fusedkv"} else 2,
                    token_ids=ids,
                    compute_ce=False,
                ).detach()
                value = wrapper(
                    inputs,
                    1 if architecture in {"vanilla", "fusedkv"} else 3,
                    1 if architecture in {"vanilla", "fusedkv"} else 2,
                    token_ids=ids,
                )
                loss = checkpointed_linear_cross_entropy(value, ids, model.lm_head) if exact_ar else value
            else:
                decoder = model.model.decoder
                if exact_ar:
                    hidden = decoder.block.forward_autoregressive(inputs, document_ids=segments, checkpoint_chunk_size=0)
                elif architecture == "white_matter":
                    hidden, _ = decoder.block.forward(
                        inputs, num_passes=3, num_gradient_passes=2, cyclic_groups=4, document_ids=segments
                    )
                elif architecture == "lckv":
                    hidden = decoder(inputs, num_passes=3, num_gradient_passes=2, document_ids=segments)
                else:
                    hidden = decoder(inputs, document_ids=segments)
                loss = lm_cross_entropy_from_hidden(
                    ids, hidden=hidden, final_norm=model.model.norm, lm_head=model.lm_head
                )
                output = hidden.detach()
        loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                gradients[name] = parameter.grad.detach().clone()
        return output, loss.detach(), inputs.grad.detach(), gradients

    expected = run(reference, False)
    actual = run(optimized, True)
    assert_close(actual[0], expected[0], rtol=5e-2, atol=2e-2, aggregate_rtol=3e-2, cosine=0.9995)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-3, atol=2e-3)
    assert_close(actual[2], expected[2], rtol=5e-2, atol=2e-4, aggregate_rtol=3e-2, cosine=0.9995)
    assert_gradient_maps_close(
        actual[3], expected[3], rtol=5e-2, atol=2e-4, aggregate_rtol=3e-2, cosine=0.9995,
    )


@pytest.mark.parametrize("residual_dtype", ["fp32", "bf16"])
def test_ar_checkpoint_compilation_limit_preserves_training(monkeypatch, caplog, residual_dtype):
    """Keep backward compiled and match limited-compile reference numerics."""
    configure_precision("cuda")
    torch.manual_seed(71)
    config = AutoConfig.for_model(
        "white_matter",
        execution_mode="autoregressive",
        residual_dtype=residual_dtype,
        vocab_size=257,
        eos_token_id=256, document_separator_token_id=256,
        hidden_size=192,
        intermediate_size=384,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=96,
        max_position_embeddings=256,
        num_kv_channels=2,
        cyclic_groups=4,
        num_passes=3,
        router_layer_stride=2,
    )
    config._attn_implementation = "flash_attention_2"
    initial = AutoModelForCausalLM.from_config(config).cuda().train()
    batches = [torch.randint(0, 256, (2, length), device="cuda") for length in (32, 48)]
    for step, ids in enumerate(batches):
        ids[0, [6 + step, 19]] = 256
        ids[1, [3, 23 + step]] = 256

    native_compile = torch.compile

    def limited_compile(*args, **kwargs):
        if kwargs.get("backend") == "aot_eager":
            kwargs.pop("recompile_limit", None)
        return native_compile(*args, **kwargs)

    def run(reference):
        torch.compiler.reset()
        model = copy.deepcopy(initial)
        with monkeypatch.context() as patch:
            if reference:
                patch.setattr(torch, "compile", limited_compile)
            compile_feedback(model, mode="default", ar_dynamic=True)
            forward = compile_training_forward(
                TrainingForward(model, checkpoint_chunk_size=16, external_ce=True)
            )
            records = []
            for ids in batches:
                model.zero_grad(set_to_none=True)
                inputs = model.get_input_embeddings()(ids)
                inputs.retain_grad()
                with model_autocast_context("cuda"):
                    hidden = forward(inputs, 1, 1, token_ids=ids, compute_ce=False).detach()
                    value = forward(inputs, 1, 1, token_ids=ids)
                    loss = checkpointed_linear_cross_entropy(value, ids, model.lm_head, token_chunk_size=16)
                loss.backward()
                gradients = {}
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        assert parameter.grad is not None, name
                        gradients[name] = parameter.grad.detach().cpu().clone()
                assert inputs.grad is not None
                records.append((hidden.cpu(), value.detach().cpu(), loss.detach().cpu(), inputs.grad.cpu(), gradients))
            return records

    # Attach directly: PyTorch's logger need not propagate to pytest's root
    # handler, and the fallback warning originates on an autograd worker.
    logger = logging.getLogger("torch._dynamo.convert_frame")
    logger.addHandler(caplog.handler)
    try:
        expected = run(True)
        assert any("hit config.recompile_limit" in record.getMessage() for record in caplog.records), (
            "reference did not reproduce the backward compilation limit"
        )
        caplog.clear()
        actual = run(False)
        assert not any("hit config.recompile_limit" in record.getMessage() for record in caplog.records), (
            "checkpoint backward fell back to eager after exhausting its compilation limit"
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        logger.removeHandler(caplog.handler)
        torch.compiler.reset()

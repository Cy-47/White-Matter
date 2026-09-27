"""Production native autocast versus an independent, cache-disabled reference.

Both sides use production kernels, pass compilation, loss, and optimizers.
The reference uses explicit scoped and global cache overrides.
"""

import copy
from contextlib import contextmanager, nullcontext

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from training.compile import compile_feedback
from training.forward import TrainingForward
from training.losses import checkpointed_linear_cross_entropy
from training.optim import (
    build_optimizers,
    clip_grad_norm_if_needed_,
    step_optimizers,
)
from training.precision import configure_precision
from white_matter.blocks._execution import cyclic, jacobi

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


def _reference_autocast(device):
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    return nullcontext()


@contextmanager
def _reference_cast_policy(monkeypatch):
    previous_cache = torch.is_autocast_cache_enabled()
    with monkeypatch.context() as patch:
        patch.setattr(cyclic, "model_autocast_context", _reference_autocast)
        patch.setattr(jacobi, "model_autocast_context", _reference_autocast)
        torch.set_autocast_cache_enabled(False)
        try:
            yield
        finally:
            torch.set_autocast_cache_enabled(previous_cache)


@pytest.mark.parametrize("architecture", ["white_matter", "lckv", "autoregressive"])
@pytest.mark.parametrize("residual_dtype", ["fp32", "bf16"])
def test_native_cast_cache_matches_reference_training(monkeypatch, architecture, residual_dtype):
    # Require the production backend; a fallback does not validate this policy.
    import flash_attn  # noqa: F401
    import tilelang  # noqa: F401

    previous_cache = torch.is_autocast_cache_enabled()
    configure_precision("cuda")
    assert torch.is_autocast_cache_enabled() == previous_cache
    torch.manual_seed(94)
    exact_ar = architecture == "autoregressive"
    white_matter = architecture != "lckv"
    config = AutoConfig.for_model(
        "white_matter" if exact_ar else architecture,
        **({"execution_mode": "autoregressive"} if exact_ar else {}),
        residual_dtype=residual_dtype,
        vocab_size=257,
        hidden_size=192,
        intermediate_size=384,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=96,
        max_position_embeddings=256,
        eos_token_id=256,
        document_separator_token_id=256,
        num_kv_channels=2 if white_matter else 1,
        cyclic_groups=4 if white_matter else None,
        num_passes=4,
        router_layer_stride=2 if white_matter else None,
        num_pre_layers=0 if white_matter else 1,
        num_post_layers=0 if white_matter else 1,
    )
    config._attn_implementation = "flash_attention_2"
    initial = AutoModelForCausalLM.from_config(config).cuda().train()
    batches = [torch.randint(0, 256, (2, 32 if exact_ar else 128), device="cuda") for _ in range(3)]
    for step, ids in enumerate(batches):
        ids[0, [5 + step, 15, 27] if exact_ar else [5 + step, 31, 77]] = 256
        ids[1, [2, 14 + step, 30] if exact_ar else [2, 46 + step, 111]] = 256

    def run(native_cache):
        with nullcontext() if native_cache else _reference_cast_policy(monkeypatch):
            torch.compiler.reset()
            model = copy.deepcopy(initial)
            optimizers = build_optimizers(
                model,
                base_lr=3e-4,
                weight_decay=0.1,
                adam_beta1=0.9,
                adam_beta2=0.95,
                muon_momentum=0.95,
                muon_ns_steps=5,
                device="cuda",
            )
            compile_feedback(model, mode="default", ar_dynamic=exact_ar)
            forward = torch.compile(
                TrainingForward(model, checkpoint_chunk_size=16 if exact_ar else 0, external_ce=exact_ar),
                fullgraph=False,
                dynamic=False,
            )
            records = []
            # Exercise a curriculum change, zero/multiple detached passes, and
            # one/multiple gradient passes. Each step changes document metadata.
            schedule = [(1, 1)] * 3 if exact_ar else [(4, 2), (3, 1), (2, 2)]
            for ids, (passes, gradient_passes) in zip(batches, schedule, strict=True):
                inputs = model.get_input_embeddings()(ids)
                inputs.retain_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
                    # Compare decoder outputs as well as the actual production loss.
                    hidden = forward(inputs, passes, gradient_passes, token_ids=ids, compute_ce=False).detach()
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
                    value = forward(inputs, passes, gradient_passes, token_ids=ids)
                    loss = (
                        checkpointed_linear_cross_entropy(value, ids, model.lm_head, token_chunk_size=16)
                        if exact_ar
                        else value
                    )
                loss.backward()
                gradients = {}
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        assert parameter.grad is not None, name
                        gradients[name] = parameter.grad.detach().cpu().clone()
                assert inputs.grad is not None
                input_gradient = inputs.grad.detach().cpu().clone()
                clip_grad_norm_if_needed_(optimizers.parameters, 1.0)
                step_optimizers(optimizers)
                records.append(
                    (
                        hidden.cpu(),
                        loss.detach().cpu(),
                        input_gradient,
                        gradients,
                        {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad},
                    )
                )
            return records

    try:
        expected = run(False)
        actual = run(True)
        for reference, candidate in zip(expected, actual, strict=True):
            torch.testing.assert_close(candidate[0], reference[0], rtol=2e-3, atol=2e-3)
            torch.testing.assert_close(candidate[1], reference[1], rtol=2e-3, atol=2e-3)
            torch.testing.assert_close(candidate[2], reference[2], rtol=5e-2, atol=2e-4)
            torch.testing.assert_close(candidate[3], reference[3], rtol=5e-2, atol=2e-4)
            torch.testing.assert_close(candidate[4], reference[4], rtol=2e-4, atol=2e-5)
    finally:
        torch.set_autocast_cache_enabled(previous_cache)
        torch.compiler.reset()


@pytest.mark.parametrize("schedule", ["cyclic", "jacobi", "lckv"])
def test_standalone_rollout_preserves_bf16_without_outer_autocast(monkeypatch, schedule):
    from white_matter.blocks import FeedbackDecoderLayer, LCKVBlock, WhiteMatterBlock
    from white_matter.layers import WhiteMatterAttention
    from white_matter.modules import GatedMLP, KVPool, RotaryEmbedding
    from white_matter.modules.routing import FixedSourceMixer

    torch.manual_seed(95)
    lckv = schedule == "lckv"
    layers = [
        FeedbackDecoderLayer(192, WhiteMatterAttention(192, 2, 96, strict_causal=lckv), GatedMLP(192, 384))
        for i in range(2)
    ]
    pool = KVPool(192, 1, 96, 2, 1, **({"mixer": FixedSourceMixer(2)} if lckv else {}))
    block_class = LCKVBlock if lckv else WhiteMatterBlock
    initial = block_class(layers, pool, RotaryEmbedding(96)).cuda().train()
    inputs = torch.randn(2, 32, 192, device="cuda")
    documents = torch.arange(32, device="cuda").expand(2, -1) // 7

    def run(reference):
        with _reference_cast_policy(monkeypatch) if reference else nullcontext():
            block = copy.deepcopy(initial)
            detached_dtypes = []

            def record_detached_dtype(module, args, output):
                if not torch.is_grad_enabled():
                    detached_dtypes.append(output.dtype)

            block.layers[0].self_attn.q_proj.register_forward_hook(record_detached_dtype)
            x = inputs.clone().requires_grad_(True)
            forward = block.forward_jacobi if schedule == "jacobi" else block.forward
            output = forward(x, num_passes=3, num_gradient_passes=2, document_ids=documents)
            if schedule == "cyclic":
                output, _ = output
            assert detached_dtypes, "detached pass was not observed"
            assert set(detached_dtypes) == {torch.bfloat16}, f"detached pass dtypes: {detached_dtypes}"
            output.square().mean().backward()
            gradients = {name: p.grad for name, p in block.named_parameters()}
            assert all(gradient is not None for gradient in gradients.values())
            return output.detach(), x.grad, gradients

    expected = run(True)
    actual = run(False)
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2e-4)

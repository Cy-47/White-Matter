from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoModelForCausalLM

from tests.numerics import assert_gradient_maps_close
from training.precision import configure_precision
from white_matter import WhiteMatterForCausalLM
from white_matter.blocks.white_matter import WhiteMatterBlock
from white_matter.modules.precision import model_autocast_context

pytestmark = pytest.mark.gpu


def _block() -> WhiteMatterBlock:
    from white_matter.blocks import FeedbackDecoderLayer
    from white_matter.layers import WhiteMatterAttention
    from white_matter.modules import GatedMLP, KVPool, RotaryEmbedding

    torch.manual_seed(11)
    layers = [FeedbackDecoderLayer(192, WhiteMatterAttention(192, 2, 96), GatedMLP(192, 384)) for i in range(4)]
    pool = KVPool(192, 1, 96, 5, 2, router_prior="cyclic:0.25", router_layer_stride=2)
    return WhiteMatterBlock(layers, pool, RotaryEmbedding(96, 10_000.0), num_passes=3).cuda().train()


def _run(block, x_source, document_ids, probe, *, tilelang: bool, compiled: bool = False):
    for layer in block.layers:
        layer.self_attn._force_cyclic_reference = not tilelang
    block.zero_grad(set_to_none=True)
    x = x_source.detach().clone().requires_grad_(True)
    forward = block.forward
    if compiled:
        forward = torch.compile(forward, mode="default", fullgraph=False, dynamic=False)
    with model_autocast_context("cuda"):
        output, _ = forward(
            x,
            num_passes=3,
            num_gradient_passes=2,
            cyclic_groups=4,
            document_ids=document_ids,
        )
        loss = (output.float() * probe).mean()
    loss.backward()
    gradients = {}
    for name, parameter in block.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            gradients[name] = parameter.grad.detach().clone()
    return output.detach(), x.grad.detach(), gradients


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tilelang_document_kernel_matches_explicit_sdpa_end_to_end() -> None:
    import tilelang  # noqa: F401

    configure_precision("cuda")
    optimized = _block()
    reference = copy.deepcopy(optimized)
    torch.manual_seed(12)
    x = torch.randn(2, 128, 192, device="cuda")
    probe = torch.randn_like(x)
    document_ids = torch.stack(
        (
            torch.repeat_interleave(torch.arange(4, device="cuda"), torch.tensor([5, 31, 17, 75], device="cuda")),
            torch.repeat_interleave(torch.arange(4, device="cuda"), torch.tensor([33, 7, 54, 34], device="cuda")),
        )
    )
    expected = _run(reference, x, document_ids, probe, tilelang=False)
    actual = _run(optimized, x, document_ids, probe, tilelang=True, compiled=True)

    torch.testing.assert_close(actual[0], expected[0], rtol=5e-2, atol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-1, atol=2e-3)
    assert_gradient_maps_close(
        actual[2],
        expected[2],
        rtol=1e-1,
        atol=2e-3,
        # The dummy token sums every document's dummy-slot gradient. Small
        # per-attention BF16 errors can cross zero elementwise, so also bound
        # every tensor's relative L2, direction, and peak error.
        aggregate_rtol=2e-2,
    )


def _lckv_model(attention_implementation: str) -> WhiteMatterForCausalLM:
    config = AutoConfig.for_model(
        "lckv",
        vocab_size=257,
        hidden_size=192,
        intermediate_size=384,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=96,
        max_position_embeddings=256,
        rope_theta=10_000.0,
        rms_norm_eps=1.0e-6,
        num_kv_channels=1,
        num_passes=3,
        num_pre_layers=1,
        num_post_layers=1,
        eos_token_id=256,
        document_separator_token_id=256,
    )
    config._attn_implementation = attention_implementation
    torch.manual_seed(21)
    return AutoModelForCausalLM.from_config(config).cuda().train()


def _set_attention_implementation(model: torch.nn.Module, implementation: str) -> None:
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None:
            config._attn_implementation = implementation
        if hasattr(module, "attention_implementation"):
            module.attention_implementation = implementation


def _run_lckv(model, inputs, document_ids, probe):
    model.zero_grad(set_to_none=True)
    x = inputs.detach().clone().requires_grad_(True)
    with model_autocast_context("cuda"):
        output = model(inputs_embeds=x, document_ids=document_ids).logits
        loss = (output.float() * probe).mean()
    loss.backward()
    missing = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing, missing
    gradients = {
        name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return output.detach(), x.grad.detach(), gradients


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shifted_segments", [False, True])
def test_lckv_flash_prune_matches_strict_causal_sdpa_end_to_end(shifted_segments) -> None:
    import flash_attn  # noqa: F401

    configure_precision("cuda")
    optimized = _lckv_model("flash_attention_2")
    reference = copy.deepcopy(optimized)
    _set_attention_implementation(reference, "sdpa")

    torch.manual_seed(22)
    inputs = torch.randn(2, 64, 192, device="cuda")
    document_ids = torch.stack(
        (
            torch.repeat_interleave(torch.arange(3, device="cuda"), torch.tensor([7, 19, 38], device="cuda")),
            torch.repeat_interleave(torch.arange(3, device="cuda"), torch.tensor([31, 2, 31], device="cuda")),
        )
    )
    if shifted_segments:
        # A row*T+segment encoding would collide at this batch boundary:
        # row 0's last document and row 1's first document both become 66.
        document_ids = document_ids + torch.tensor([[64], [2]], device="cuda")
    probe = torch.randn(2, 64, 257, device="cuda")
    expected = _run_lckv(reference, inputs, document_ids, probe)
    actual = _run_lckv(optimized, inputs, document_ids, probe)

    torch.testing.assert_close(actual[0], expected[0], rtol=5e-2, atol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-1, atol=2e-3)
    assert_gradient_maps_close(actual[2], expected[2], rtol=1e-1, atol=2e-3, aggregate_rtol=2e-2)


def _nccl_gradient_reduce_worker(rank: int, rendezvous: str) -> None:
    world_size = 2
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from training.distributed import all_reduce_grads
        from training.optim import (
            build_optimizers,
            step_optimizers,
        )

        configure_precision(f"cuda:{rank}")
        torch.manual_seed(31)
        config = AutoConfig.for_model(
            "white_matter",
            vocab_size=101,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            rope_theta=10_000.0,
            rms_norm_eps=1.0e-6,
            num_kv_channels=2,
            num_passes=1,
            cyclic_groups=1,
            router_layer_stride=1,
            router_prior="shifted_identity:0.25",
            eos_token_id=100,
            document_separator_token_id=100,
        )
        config._attn_implementation = "sdpa"
        model = AutoModelForCausalLM.from_config(config).cuda().train()
        for layer in model.model.decoder.block.layers:
            layer.self_attn._force_cyclic_reference = True
        optimizers = build_optimizers(
            model,
            base_lr=3.0e-4,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            muon_momentum=0.95,
            muon_ns_steps=5,
            device=f"cuda:{rank}",
        )

        generator = torch.Generator(device=f"cuda:{rank}").manual_seed(41 + rank)
        inputs = torch.randn(1, 8, 32, generator=generator, device=f"cuda:{rank}")
        inputs.requires_grad_(True)
        document_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]], device=f"cuda:{rank}")
        probe = torch.randn(1, 8, 101, generator=generator, device=f"cuda:{rank}")
        with model_autocast_context(f"cuda:{rank}"):
            output = model(inputs_embeds=inputs, document_ids=document_ids).logits
            (output.float() * probe).mean().backward()

        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        assert all(parameter.grad is not None for parameter in parameters)
        output_before = output.detach().clone()
        input_gradient_before = inputs.grad.detach().clone()
        reference_gradients = [parameter.grad.detach().clone() for parameter in parameters]
        for gradient in reference_gradients:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            gradient.div_(world_size)

        all_reduce_grads(parameters, world_size, validate_presence=True)

        torch.testing.assert_close(output, output_before, rtol=0, atol=0)
        torch.testing.assert_close(inputs.grad, input_gradient_before, rtol=0, atol=0)
        for parameter, reference_gradient in zip(parameters, reference_gradients, strict=True):
            torch.testing.assert_close(parameter.grad, reference_gradient, rtol=0, atol=0)
        step_optimizers(optimizers)
        assert all(torch.isfinite(parameter).all() for parameter in parameters)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_bounded_nccl_gradient_reduce_matches_per_parameter_reference() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        rendezvous = str(Path(temporary_directory) / "nccl_init")
        mp.spawn(
            _nccl_gradient_reduce_worker,
            args=(rendezvous,),
            nprocs=2,
            join=True,
        )

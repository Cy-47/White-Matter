from __future__ import annotations

import copy
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from training.optim import (
    build_optimizers,
    partition_optimizer_parameters,
)
from white_matter import WhiteMatterConfig, WhiteMatterForCausalLM


def tiny_config() -> WhiteMatterConfig:
    return AutoConfig.for_model(
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


def _forward_and_gradients(model: WhiteMatterForCausalLM, embeddings: torch.Tensor):
    model.zero_grad(set_to_none=True)
    inputs = embeddings.detach().clone().requires_grad_(True)
    logits = model(inputs_embeds=inputs, document_ids=torch.tensor([[0, 0, 1, 1]])).logits
    probe = torch.linspace(-1.0, 1.0, logits.numel()).reshape_as(logits)
    (logits * probe).sum().backward()
    missing = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing
    gradients = {
        name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return logits.detach(), inputs.grad.detach(), gradients


def test_hf_wrapper_preserves_output_input_grad_and_every_parameter_grad() -> None:
    torch.manual_seed(11)
    model = AutoModelForCausalLM.from_config(tiny_config()).train()
    reference = copy.deepcopy(model)
    embeddings = torch.randn(1, 4, 32)

    actual = _forward_and_gradients(model, embeddings)
    reference.zero_grad(set_to_none=True)
    reference_inputs = embeddings.detach().clone().requires_grad_(True)
    document_ids = torch.tensor([[0, 0, 1, 1]])
    hidden = reference.model(
        inputs_embeds=reference_inputs,
        document_ids=document_ids,
        num_passes=1,
    )
    reference_logits = reference.lm_head(hidden.last_hidden_state)
    probe = torch.linspace(-1.0, 1.0, reference_logits.numel()).reshape_as(reference_logits)
    (reference_logits * probe).sum().backward()
    reference_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
        if parameter.requires_grad
    }

    torch.testing.assert_close(actual[0], reference_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], reference_inputs.grad, rtol=0, atol=0)
    assert actual[2].keys() == reference_gradients.keys()
    for name in reference_gradients:
        torch.testing.assert_close(actual[2][name], reference_gradients[name], rtol=0, atol=0)


def test_batched_muon_production_step_matches_stock_muon_end_to_end() -> None:
    """Gate the only maintained Muon optimization on the real model partition."""
    torch.manual_seed(19)
    optimized_model = AutoModelForCausalLM.from_config(tiny_config()).train()
    reference_model = copy.deepcopy(optimized_model)
    optimizer_args = {
        "base_lr": 3.0e-4,
        "weight_decay": 0.1,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "muon_momentum": 0.95,
        "muon_ns_steps": 5,
        "device": "cpu",
    }
    optimized = build_optimizers(optimized_model, **optimizer_args)

    reference_partition = partition_optimizer_parameters(reference_model)
    reference_decay, reference_nodecay, reference_muon = reference_partition
    reference_adamw = torch.optim.AdamW(
        [
            {"params": reference_decay, "weight_decay": 0.1},
            {"params": reference_nodecay, "weight_decay": 0.0},
        ],
        lr=3.0e-4,
        betas=(0.9, 0.95),
    )
    reference_muon_optimizer = torch.optim.Muon(
        reference_muon,
        lr=3.0e-4,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.1,
        ns_steps=5,
        adjust_lr_fn="match_rms_adamw",
    )

    embeddings = torch.randn(1, 4, 32)
    actual = _forward_and_gradients(optimized_model, embeddings)
    expected = _forward_and_gradients(reference_model, embeddings)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert actual[2].keys() == expected[2].keys()
    for name in expected[2]:
        torch.testing.assert_close(actual[2][name], expected[2][name], rtol=0, atol=0)

    optimized.adamw.step()
    assert optimized.muon is not None
    optimized.muon.step()
    reference_adamw.step()
    reference_muon_optimizer.step()
    for (actual_name, actual_parameter), (expected_name, expected_parameter) in zip(
        optimized_model.named_parameters(),
        reference_model.named_parameters(),
        strict=True,
    ):
        assert actual_name == expected_name
        torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)


def test_save_pretrained_safetensors_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(13)
    model = AutoModelForCausalLM.from_config(tiny_config()).eval()
    # Noninitial weights catch loading hooks that accidentally reset learned parameters.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    input_ids = torch.tensor([[1, 2, 100, 3]])
    expected = model(input_ids).logits
    model.save_pretrained(tmp_path, safe_serialization=True)
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "model.safetensors").is_file()

    loaded = WhiteMatterForCausalLM.from_pretrained(tmp_path).eval()
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], parameter, rtol=0, atol=0, msg=name)
    actual = loaded(input_ids).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert loaded.lm_head.weight is loaded.get_input_embeddings().weight

    auto_config = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(auto_config, WhiteMatterConfig)
    auto_model = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(auto_model(input_ids).logits, expected, rtol=0, atol=0)

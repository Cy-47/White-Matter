"""Clipping before accumulation must preserve the production update."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from training.forward import TrainingForward
from training.optim import build_optimizers, step_optimizers


def test_microbatch_clipping_accumulation_and_update_match_native_reference():
    import copy

    from training.optim import clip_grad_norm_if_needed_

    torch.manual_seed(59)
    config = AutoConfig.for_model(
        "white_matter",
        vocab_size=101,
        eos_token_id=100, document_separator_token_id=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_kv_channels=2,
        cyclic_groups=2,
        num_passes=3,
    )
    initial = AutoModelForCausalLM.from_config(config)
    batches = [torch.randint(0, 100, (1, 8)) for _ in range(3)]
    for ids in batches:
        ids[:, 3] = 100

    def run(native):
        model = copy.deepcopy(initial)
        parameters = list(model.parameters())
        records, clipped, raw = [], [], []
        for ids, scale in zip(batches, (1.0, 50.0, 0.25), strict=True):
            model.zero_grad(set_to_none=True)
            inputs = model.get_input_embeddings()(ids)
            inputs.retain_grad()
            loss = TrainingForward(model)(inputs, 3, 2, token_ids=ids) * scale
            loss.backward()
            records.append((loss.detach(), inputs.grad.clone(), [p.grad.clone() for p in parameters]))
            raw.append(torch.cat([p.grad.flatten().clone() for p in parameters]))
            clip = torch.nn.utils.clip_grad_norm_ if native else clip_grad_norm_if_needed_
            clip(parameters, 0.1)
            clipped.append([p.grad.clone() for p in parameters])
        for i, parameter in enumerate(parameters):
            parameter.grad = sum(batch[i] for batch in clipped) / len(clipped)
        # This fixture must distinguish clipping before averaging from clipping after averaging.
        averaged_raw = torch.stack(raw).mean(0)
        clipped_mean = averaged_raw * min(1.0, 0.1 / (averaged_raw.norm().item() + 1e-6))
        actual_mean = torch.cat([p.grad.flatten() for p in parameters])
        assert not torch.allclose(actual_mean, clipped_mean, rtol=1e-3, atol=1e-6)
        clip(parameters, 0.1)
        accumulated = [p.grad.clone() for p in parameters]
        opts = build_optimizers(
            model,
            base_lr=3e-4,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            muon_momentum=0.95,
            muon_ns_steps=5,
            device="cpu",
        )
        step_optimizers(opts)
        return records, accumulated, [p.detach().clone() for p in parameters]

    torch.testing.assert_close(run(False), run(True), rtol=0, atol=0)

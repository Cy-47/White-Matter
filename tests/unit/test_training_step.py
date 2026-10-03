"""The shared production step preserves complete packed-model gradients and updates."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from training.compile import configure_training_compilation
from training.forward import TrainingForward
from training.losses import cce_linear_cross_entropy
from training.optim import (
    BatchedMuon,
    CompiledBatchedMuon,
    build_optimizers,
    clip_grad_norm_if_needed_,
    step_optimizers,
)
from training.precision import attention_kernel_context, configure_precision
from training.recipes import load_recipe
from training.step import prepare_training_forward, training_gradients
from white_matter.modules.precision import cast_residual, model_autocast_context


def small_recipe(family="white_matter"):
    recipe = load_recipe(f"recipes/paper/{family}_1p3b.yaml")
    config = AutoConfig.for_model(
        family,
        vocab_size=257,
        hidden_size=192,
        intermediate_size=384,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=96,
        num_kv_channels=2,
        num_passes=3,
        cyclic_groups=4,
        router_layer_stride=2,
        eos_token_id=256,
        document_separator_token_id=256,
        residual_dtype="bf16",
    )
    return replace(recipe, model=config, data=replace(recipe.data, sequence_length=32, eos_token_id=256))


@pytest.mark.parametrize("optimizer_class", [BatchedMuon, CompiledBatchedMuon])
def test_batched_muon_deepcopy_preserves_step(optimizer_class):
    parameters = [torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2)]
    optimizer = copy.deepcopy(optimizer_class(parameters, lr=1e-3))
    for parameter in optimizer.param_groups[0]["params"]:
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("loss_backend", ["torch", "cce"])
def test_shared_step_matches_reference(device, accumulation, loss_backend, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if loss_backend == "cce" and device == "cpu":
        pytest.skip("CCE requires CUDA")
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    rank = torch.distributed.get_rank() if world > 1 else 0
    compiled = device == "cuda" and world == 1
    recipe = replace(small_recipe(), gradient_accumulation_steps=accumulation, loss_backend=loss_backend)
    recipe.model._attn_implementation = "flash_attention_2" if device == "cuda" else "sdpa"
    configure_precision(device)
    torch.manual_seed(1337)
    initial = AutoModelForCausalLM.from_config(recipe.model).to(device)
    batches = [torch.randint(0, 255, (1, 32), device=device) for _ in range(accumulation)]
    for i, ids in enumerate(batches):
        ids.add_(rank).remainder_(255)
        ids[:, 5 + i :: 9] = 256

    # CCE's lock-ordered reductions vary across repeated reference executions.
    # Reuse one real CCE VJP per microbatch, checking every operand exactly, so
    # the complete decoder/optimizer refactor retains a bitwise acceptance gate.
    cce_records, cce_index, recording = [], [0], [True]

    class RecordedCCE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight, record):
            ctx.save_for_backward(record[4], record[5])
            return record[3].clone()

        @staticmethod
        def backward(ctx, gradient):
            dh, dw = ctx.saved_tensors
            return dh * gradient, dw * gradient, None

    def fixed_cce(hidden, ids, head):
        if recording[0]:
            h = hidden.detach().requires_grad_()
            w = head.weight.detach().requires_grad_()
            value = cce_linear_cross_entropy(h, ids, SimpleNamespace(weight=w))
            dh, dw = torch.autograd.grad(value, (h, w))
            cce_records.append((h.detach().clone(), w.detach().clone(), ids.clone(), value.detach(), dh, dw))
        record = cce_records[cce_index[0]]
        torch.testing.assert_close((hidden, head.weight, ids), record[:3], rtol=0, atol=0)
        cce_index[0] += 1
        return RecordedCCE.apply(hidden, head.weight, record)

    if loss_backend == "cce":
        monkeypatch.setattr("training.step.cce_linear_cross_entropy", fixed_cce)

    def run(shared):
        cce_index[0], recording[0] = 0, not shared
        model = copy.deepcopy(initial)
        opt = recipe.optimizer
        optimizers = build_optimizers(
            model,
            base_lr=opt.learning_rate,
            weight_decay=opt.weight_decay,
            adam_beta1=opt.adam_beta1,
            adam_beta2=opt.adam_beta2,
            muon_momentum=opt.muon_momentum,
            muon_ns_steps=opt.muon_ns_steps,
            device=device,
            distributed_muon=world > 1,
        )
        if shared:
            runner = prepare_training_forward(model, recipe, 1, compiled=compiled)
        else:
            if compiled:
                configure_training_compilation(model)
            runner = TrainingForward(model, external_ce=loss_backend == "cce")
            if compiled:
                runner = torch.compile(
                    runner, options={"emulate_precision_casts": True}, fullgraph=False, dynamic=False
                )
        inputs, outputs, raw_gradients = [], [], []

        def embedding_hook(module, args, output):
            output.retain_grad()
            inputs.append(output)

        model.get_input_embeddings().register_forward_hook(embedding_hook)

        # Observe the same compiled runner without inserting hooks into its graph.
        class Observed(torch.nn.Module):
            external_ce = loss_backend == "cce"

            def forward(self, *args, **kwargs):
                value = runner(*args, **kwargs)
                outputs.append(value.detach().clone())
                return value

        observed = Observed()
        for parameter in model.parameters():
            parameter.register_hook(lambda grad: raw_gradients.append(grad.detach().clone()))
        records = []
        for _ in range(2):
            if shared:
                loss, _ = training_gradients(
                    model, observed, optimizers, batches, recipe, world=world, check_gradients=True
                )
            else:
                clipped, losses = [], []
                for ids in batches:
                    model.zero_grad(set_to_none=True)
                    with attention_kernel_context(device), model_autocast_context(device):
                        hidden = cast_residual(
                            model.get_input_embeddings()(ids), residual_dtype=recipe.model.residual_dtype
                        )
                        value = observed(hidden, 3, 2, token_ids=ids)
                        loss = fixed_cce(value, ids, model.lm_head) if loss_backend == "cce" else value
                    loss.backward()
                    clip_grad_norm_if_needed_(optimizers.parameters, opt.max_gradient_norm)
                    clipped.append([p.grad.clone() for p in optimizers.parameters])
                    losses.append(loss.detach().double())
                if accumulation > 1:
                    for i, p in enumerate(optimizers.parameters):
                        p.grad = sum((batch[i] for batch in clipped[1:]), clipped[0][i]) * (1 / accumulation)
                if world > 1:
                    for p in optimizers.parameters:
                        torch.distributed.all_reduce(p.grad)
                        p.grad.div_(world)
                if accumulation > 1 or world > 1:
                    clip_grad_norm_if_needed_(optimizers.parameters, opt.max_gradient_norm)
                mean_loss = torch.stack(losses).sum() / accumulation
                if world > 1:
                    torch.distributed.all_reduce(mean_loss, op=torch.distributed.ReduceOp.AVG)
                loss = float(mean_loss)
            records.append((loss, [p.grad.clone() for p in optimizers.parameters]))
            step_optimizers(optimizers)
            records.append([p.detach().clone() for p in model.parameters()])
        return records, outputs, [x.grad for x in inputs], raw_gradients

    expected = run(False)
    torch.testing.assert_close(run(True), expected, rtol=0, atol=0)
    if loss_backend == "cce":
        assert cce_index[0] == len(cce_records) == 2 * accumulation

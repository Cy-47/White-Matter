"""Optional optimizations must preserve the complete feedback training gradient."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from training.compile import configure_training_compilation
from training.forward import TrainingForward
from training.losses import cce_linear_cross_entropy, checkpointed_linear_cross_entropy
from training.optim import build_optimizers, set_learning_rate, step_optimizers
from training.precision import attention_kernel_context, configure_precision
from white_matter.modules.precision import cast_residual

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


def make_model(*, mode="cyclic", residual_dtype="fp32"):
    config = AutoConfig.for_model(
        "white_matter",
        vocab_size=257,
        eos_token_id=256,
        document_separator_token_id=256,
        hidden_size=192,
        intermediate_size=384,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=96,
        num_kv_channels=2,
        cyclic_groups=4,
        num_passes=3,
        router_layer_stride=2,
        execution_mode=mode,
        residual_dtype=residual_dtype,
    )
    config._attn_implementation = "flash_attention_2"
    return AutoModelForCausalLM.from_config(config).cuda().train()


def make_optimizers(model, *, distributed=False, compiled=True):
    return build_optimizers(
        model,
        base_lr=3e-4,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        muon_momentum=0.95,
        muon_ns_steps=5,
        device="cuda",
        distributed_muon=distributed,
        compiled=compiled,
    )


def training_step(model, runner, ids, *, cce=False, checkpointed=False):
    model.zero_grad(set_to_none=True)
    inputs = model.get_input_embeddings()(ids)
    inputs.retain_grad()
    with attention_kernel_context("cuda"), torch.autocast("cuda", dtype=torch.bfloat16):
        decoder_inputs = cast_residual(inputs, residual_dtype=model.config.residual_dtype)
        hidden = runner(decoder_inputs, 3, 2, token_ids=ids, compute_ce=False).detach().clone()
        value = runner(decoder_inputs, 3, 2, token_ids=ids)
        loss = (
            cce_linear_cross_entropy(value, ids, model.lm_head)
            if cce
            else checkpointed_linear_cross_entropy(value, ids, model.lm_head, token_chunk_size=16)
            if checkpointed
            else value
        )
    loss.backward()
    record = {"hidden": hidden, "loss": loss.detach().clone(), "input_grad": inputs.grad.detach().clone()}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        record[name] = parameter.grad.detach().clone()
    assert all(torch.isfinite(t).all() for t in record.values())
    return record


def assert_training_close(actual, expected, *, exact=False):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        reference = expected[name]
        if exact:
            torch.testing.assert_close(value, reference, rtol=0, atol=0, msg=name)
        else:
            # Compare each gradient tensor, including near-zero/cancelling router entries.
            delta, ref = (value.double() - reference.double()), reference.double()
            assert delta.norm() <= 0.025 * ref.norm() + 1e-7, (name, (delta.norm() / ref.norm()).item())
            assert delta.abs().max() <= 0.05 * ref.abs().max() + 1e-6, name


def test_compiled_optimizer_preserves_training():
    configure_precision("cuda")
    torch.manual_seed(104)
    models = [make_model(residual_dtype="bf16") for _ in range(2)]
    models[1].load_state_dict(models[0].state_dict())
    optimizers = [make_optimizers(model, compiled=bool(i)) for i, model in enumerate(models)]
    runners = []
    for model in models:
        configure_training_compilation(model)
        runners.append(torch.compile(TrainingForward(model), fullgraph=False, dynamic=False))
    for step in range(3):
        ids = torch.randint(0, 256, (2, 128), device="cuda")
        ids[0, 3 + step :: 19] = ids[1, 7 + step :: 23] = 256
        records = [training_step(model, runner, ids) for model, runner in zip(models, runners, strict=True)]
        assert_training_close(records[1], records[0], exact=True)
        for optimizer in optimizers:
            set_learning_rate(
                optimizer, step=step + 1, total_steps=20, schedule="cosine", warmup_frac=0.1, floor_frac=0.1
            )
            # Changing LR must not compile a new optimizer graph each update.
            with torch.compiler.set_stance("fail_on_recompile" if step else "default"):
                step_optimizers(optimizer)
        torch.testing.assert_close(list(models[1].parameters()), list(models[0].parameters()), rtol=0, atol=0)
        for name in ("muon", "adamw"):
            actual, expected = [getattr(optimizer, name).state_dict() for optimizer in optimizers[::-1]]
            assert actual["param_groups"] == expected["param_groups"]
            torch.testing.assert_close(actual["state"], expected["state"], rtol=0, atol=0)


@pytest.mark.parametrize("residual_dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("optimization", ["cce", "ar_graph", "ar_cce", "ar_checkpointed_cce"])
def test_optimizations_preserve_training(optimization, residual_dtype):
    cce = optimization != "ar_graph"
    if cce:
        pytest.importorskip("cut_cross_entropy")
    configure_precision("cuda")
    torch.manual_seed(104)
    graph = optimization in {"ar_graph", "ar_cce"}
    # Isolate capture from loss precision: graph-on/off use the same criterion.
    reference_cce = graph and cce
    checkpointed = optimization == "ar_checkpointed_cce"
    mode = "autoregressive" if graph or checkpointed else "cyclic"
    reference = make_model(mode=mode, residual_dtype=residual_dtype)
    candidate = make_model(mode=mode, residual_dtype=residual_dtype)
    candidate.load_state_dict(reference.state_dict())
    length = 17 if checkpointed else 16 if graph else 128
    batches = [torch.randint(0, 256, (2, length), device="cuda") for _ in range(3)]
    for step, ids in enumerate(batches):
        ids[0, [3 + step, length - 3]] = 256
        ids[1, [1, length // 2 + step]] = 256
    runners = []
    for model in (reference, candidate):
        configure_training_compilation(model, ar_dynamic=graph or checkpointed)
        runner = TrainingForward(
            model,
            checkpoint_chunk_size=16 if checkpointed else 0,
            external_ce=checkpointed or reference_cce or (cce and model is candidate),
        )
        if graph and model is candidate:
            dtype = torch.bfloat16 if residual_dtype == "bf16" else torch.float32
            with attention_kernel_context("cuda"):
                runner.capture_ar_graph(torch.zeros(2, length, 192, device="cuda", dtype=dtype))
        runners.append(torch.compile(runner, fullgraph=False, dynamic=False))
    optimizers = make_optimizers(reference)
    try:
        for ids in batches:
            if graph:
                # Large post-capture changes expose accidentally cached BF16 weights.
                with torch.no_grad():
                    reference.model.decoder.block.layers[0].self_attn.q_proj.weight.add_(0.01)
            # Identical weights isolate the kernel error from trajectory divergence.
            candidate.load_state_dict(reference.state_dict())
            expected = training_step(reference, runners[0], ids, cce=reference_cce, checkpointed=checkpointed)
            actual = training_step(candidate, runners[1], ids, cce=cce)
            assert_training_close(actual, expected)
            # Exercise updated weights without comparing diverging Adam trajectories:
            # loss rounding can reverse the sign of a near-zero gradient.
            step_optimizers(optimizers)
        if graph:
            # A shorter batch must take the ordinary decoder path.
            ids = batches[0][:1, :7]
            candidate.load_state_dict(reference.state_dict())
            assert_training_close(
                training_step(candidate, runners[1], ids, cce=cce),
                training_step(reference, runners[0], ids, cce=reference_cce),
            )
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cce_shift_and_gradients_match_fp64(bias, dtype):
    pytest.importorskip("cut_cross_entropy")
    torch.manual_seed(44)
    hidden = torch.randn(2, 7, 64, device="cuda", dtype=dtype, requires_grad=True)
    head = torch.nn.Linear(64, 131, bias=bias, device="cuda", dtype=dtype)
    ids = torch.randint(0, 131, (2, 7), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = cce_linear_cross_entropy(hidden, ids, head)
    inputs = (hidden, *head.parameters())
    # Evaluate exactly the same BF16 operands in FP64, without rounding gradients
    # through an autograd-tracked BF16 cast on the way back to FP32 masters.
    oracle = [t.detach().bfloat16().double().requires_grad_() for t in inputs]
    expected = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(oracle[0][:, :-1], oracle[1], oracle[2] if bias else None).flatten(0, 1),
        ids[:, 1:].flatten(),
    )
    gradients = torch.autograd.grad(actual, inputs)
    reference = torch.autograd.grad(expected, oracle)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-4, atol=1e-5)
    for a, b in zip(gradients, reference, strict=True):
        torch.testing.assert_close(a.double(), b, rtol=1e-2, atol=5e-4)

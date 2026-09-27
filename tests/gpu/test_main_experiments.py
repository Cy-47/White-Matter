"""Production backend gates for the paper experiment entry points."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from benchmarks.flops import measure as measure_flops
from studies.prefill_convergence.benchmark import measure as measure_timing
from studies.prefill_convergence.evaluate import measure_curves
from training.recipes import load_recipe

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


def small_recipe(name):
    recipe = load_recipe(Path("recipes/paper") / f"{name}.yaml")
    c = recipe.model
    c.vocab_size = 257
    c.hidden_size = 192
    c.intermediate_size = 384
    c.num_attention_heads = 2
    c.num_key_value_heads = 1
    c.head_dim = 96
    c.eos_token_id = 256
    c.document_separator_token_id = None
    return replace(recipe, data=replace(recipe.data, eos_token_id=256))


@torch.inference_mode()
def test_fp32_reference_and_compiled_bf16_timing():
    recipe = small_recipe("white_matter_k16")
    c = recipe.model
    c.num_hidden_layers = c.num_kv_channels = 4
    model = AutoModelForCausalLM.from_config(c).cuda().eval()
    ids = torch.randint(0, 256, (2, 64), device="cuda")
    observed = []
    hook = model.model.decoder.block.layers[0].self_attn.q_proj.register_forward_hook(
        lambda module, args, output: observed.append(output.dtype),
    )
    curves = measure_curves(model, ids, limits={"jacobi": 3, "cyclic2": 3}, batch_size=1)
    hook.remove()
    assert observed
    assert set(observed) == {torch.float32}
    assert all(torch.isfinite(torch.tensor(v)).all() for v in curves["curves"].values())
    c._attn_implementation = "flash_attention_2"
    for module in model.modules():
        if hasattr(module, "attention_implementation"):
            module.attention_implementation = "flash_attention_2"
    rows = measure_timing(model, ids, {"jacobi": 3, "cyclic2": 3}, warmups=1, repetitions=2, ar_repetitions=2)
    assert set(rows) == {"ar", "jacobi", "cyclic2"}
    assert all(r["median_seconds"] > 0 for r in rows.values())


@pytest.mark.parametrize(
    "name", ["vanilla_16l", "fusedkv", "lckv_w4", "lckv_w7", "white_matter_k8", "white_matter_k16"]
)
def test_flop_counter_measures_actual_backends(name):
    recipe = small_recipe(name)
    # Keep production pass counts, routing, depth, and gradient combination.
    result = measure_flops(recipe, length=64, context=64)
    assert set(result) == {"train", "prefill", "decode"}
    assert all(row["flops"] > 0 for row in result.values())
    assert result["train"]["flops_per_token"] > result["prefill"]["flops_per_token"]
    for row in result.values():
        assert any("flash" in op or "cyclic_attn" in op for op in row["per_op"])

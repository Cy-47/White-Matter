"""LCKV must define zero attention for documents with no earlier tokens."""

import copy

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from white_matter.models import register_models


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("length", [1, 5])
@pytest.mark.parametrize("compiled", [False, True])
def test_all_singleton_documents(length, compiled):
    register_models()
    torch.manual_seed(903)
    config = AutoConfig.for_model(
        "lckv",
        vocab_size=31,
        eos_token_id=30,
        document_separator_token_id=30,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        num_hidden_layers=2,
        num_passes=3,
    )
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_config(config).cuda().train()
    reference = copy.deepcopy(model)
    reference.config._attn_implementation = "sdpa"
    for module in reference.modules():
        if hasattr(module, "attention_implementation"):
            module.attention_implementation = "sdpa"
    if compiled:
        from training.compile import compile_feedback

        compile_feedback(model, mode="default")
        model.compile(options={"emulate_precision_casts": True})
    inputs = torch.randn(2, length, 64, device="cuda")
    documents = torch.arange(length, device="cuda")[None].expand(2, -1)
    records = []
    for current in (reference, model):
        x = inputs.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = current(inputs_embeds=x, document_ids=documents).logits
            loss = output.float().square().mean()
        loss.backward()
        records.append((output, x.grad, {name: p.grad for name, p in current.named_parameters()}))
    assert all(grad is not None for record in records for grad in record[2].values())
    torch.testing.assert_close(*records, rtol=1e-3, atol=1e-5)

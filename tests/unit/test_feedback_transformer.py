"""Independent arithmetic and cache checks for Fan et al. feedback memory."""

import copy

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from white_matter.models import register_models
from white_matter.modules.rotary import rotate_half


def _model():
    register_models()
    config = AutoConfig.for_model(
        "feedback_transformer", vocab_size=97, eos_token_id=96,
        hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=128, rope_theta=10000.0,
        document_separator_token_id=None,
    )
    config._attn_implementation = "sdpa"
    return AutoModelForCausalLM.from_config(config)


def _manual_block(block, x):
    """Plain attention arithmetic, independent of production attention/cache helpers."""
    keys = values = None
    outputs = []
    for t in range(x.shape[1]):
        hidden = x[:, t:t + 1]
        states = [hidden]
        cosine, sine = block.rotary_emb(hidden, torch.full((x.shape[0], 1), t, device=x.device))
        for layer in block.layers:
            if keys is not None:
                attn = layer.self_attn
                query = attn.q_norm(attn.q_proj(layer.input_layernorm(hidden)).view(
                    x.shape[0], 1, 2, 16,
                )).transpose(1, 2)
                query = query * cosine[:, None] + rotate_half(query) * sine[:, None]
                score = query @ keys.repeat_interleave(2, dim=1).transpose(-2, -1) * attn.scaling
                weights = score.float().softmax(-1).to(query.dtype)
                context = weights @ values.repeat_interleave(2, dim=1)
                hidden = hidden + attn.o_proj(context.transpose(1, 2).reshape(x.shape[0], 1, 32))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            states.append(hidden)
        memory = torch.stack(states, dim=0).mul(
            block.memory.layer_logits.softmax(0).view(-1, 1, 1, 1),
        ).sum(0)
        new_key = block.memory.key_norm(block.memory.key(memory).view(x.shape[0], 1, 1, 16)).transpose(1, 2)
        new_key = new_key * cosine[:, None] + rotate_half(new_key) * sine[:, None]
        new_value = block.memory.value(memory).view(x.shape[0], 1, 1, 16).transpose(1, 2)
        keys = new_key if keys is None else torch.cat((keys, new_key), -2)
        values = new_value if values is None else torch.cat((values, new_value), -2)
        outputs.append(hidden)
    return torch.cat(outputs, 1), keys, values


def test_feedback_block_matches_independent_arithmetic_and_gradients():
    torch.manual_seed(12)
    model = _model()
    actual = model.model.decoder.block
    reference = copy.deepcopy(actual)
    # Nonuniform layer weights detect omitted top states and wrong source order.
    with torch.no_grad():
        actual.memory.layer_logits.copy_(torch.tensor([-1.0, 0.2, 1.3, -0.5]))
        reference.memory.layer_logits.copy_(actual.memory.layer_logits)
    inputs = torch.randn(2, 4, 32)
    rows = []
    for block, run in ((actual, actual.forward_reference), (reference, lambda x: _manual_block(reference, x))):
        x = inputs.clone().requires_grad_()
        output, key, value = run(x)
        (output.square().mean() + key.square().mean() + value.square().mean()).backward()
        rows.append((output.detach(), key.detach(), value.detach(), x.grad,
                     {name: p.grad for name, p in block.named_parameters()}))
    for left, right in zip(rows[0][:4], rows[1][:4], strict=True):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)
    for name, gradient in rows[0][4].items():
        torch.testing.assert_close(gradient, rows[1][4][name], rtol=2e-5, atol=2e-6)


def test_cached_prefill_chunks_and_decode_match_full_prefix():
    torch.manual_seed(13)
    model = _model().eval()
    ids = torch.randint(1, 95, (2, 9))
    with torch.no_grad():
        expected = model(ids).logits
        for capacity in (None, 16):
            cache = model.allocate_inference_cache(capacity)
            parts = []
            for start, end in ((0, 1), (1, 4), (4, 7), (7, 9)):
                parts.append(model(ids[:, start:end], past_key_values=cache, use_cache=True).logits)
            torch.testing.assert_close(torch.cat(parts, 1), expected, rtol=1e-5, atol=1e-6)
            assert cache.get_seq_length() == 9
            assert cache.layers[0].keys.shape[-2] == (16 if capacity else 9)


def test_last_logit_prefill_matches_full_output_and_cache():
    torch.manual_seed(16)
    model = _model().eval()
    ids = torch.randint(1, 95, (2, 9))
    with torch.no_grad():
        full_cache = model.allocate_inference_cache(16)
        last_cache = model.allocate_inference_cache(16)
        full = model(ids, past_key_values=full_cache, use_cache=True).logits
        last = model(ids, past_key_values=last_cache, use_cache=True, logits_to_keep=1).logits
    torch.testing.assert_close(last, full[:, -1:])
    for left, right in zip((full_cache.layers[0].keys, full_cache.layers[0].values),
                           (last_cache.layers[0].keys, last_cache.layers[0].values), strict=True):
        torch.testing.assert_close(left, right)
    assert last_cache.get_seq_length() == full_cache.get_seq_length() == 9


@pytest.mark.parametrize("packed", [False, True])
@torch.no_grad()
def test_document_cached_prefill_and_decode_match_independent_documents(packed):
    torch.manual_seed(17)
    model = _model().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    documents = torch.tensor([[0, 0, 1, 1, 1, 2]]) if packed else torch.zeros_like(ids)
    ends = (0, 2, 5, 6) if packed else (0, 6)
    expected = torch.cat([model(ids[:, start:end]).logits for start, end in zip(ends, ends[1:])], dim=1)
    cache = model.allocate_inference_cache()
    actual = torch.cat([
        model(ids[:, start:end], document_ids=documents[:, start:end], past_key_values=cache).logits
        for start, end in ((0, 3), (3, 4), (4, 6))
    ], dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_differentiable_prefix_never_uses_decode_kernel(monkeypatch):
    from white_matter.layers import white_matter as attention_module

    original = attention_module.attention_forward
    calls = []

    def checked_attention(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["cache_seqlens"] is None, "differentiable prefix must not select the decode kernel"
        return original(*args, **kwargs)

    monkeypatch.setattr(attention_module, "attention_forward", checked_attention)
    model = _model().train()
    ids = torch.tensor([[1, 2, 3]])
    model(ids, labels=ids).loss.backward()
    assert calls


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_training_matches_sdpa_outputs_and_all_gradients():
    torch.manual_seed(18)
    reference = _model().cuda().train()
    actual = copy.deepcopy(reference)
    for layer in actual.model.decoder.block.layers:
        layer.self_attn.attention_implementation = "flash_attention_2"
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="cuda")
    results = []
    for model in (reference, actual):
        inputs = model.get_input_embeddings()(ids)
        inputs.retain_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(inputs_embeds=inputs, labels=ids)
        output.loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            gradients[name] = parameter.grad.detach().clone()
        results.append((output.logits.detach(), inputs.grad, gradients))
    torch.testing.assert_close(results[0], results[1], rtol=5e-2, atol=2e-3)


def test_future_tokens_do_not_affect_prefix():
    torch.manual_seed(14)
    model = _model().eval()
    prefix = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        left = model(torch.cat((prefix, torch.tensor([[4, 5]])), 1)).logits[:, :3]
        right = model(torch.cat((prefix, torch.tensor([[6, 7]])), 1)).logits[:, :3]
    torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_hf_roundtrip_preserves_feedback_weights(tmp_path):
    model = _model().eval()
    with torch.no_grad():
        model.model.decoder.block.memory.layer_logits.copy_(torch.tensor([0.1, -0.4, 0.6, 1.1]))
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert loaded.config.model_type == "feedback_transformer"
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids).logits, model(ids).logits, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_flash_cached_prefill_decode_and_graph():
    from white_matter.models.generation import DecodeGraph, prefill

    torch.manual_seed(15)
    model = _model().cuda().eval()
    model.config._attn_implementation = "flash_attention_2"
    for module in model.modules():
        if hasattr(module, "attention_implementation"):
            module.attention_implementation = "flash_attention_2"
    ids = torch.randint(1, 95, (2, 8), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model(ids).logits
        cache = model.allocate_inference_cache(12)
        actual = prefill(model, ids, cache, batch_size=1)
        torch.testing.assert_close(actual, expected[:, -1:], rtol=2e-2, atol=5e-3)
        token = torch.tensor([[5], [6]], device="cuda")
        expected_next = model(torch.cat((ids, token), 1)).logits[:, -1:]
        graph = DecodeGraph(model, cache)
        actual_next = graph(token)
        torch.testing.assert_close(actual_next, expected_next, rtol=2e-2, atol=5e-3)

        model.compile(options={"emulate_precision_casts": True, "reorder_for_locality": False})
        compiled_cache = model.allocate_inference_cache(12)
        compiled_prefill = prefill(model, ids, compiled_cache, batch_size=1)
        torch.testing.assert_close(compiled_prefill, expected[:, -1:], rtol=2e-2, atol=5e-3)
        compiled_next = DecodeGraph(model, compiled_cache)(token)
        torch.testing.assert_close(compiled_next, expected_next, rtol=2e-2, atol=5e-3)

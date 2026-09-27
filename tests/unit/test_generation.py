"""Persistent state correctness, including the real CUDA cyclic attention path."""

import contextlib
import copy

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from white_matter.blocks._execution.cyclic import _prepare_cyclic_groups
from white_matter.blocks._execution.metadata import prepare_feedback_metadata
from white_matter.modules.documents import document_ids_from_eos
from white_matter.modules.precision import model_autocast_context


@pytest.fixture(autouse=True)
def reset_compiler():
    # Cases construct independent architectures; retain reuse within each case.
    torch.compiler.reset()


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda", marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]
        ),
    ]
)
def device(request):
    return request.param


def make_model(device, *, k=2, surrounding=0, prefill="autoregressive", residual_dtype="fp32"):
    torch.manual_seed(73)
    config = AutoConfig.for_model(
        "white_matter",
        vocab_size=101,
        hidden_size=128 if device == "cuda" else 32,
        intermediate_size=192 if device == "cuda" else 64,
        num_hidden_layers=4 + 2 * surrounding,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64 if device == "cuda" else 16,
        max_position_embeddings=128,
        rope_theta=10_000.0,
        eos_token_id=100,
        document_separator_token_id=100,
        pad_token_id=0,
        num_kv_channels=k,
        cyclic_groups=4,
        num_passes=3,
        num_pre_layers=surrounding,
        num_post_layers=surrounding,
        execution_mode=prefill,
        prefill_mode=prefill,
        residual_dtype=residual_dtype,
    )
    config._attn_implementation = "flash_attention_2" if device == "cuda" else "sdpa"
    return AutoModelForCausalLM.from_config(config).to(device).eval()


def close(actual, expected, device):
    # CPU paths should agree to FP32 rounding. CUDA compares different BF16
    # attention/reduction shapes, including flash-varlen versus cached SDPA.
    torch.testing.assert_close(
        actual, expected, rtol=2e-2 if device == "cuda" else 2e-5, atol=5e-3 if device == "cuda" else 3e-6
    )


def close_kv(actual, expected, device):
    if device == "cpu":
        return close(actual, expected, device)
    # Bound eager/compiled BF16 differences per tensor; relative elementwise
    # error near zero is unstable.
    for a, b in zip(actual, expected, strict=True):
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        delta, reference = a.float() - b.float(), b.float()
        assert delta.norm() <= 0.01 * reference.norm() + 1e-6
        assert delta.abs().max() <= 0.02 * reference.abs().max() + 1e-6


@pytest.mark.parametrize("surrounding", [0, 1])
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("residual_dtype", ["fp32", "bf16"])
@torch.no_grad()
def test_exact_ar_continuation_matches_full_reference(device, surrounding, k, residual_dtype):
    model = make_model(device, k=k, surrounding=surrounding, residual_dtype=residual_dtype)
    ids = torch.tensor([[2, 3, 100, 4, 5, 6, 100, 7, 8], [9, 100, 10, 11, 12, 100, 13, 14, 15]], device=device)
    expected = model(ids).logits
    cache, outputs = None, []
    for start, end in ((0, 3), (3, 4), (4, 7), (7, 9)):
        result = model(ids[:, start:end], past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        outputs.append(result.logits)
        assert cache.get_seq_length() == end
    actual = torch.cat(outputs, dim=1)
    if device == "cuda" and residual_dtype == "bf16" and surrounding:
        # Ordinary pre/post layers change GEMM/attention shapes across calls.
        # Bound BF16 rounding separately; full-call caching must remain exact.
        delta = actual.float() - expected.float()
        assert delta.abs().max() < 0.01
        assert delta.norm() / expected.float().norm() < 0.01
        torch.testing.assert_close(model(ids, use_cache=True).logits, expected, rtol=0, atol=0)
    else:
        close(actual, expected, device)
    assert len(cache.layers) == 1 + 2 * surrounding
    assert cache.layers[model.config.num_pre_layers].keys.shape[:2] == (2, k)
    assert cache.layers[model.config.num_pre_layers].keys.shape[-2] == ids.shape[1] + 1


@pytest.mark.parametrize("prefill", ["autoregressive", "cyclic"])
@pytest.mark.parametrize("side", ["left", "right"])
@torch.no_grad()
def test_padded_prefill_and_document_reset_across_calls(device, prefill, side):
    model = make_model(device, surrounding=1, prefill=prefill)
    rows = [torch.tensor([[2, 100, 3]], device=device), torch.tensor([[4, 5, 6, 100, 7]], device=device)]
    ids = torch.zeros((2, 5), device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for b, row in enumerate(rows):
        slots = slice(-row.shape[1], None) if side == "left" else slice(0, row.shape[1])
        ids[b, slots] = row
        mask[b, slots] = 1
    result = model(ids, attention_mask=mask, use_cache=True)
    prefix_cache = result.past_key_values
    continuation = torch.tensor([[8, 100, 9], [10, 100, 11]], device=device)
    actual = model(continuation, past_key_values=prefix_cache, use_cache=True).logits
    for b, row in enumerate(rows):
        reference = model(row, use_cache=True)
        close(result.logits[b, mask[b].bool()], reference.logits[0], device)
        # Exact AR also checks an independent full-prefix run, not just cache splitting.
        expected = (
            model(torch.cat((row, continuation[b : b + 1]), dim=1)).logits[:, -3:]
            if prefill == "autoregressive"
            else model(continuation[b : b + 1], past_key_values=reference.past_key_values, use_cache=True).logits
        )
        close(actual[b : b + 1], expected, device)
        close(actual[b, -1:], model(continuation[b : b + 1, -1:]).logits[0], device)


@pytest.mark.parametrize("k", [1, 2, 4])
@torch.no_grad()
def test_cyclic_export_matches_full_layout_and_continuation(device, k):
    model = make_model(device, k=k, prefill="cyclic")
    ids = torch.tensor([[2, 3, 100, 4, 5, 6, 7], [8, 100, 9, 10, 11, 12, 13]], device=device)
    block = model.model.decoder.block
    docs = document_ids_from_eos(ids, 100)
    with model_autocast_context(device):
        x = model.model.embed_tokens(ids)
        q_rope, k_rope = block._prepare_rope(x, docs)
        schedule = _prepare_cyclic_groups(block, x.shape[1], 4, q_rope, k_rope, x.device, cache_rope=False)
        metadata = prepare_feedback_metadata(docs, schedule[0], x.shape[1])
        K, V = block.kv_pool.project_sequence(
            x.unsqueeze(2).expand(-1, -1, block.num_layers, -1), k_rope, dummy_token=block.dummy_token
        )
        # Independent full-layout rollout retains the last token at every pass.
        for _ in range(3):
            expected, K, V, _ = block.cyclic_pass(
                x, K, V, block.dummy_token.expand(x.shape[0], 1, -1), *schedule, metadata=metadata, consume_state=True
            )
        actual, state = block(x, cyclic_groups=4, document_ids=docs, output_final_state=True)
        close(actual, expected, device)
        close(state, (K, V), device)
        result = model(ids, use_cache=True)
        feedback = result.past_key_values.layers[0]
        close((feedback.keys, feedback.values), (K.to(V.dtype).flatten(1, 2), V.flatten(1, 2)), device)
        next_ids = torch.tensor([[14], [15]], device=device)
        next_x = model.model.embed_tokens(next_ids)
        positions = torch.tensor([[5], [6]], device=device)
        keep = torch.cat((torch.ones_like(docs[:, :1], dtype=torch.bool), docs == docs[:, -1:]), dim=1)
        mask = next_x.new_zeros((2, 1, 1, 8)).masked_fill(~keep[:, None, None], torch.finfo(next_x.dtype).min)
        expected, _, _ = block._run_token_layers(next_x, (K, V), block.rotary_emb(next_x, positions), mask)
        expected = model.lm_head(model.model.norm(expected.to(model.model.norm.weight.dtype)))
        close(model(next_ids, past_key_values=result.past_key_values, use_cache=True).logits, expected, device)


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("packed", [False, True])
def test_short_block_matches_explicit_padding_and_all_gradients(device, k, packed):
    block = make_model(device, k=k, prefill="cyclic").model.decoder.block
    reference = copy.deepcopy(block)
    inputs = torch.randn(2, 3, block.kv_pool.hidden_size, device=device)
    docs = torch.tensor([[0, 0, 1], [0, 1, 1]], device=device) if packed else None
    results = []
    for current, padded in ((block, False), (reference, True)):
        x = inputs.clone().requires_grad_()
        mask = torch.ones((2, 3), device=device, dtype=torch.bool)
        block_x, block_docs = x, docs
        if padded:
            block_x = torch.nn.functional.pad(x, (0, 0, 0, 5))
            mask = torch.nn.functional.pad(mask, (0, 5))
            if docs is not None:
                block_docs = torch.cat((docs, (docs[:, -1:] + 1).expand(-1, 5)), dim=1)
        with model_autocast_context(device):
            hidden, state = current(
                block_x,
                cyclic_groups=4,
                num_passes=3,
                num_gradient_passes=2,
                attention_mask=mask if padded else None,
                document_ids=block_docs,
                output_final_state=True,
            )
            assert state is not None
            if not padded:
                assert hidden.shape[1] == 3
                assert all(t.shape[-2] == 4 for t in state)
            hidden, state = hidden[:, :3], tuple(t[..., :4, :] for t in state)
            loss = hidden.float().square().mean() + sum(t.float().square().mean() for t in state)
        loss.backward()
        gradients = {name: p.grad for name, p in current.named_parameters() if p.requires_grad}
        assert x.grad is not None
        assert all(g is not None and torch.isfinite(g).all() for g in gradients.values())
        results.append((hidden.detach(), tuple(t.detach() for t in state), x.grad, gradients))
    torch.testing.assert_close(*results, rtol=0, atol=0)


@pytest.mark.parametrize("beams", [1, 3])
@torch.no_grad()
def test_hf_generation_reuses_cache_and_reorders_documents(device, beams):
    model = make_model(device, surrounding=1)
    ids = torch.tensor([[2, 100, 3], [4, 5, 6]], device=device)
    options = {"max_new_tokens": 3, "do_sample": False, "num_beams": beams}
    expected = model.generate(ids, use_cache=False, **options)
    calls = []
    handle = model.model.decoder.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[1]))
    try:
        actual = model.generate(ids, use_cache=True, **options)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected)
    assert calls[0] == 3
    assert all(length == 1 for length in calls[1:])


@torch.no_grad()
def test_cache_reorder_repeat_and_reset(device):
    model = make_model(device, surrounding=1)
    ids = torch.tensor([[2, 100, 3], [4, 5, 100]], device=device)
    cache = model(ids, use_cache=True).past_key_values
    cache.batch_repeat_interleave(2)
    cache.batch_select_indices(torch.tensor([3, 0, 2], device=device))
    token = torch.tensor([[6], [7], [8]], device=device)
    actual = model(token, use_cache=True, past_key_values=cache).logits
    prefix = ids[torch.tensor([1, 0, 1], device=device)]
    expected = model(torch.cat((prefix, token), dim=1)).logits[:, -1:]
    close(actual, expected, device)
    cache.reset()
    assert cache.get_seq_length() == 0
    close(model(ids, use_cache=True, past_key_values=cache).logits, model(ids).logits, device)


def test_cyclic_cached_policy_is_explicit():
    model = make_model("cpu", prefill="cyclic")
    model.config.prefill_mode = None
    with torch.no_grad(), pytest.raises(ValueError, match="explicitly"):
        model(torch.tensor([[2, 3]]), use_cache=True)
    model.config.prefill_mode = "autoregressive"
    with pytest.raises(ValueError, match="inference-only"):
        model(torch.tensor([[2, 3]]), use_cache=True)


@torch.no_grad()
def test_generation_accepts_an_existing_cache():
    model = make_model("cpu", surrounding=1)
    ids = torch.tensor([[2, 100, 3, 4]])
    cache = model(ids[:, :2], use_cache=True).past_key_values
    options = {"use_cache": True, "max_new_tokens": 2, "do_sample": False}
    torch.testing.assert_close(model.generate(ids, past_key_values=cache, **options), model.generate(ids, **options))
    with pytest.raises(ValueError, match="single document"):
        model.generate(ids, cache_implementation="static", **options)


@pytest.mark.parametrize("packed", [False, True])
def test_recurrent_state_preserves_full_gradient_history(packed):
    model = make_model("cpu")
    reference = copy.deepcopy(model)
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    docs = torch.tensor([[0, 0, 1, 1, 1]]) if packed else None
    positions = torch.tensor([[1, 2, 1, 2, 3]]) if packed else None
    results = []
    for current, cached in ((reference, False), (model, True)):
        block = current.model.decoder.block
        inputs = current.model.embed_tokens(ids)
        inputs.retain_grad()
        if cached:
            first, state = block.forward_recurrent(
                inputs[:, :3],
                document_ids=docs[:, :3] if packed else None,
                position_ids=positions[:, :3] if packed else None,
            )
            last, state = block.forward_recurrent(
                inputs[:, 3:], initial_state=state, document_ids=docs, position_ids=positions[:, 3:] if packed else None
            )
            hidden = torch.cat((first, last), dim=1)
        else:
            hidden = block.forward_autoregressive(inputs, document_ids=docs)
        logits = current.lm_head(current.model.norm(hidden))
        logits.square().mean().backward()
        results.append((logits.detach(), inputs.grad, {n: p.grad for n, p in current.named_parameters()}))
    torch.testing.assert_close(results[0], results[1], rtol=2e-5, atol=3e-6)


@torch.no_grad()
def test_fp32_cache_continuation_with_ordinary_layers(device, monkeypatch):
    from white_matter.models import modeling_base

    monkeypatch.setattr(modeling_base, "model_autocast_context", lambda device: contextlib.nullcontext())
    model = make_model(device, surrounding=1)
    model.config._attn_implementation = "sdpa"
    for module in model.modules():
        if hasattr(module, "attention_implementation"):
            module.attention_implementation = "sdpa"
    ids = torch.tensor([[2, 3, 100, 4, 5, 6, 100, 7, 8], [9, 100, 10, 11, 12, 100, 13, 14, 15]], device=device)
    expected = model(ids).logits
    cache, outputs = None, []
    for start, end in ((0, 3), (3, 4), (4, 7), (7, 9)):
        result = model(ids[:, start:end], use_cache=True, past_key_values=cache)
        cache = result.past_key_values
        outputs.append(result.logits)
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=2e-5, atol=3e-6)


@torch.no_grad()
def test_cached_embeddings_with_explicit_document_ids(device):
    model = make_model(device, surrounding=1)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]], device=device)
    docs = torch.tensor([[100, 100, 100, 900, 900, 901]], device=device)
    expected = model(input_ids=ids, document_ids=docs).logits
    cache, outputs = None, []
    for start, end in ((0, 2), (2, 4), (4, 6)):
        result = model(
            inputs_embeds=model.model.embed_tokens(ids[:, start:end]),
            document_ids=docs[:, start:end],
            attention_mask=torch.ones_like(ids[:, :end]),
            use_cache=True,
            past_key_values=cache,
        )
        cache = result.past_key_values
        outputs.append(result.logits)
    close(torch.cat(outputs, dim=1), expected, device)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("prefill", ["autoregressive", "cyclic"])
@pytest.mark.parametrize("surrounding", [0, 1])
@pytest.mark.parametrize("masked", [False, True])
@torch.no_grad()
def test_flash_decoding_matches_masked_sdpa_without_expanding_kv(monkeypatch, prefill, surrounding, masked):
    import flash_attn
    import torch.nn.functional as F

    model = make_model("cuda", prefill=prefill, surrounding=surrounding)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15]], device="cuda")
    mask = torch.ones_like(ids)
    if masked:
        ids[0, :2], mask[0, :2] = 0, 0
        ids[1, -2:], mask[1, -2:] = 0, 0
        ids[:, 3] = 100
    cache = model(ids, attention_mask=mask, use_cache=True).past_key_values
    reference_cache = copy.deepcopy(cache)
    reference = copy.deepcopy(model)
    reference.config._attn_implementation = "sdpa"
    for layer in reference.model.decoder.block.layers:
        layer.self_attn.attention_implementation = "sdpa"

    native_flash = flash_attn.flash_attn_with_kvcache
    calls = []

    def flash(query, key, value, **kwargs):
        assert query.shape[1] == 1
        assert key.shape[2] == value.shape[2] == model.config.num_key_value_heads
        calls.append(kwargs["cache_seqlens"].clone())
        return native_flash(query, key, value, **kwargs)

    def unexpected_sdpa(*args, **kwargs):
        pytest.fail("FlashAttention decoding fell back to SDPA")

    continuation = torch.tensor([[16, 100, 17], [18, 19, 20]], device="cuda")
    for token in continuation.split(1, dim=1):
        expected = reference(token, past_key_values=reference_cache, use_cache=True)
        with monkeypatch.context() as patch:
            patch.setattr(flash_attn, "flash_attn_with_kvcache", flash)
            patch.setattr(F, "scaled_dot_product_attention", unexpected_sdpa)
            actual = model(token, past_key_values=cache, use_cache=True)
        close(actual.logits, expected.logits, "cuda")
        for a, b in zip(cache.layers, reference_cache.layers, strict=True):
            close((a.keys, a.values), (b.keys, b.values), "cuda")
    assert len(calls) == 3 * model.config.num_hidden_layers


def inference_model(device, family, surrounding=0, prefill="cyclic", k=2):
    model = make_model(device, surrounding=surrounding, prefill=prefill, k=k)
    if family == "vanilla":
        config = model.config.to_dict()
        config.pop("model_type")
        config = AutoConfig.for_model("vanilla", **config)
        config._attn_implementation = "flash_attention_2" if device == "cuda" else "sdpa"
        model = AutoModelForCausalLM.from_config(config).to(device).eval()
    model.config.document_separator_token_id = None
    return model


@pytest.mark.parametrize("family", ["vanilla", "white_matter"])
@pytest.mark.parametrize("surrounding", [0, 1])
@pytest.mark.parametrize("k", [1, 2, 4])
@torch.no_grad()
def test_static_cache_matches_dynamic_and_preserves_prefix(device, family, surrounding, k, monkeypatch):
    model = inference_model(device, family, surrounding, k=k)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19]], device=device)
    dynamic = model.allocate_inference_cache()
    static = model.allocate_inference_cache(12)
    for start, end in ((0, 5), (5, 6), (6, 9)):
        expected = model(ids[:, start:end], past_key_values=dynamic, use_cache=True)
        with monkeypatch.context() as patch:
            if device == "cuda" and start:
                import flash_attn

                flash = flash_attn.flash_attn_with_kvcache
                owners = {layer.keys.untyped_storage().data_ptr() for layer in static.layers}

                def read_cache(query, key, value, flash=flash, owners=owners, **kwargs):
                    # FA must read native KV views, never a packed/copied prefix.
                    assert key.untyped_storage().data_ptr() in owners
                    assert key.dtype == value.dtype == torch.bfloat16
                    assert key.shape[2] == model.config.num_key_value_heads
                    return flash(query, key, value, **kwargs)

                patch.setattr(flash_attn, "flash_attn_with_kvcache", read_cache)
            actual = model(ids[:, start:end], past_key_values=static, use_cache=True)
        close(actual.logits, expected.logits, device)
        assert static.get_seq_length() == dynamic.get_seq_length() == end
        for a, b in zip(static.layers, dynamic.layers, strict=True):
            close_kv(
                (a.keys[..., : b.keys.shape[-2], :], a.values[..., : b.values.shape[-2], :]), (b.keys, b.values), device
            )
        if start == 0:
            pointers = [(layer.keys.data_ptr(), layer.values.data_ptr()) for layer in static.layers]
            prefix = [
                (layer.keys[..., : 5 + extra, :].clone(), layer.values[..., : 5 + extra, :].clone())
                for layer, extra in zip(static.layers, static.prefix_slots, strict=True)
            ]
            for layer, extra in zip(static.layers, static.prefix_slots, strict=True):
                layer.keys[..., 5 + extra :, :].fill_(float("nan"))
                layer.values[..., 5 + extra :, :].fill_(float("nan"))
    for layer, ptr, (key, value) in zip(static.layers, pointers, prefix, strict=True):
        assert (layer.keys.data_ptr(), layer.values.data_ptr()) == ptr
        torch.testing.assert_close(layer.keys[..., : key.shape[-2], :], key, rtol=0, atol=0)
        torch.testing.assert_close(layer.values[..., : value.shape[-2], :], value, rtol=0, atol=0)
    if family == "vanilla":
        close(model(ids).logits[:, -3:], actual.logits, device)
    with pytest.raises(ValueError, match="capacity"):
        model(ids[:, :4], past_key_values=static, use_cache=True)
    static.reset()
    assert static.get_seq_length() == 0
    close(
        model(ids[:, :5], past_key_values=static, use_cache=True).logits,
        model(ids[:, :5], use_cache=True).logits,
        device,
    )


@pytest.mark.parametrize("family", ["vanilla", "white_matter"])
@torch.no_grad()
def test_static_hf_generation_and_pending_token(device, family):
    model = inference_model(device, family, prefill="autoregressive")
    ids = torch.tensor([[2, 3, 100]], device=device)
    options = {"max_new_tokens": 3, "do_sample": False, "eos_token_id": None, "return_dict_in_generate": True}
    dynamic = model.generate(ids, use_cache=True, **options)
    static = model.generate(ids, use_cache=True, cache_implementation="static", **options)
    torch.testing.assert_close(static.sequences, dynamic.sequences)
    assert static.past_key_values.get_seq_length() == static.sequences.shape[1] - 1
    cache = dynamic.past_key_values
    actual = model(dynamic.sequences[:, -1:], past_key_values=cache, use_cache=True).logits
    close(actual, model(dynamic.sequences).logits[:, -1:], device)
    assert cache.get_seq_length() == dynamic.sequences.shape[1]


@pytest.mark.parametrize("family", ["vanilla", "white_matter"])
@torch.no_grad()
def test_explicit_cached_positions_continue(device, family):
    model = inference_model(device, family, prefill="autoregressive")
    ids = torch.tensor([[2, 3, 4, 5]], device=device)
    positions = torch.arange(7, 11, device=device)[None]
    reference = model(ids, use_cache=True, position_ids=positions).logits
    cache = model.allocate_inference_cache(6)
    first = model(ids[:, :2], use_cache=True, past_key_values=cache, position_ids=positions[:, :2]).logits
    last = model(ids[:, 2:], use_cache=True, past_key_values=cache).logits
    close(torch.cat((first, last), dim=1), reference, device)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("family", ["vanilla", "white_matter"])
@pytest.mark.parametrize("surrounding", [0, 1])
@torch.no_grad()
def test_decode_graph_changes_tokens_lengths_and_cache_contents(family, surrounding):
    from white_matter.models.generation import DecodeGraph

    model = inference_model("cuda", family, surrounding)
    cache = model.allocate_inference_cache(32)
    reference = model.allocate_inference_cache(32)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17]], device="cuda")
    model(ids, use_cache=True, past_key_values=cache)
    graph = DecodeGraph(model, cache)
    for prompt in (ids, ids.flip(1)[:, :5]):
        cache.reset()
        reference.reset()
        model(prompt, use_cache=True, past_key_values=cache)
        model(prompt, use_cache=True, past_key_values=reference)
        for token in (ids[:, :1], ids[:, 3:4], ids[:, 6:7]):
            expected = model(token, past_key_values=reference, use_cache=True, logits_to_keep=1).logits
            actual = graph(token)
            close(actual, expected, "cuda")
            assert cache.get_seq_length() == reference.get_seq_length()
            torch.testing.assert_close(cache.position, reference.position)
            for a, b, extra in zip(cache.layers, reference.layers, cache.prefix_slots, strict=True):
                end = cache.get_seq_length() + extra
                close(
                    (a.keys[..., :end, :], a.values[..., :end, :]),
                    (b.keys[..., :end, :], b.values[..., :end, :]),
                    "cuda",
                )
    cache.batch_select_indices(torch.tensor([1, 0], device="cuda"))
    with pytest.raises(ValueError, match="storage changed"):
        graph(ids[:, :1])


@torch.no_grad()
def test_dynamic_cache_adds_document_metadata_after_plain_prefix():
    model = inference_model("cpu", "white_matter", surrounding=1, prefill="autoregressive")
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    cache = model(ids[:, :2], use_cache=True).past_key_values
    result = model(ids[:, 2:], use_cache=True, past_key_values=cache, document_ids=torch.tensor([[0, 1, 1]]))
    expected = model(ids, document_ids=torch.tensor([[0, 0, 0, 1, 1]])).logits[:, 2:]
    close(result.logits, expected, "cpu")


@pytest.mark.parametrize("prefill", ["cyclic", "jacobi"])
@torch.no_grad()
def test_iterative_prefill_rejects_explicit_positions_without_mutating_cache(prefill):
    model = inference_model("cpu", "white_matter", prefill=prefill)
    cache = model.allocate_inference_cache(8)
    with pytest.raises(ValueError, match="explicit position_ids"):
        model(torch.tensor([[2, 3]]), use_cache=True, past_key_values=cache, position_ids=torch.tensor([[7, 8]]))
    assert cache.seen_tokens == 0
    assert all(not layer.is_initialized for layer in cache.layers)


@torch.no_grad()
def test_generation_preserves_explicit_positions_and_cyclic_policy():
    ids = torch.tensor([[2, 3, 4]])
    positions = torch.tensor([[7, 8, 9]])
    model = inference_model("cpu", "white_matter", prefill="autoregressive")
    first = model(ids, use_cache=True, position_ids=positions).logits[:, -1].argmax(-1)
    generated = model.generate(ids, use_cache=True, max_new_tokens=1, position_ids=positions)
    torch.testing.assert_close(generated[:, -1], first)
    model.config.prefill_mode = "cyclic"
    dynamic = model.generate(ids, use_cache=True, max_new_tokens=3, eos_token_id=None)
    static = model.generate(ids, use_cache=True, max_new_tokens=3, eos_token_id=None, cache_implementation="static")
    torch.testing.assert_close(static, dynamic)


@pytest.mark.parametrize("family", ["vanilla", "white_matter"])
@torch.no_grad()
def test_explicit_positions_with_padding_continue_from_last_valid_token(device, family):
    model = inference_model(device, family, prefill="autoregressive")
    model.config.document_separator_token_id = 100
    ids = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], device=device)
    reference = model(ids, use_cache=True, position_ids=torch.tensor([[7, 8, 9, 10]], device=device)).logits
    padded = torch.cat((ids[:, :2], torch.zeros_like(ids[:, :1])), dim=1)
    first = model(
        padded,
        attention_mask=torch.tensor([[1, 1, 0], [1, 1, 0]], device=device),
        use_cache=True,
        position_ids=torch.tensor([[7, 8, 0]], device=device),
    )
    cache = first.past_key_values
    torch.testing.assert_close(cache.position, torch.full((2, 1), 9, device=device))
    last = model(ids[:, 2:], past_key_values=cache, use_cache=True).logits
    close(torch.cat((first.logits[:, :2], last), dim=1), reference, device)
    torch.testing.assert_close(cache.position, torch.full((2, 1), 11, device=device))


@pytest.mark.parametrize("family", ["white_matter", "vanilla"])
@torch.no_grad()
def test_batched_prefill_shares_cache_and_supports_refill_then_decode(device, family):
    from white_matter.models.generation import DecodeGraph, prefill

    model = inference_model(device, family, surrounding=1)
    # Cross pool-tile boundaries, with ragged groups and a final partial batch.
    length = 521 if device == "cuda" else 17
    ids = (torch.arange(3 * length, device=device).reshape(3, length) % 98) + 1
    cache, reference = (model.allocate_inference_cache(length + 15) for _ in range(2))
    expected = model(ids, past_key_values=reference, use_cache=True, logits_to_keep=1).logits
    close(prefill(model, ids, cache, batch_size=2), expected, device)
    pointers = [(layer.keys.data_ptr(), layer.values.data_ptr()) for layer in cache.layers]
    graph = None
    if device == "cuda":
        model.compile(options={"emulate_precision_casts": True, "reorder_for_locality": False})
        graph = DecodeGraph(model, cache)
    for prompt in (ids, ids.flip(1)[:, :-4]):
        cache.reset()
        reference.reset()
        expected = model(prompt, past_key_values=reference, use_cache=True, logits_to_keep=1).logits
        close(prefill(model, prompt, cache, batch_size=2), expected, device)
        assert cache.get_seq_length() == prompt.shape[1]
        for actual, target, pointer in zip(cache.layers, reference.layers, pointers, strict=True):
            assert (actual.keys.data_ptr(), actual.values.data_ptr()) == pointer
            torch.testing.assert_close(actual.cumulative_length, target.cumulative_length)
            close_kv((actual.keys, actual.values), (target.keys, target.values), device)
        for token in ids[:, :3].split(1, dim=1):
            expected = model(token, past_key_values=reference, use_cache=True, logits_to_keep=1).logits
            actual = (
                graph(token)
                if graph is not None
                else model(token, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
            )
            close(actual, expected, device)
            torch.testing.assert_close(cache.position, reference.position)
            for a, b in zip(cache.layers, reference.layers, strict=True):
                close_kv((a.keys, a.values), (b.keys, b.values), device)


@pytest.mark.parametrize("family", ["white_matter", "vanilla"])
@torch.no_grad()
def test_repeat_prefix_and_position_restore_preserve_storage(family):
    model = inference_model("cpu", family, surrounding=1, prefill="autoregressive")
    ids = torch.tensor([[2, 3, 4]])
    prefix = model.allocate_inference_cache(9)
    model(ids, past_key_values=prefix, use_cache=True)
    cache = prefix.repeat_prefix(3, 7)
    reference = model.allocate_inference_cache(7)
    model(ids.expand(3, -1), past_key_values=reference, use_cache=True)
    pointers = [
        (layer.keys.data_ptr(), layer.values.data_ptr(), layer.cumulative_length.data_ptr()) for layer in cache.layers
    ]
    position_pointer = cache.position.data_ptr()
    snapshot = cache._snapshot_position()
    token = torch.tensor([[5], [6], [7]])
    expected = model(token, past_key_values=reference, use_cache=True).logits
    for _ in range(2):
        cache._restore_position(snapshot)
        actual = model(token, past_key_values=cache, use_cache=True).logits
        close(actual, expected, "cpu")
        assert cache.position.data_ptr() == position_pointer
        for layer, source, pointer, extra in zip(
            cache.layers, prefix.layers, pointers, cache.prefix_slots, strict=True
        ):
            assert (layer.keys.data_ptr(), layer.values.data_ptr(), layer.cumulative_length.data_ptr()) == pointer
            end = ids.shape[1] + extra
            torch.testing.assert_close(layer.keys[..., :end, :], source.keys[..., :end, :].expand(3, -1, -1, -1))
            assert layer.keys.data_ptr() != source.keys.data_ptr()
    assert prefix.seen_tokens == 3
    with pytest.raises(ValueError, match="single-row"):
        cache.repeat_prefix(2, 7)
    with pytest.raises(ValueError, match="capacity"):
        prefix.repeat_prefix(2, 2)


@torch.no_grad()
def test_batch_views_commit_only_after_last_chunk_and_reset_after_failure():
    model = inference_model("cpu", "white_matter", prefill="autoregressive")
    cache = model.allocate_inference_cache(6)
    ids = torch.tensor([[2, 3], [4, 5], [6, 7]])
    with cache._prefill_batch(0, 2, batch_size=3) as chunk:
        model(ids[:2], past_key_values=chunk, use_cache=True)
    assert cache.seen_tokens == 0
    assert all(layer.cumulative_length.item() == 0 for layer in cache.layers)
    with cache._prefill_batch(2, 3, batch_size=3) as chunk:
        assert chunk.seen_tokens == 0
        model(ids[2:], past_key_values=chunk, use_cache=True)
    assert cache.seen_tokens == 2
    cache.reset()

    def interrupted_prefill():
        with cache._prefill_batch(0, 2, batch_size=3) as chunk:
            model(ids[:2], past_key_values=chunk, use_cache=True)
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        interrupted_prefill()
    assert cache.seen_tokens == 0
    cache.reset()
    from white_matter.models.generation import prefill

    close(prefill(model, ids, cache, batch_size=2), model(ids, use_cache=True, logits_to_keep=1).logits, "cpu")


@pytest.mark.parametrize("positions", [torch.tensor([[0, 1]]), torch.tensor([[0.0]])])
@torch.no_grad()
def test_invalid_cached_positions_do_not_mutate_document_state(positions):
    model = make_model("cpu")
    cache = model(torch.tensor([[2, 100, 3]]), use_cache=True).past_key_values
    reference = copy.deepcopy(cache)
    token = torch.tensor([[100]])
    with pytest.raises(ValueError, match="position_ids"):
        model(token, past_key_values=cache, use_cache=True, position_ids=positions)
    assert cache.seen_tokens == reference.seen_tokens
    for name in ("position", "document_ids", "last_document_id", "next_document_id"):
        torch.testing.assert_close(getattr(cache, name), getattr(reference, name), rtol=0, atol=0)
    for actual, expected in zip(cache.layers, reference.layers, strict=True):
        torch.testing.assert_close((actual.keys, actual.values), (expected.keys, expected.values), rtol=0, atol=0)
    close(
        model(token, past_key_values=cache, use_cache=True).logits,
        model(token, past_key_values=reference, use_cache=True).logits,
        "cpu",
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("failure_call", [2, 4], ids=["warmup", "capture"])
@torch.no_grad()
def test_decode_capture_failure_restores_cache_position(failure_call):
    from white_matter.models.generation import DecodeGraph

    model = inference_model("cuda", "vanilla", prefill="autoregressive")
    ids = torch.tensor([[2, 3, 4]], device="cuda")
    cache = model.allocate_inference_cache(8)
    model(ids, past_key_values=cache, use_cache=True)
    reference = copy.deepcopy(cache)
    calls = 0

    def fail_after_forward(module, args, output):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError("recoverable setup failure")

    handle = model.register_forward_hook(fail_after_forward)
    try:
        with pytest.raises(RuntimeError, match="recoverable setup failure"):
            DecodeGraph(model, cache)
    finally:
        handle.remove()
    assert calls == failure_call
    assert cache.seen_tokens == reference.seen_tokens
    torch.testing.assert_close(cache.position, reference.position, rtol=0, atol=0)
    for actual, expected in zip(cache.layers, reference.layers, strict=True):
        torch.testing.assert_close(actual.cumulative_length, expected.cumulative_length, rtol=0, atol=0)
        torch.testing.assert_close(
            (actual.keys[..., :3, :], actual.values[..., :3, :]),
            (expected.keys[..., :3, :], expected.values[..., :3, :]),
            rtol=0,
            atol=0,
        )
    token = torch.tensor([[5]], device="cuda")
    close(
        model(token, past_key_values=cache, use_cache=True).logits,
        model(token, past_key_values=reference, use_cache=True).logits,
        "cuda",
    )

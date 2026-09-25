"""Exercise the harness adapter through HF's cached and uncached generation."""
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models
from transformers import AutoConfig, AutoModelForCausalLM, LogitsProcessor, PreTrainedTokenizerFast

pytest.importorskip("lm_eval")
from evals.lm_eval import WhiteMatterHarnessLM, configure_eval_compiler


class ForcedTokens(LogitsProcessor):
    def __init__(self, width, tokens):
        self.width, self.tokens = width, tokens

    def __call__(self, input_ids, scores):
        scores.fill_(-torch.inf)
        step = min(input_ids.shape[1] - self.width, len(self.tokens) - 1)
        scores[:, self.tokens[step]] = 0
        return scores


def adapter(prefill="autoregressive", device="cpu", family="white_matter"):
    vocab = {char: index for index, char in enumerate(["<pad>", "<eos>"] + list("abcdefghijklmnopqrstuvwxyzXYZ "))}
    backend = Tokenizer(models.BPE(vocab, [], unk_token="<pad>"))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>")
    options = {}
    if family == "white_matter":
        options = dict(num_kv_channels=1, cyclic_groups=2, num_passes=2,
                       execution_mode="cyclic", prefill_mode=prefill)
    config = AutoConfig.for_model(
        family, vocab_size=len(vocab), hidden_size=64 if device == "cuda" else 16,
        intermediate_size=128 if device == "cuda" else 24,
        num_hidden_layers=2, num_attention_heads=1, num_key_value_heads=1, head_dim=64 if device == "cuda" else 16,
        max_position_embeddings=256 if device == "cuda" else 32,
        pad_token_id=0, eos_token_id=1, document_separator_token_id=None, **options,
    )
    config._attn_implementation = "flash_attention_2" if device == "cuda" else "sdpa"
    model = AutoModelForCausalLM.from_config(config).to(device).eval()
    return WhiteMatterHarnessLM(model=model, tokenizer=tokenizer, batch_size=2)


@pytest.mark.parametrize("family,prefill", [
    ("white_matter", "autoregressive"), ("white_matter", "cyclic"),
    ("white_matter", "jacobi"), ("fusedkv", None),
])
def test_generation_stops_without_mutating_requests(family, prefill, monkeypatch):
    lm = adapter(prefill, family=family)
    generated = lm.tok_encode("XYZ")
    options = {"until": ["XY"], "max_gen_toks": 5,
               "logits_processor": [ForcedTokens(3, generated)]}
    requests = [SimpleNamespace(args=(context, options)) for context in ["a", "abc"]]
    calls = []
    forward = lm.model.forward

    def record(*args, **kwargs):
        calls.append((kwargs.get("past_key_values"), kwargs.get("input_ids").shape[1]))
        return forward(*args, **kwargs)

    # Preserve the signature used by HF generation's model-kwargs validation.
    import functools
    monkeypatch.setattr(lm.model, "forward", functools.wraps(forward)(record))
    assert lm.generate_until(requests) == ["", ""]
    assert [length for _, length in calls] == ([3, 4] if family == "fusedkv" else [3, 1])
    if family == "fusedkv":
        assert all(cache is None for cache, _ in calls)
    else:
        assert calls[0][0] is not None
        assert all(cache is calls[0][0] for cache, _ in calls)
        assert lm.model.config.prefill_mode == prefill
    assert options["until"] == ["XY"] and "use_cache" not in options


@pytest.mark.parametrize("execution_mode", ["cyclic", "jacobi"])
def test_generation_requires_explicit_iterative_policy(execution_mode):
    lm = adapter(None)
    lm.model.config.execution_mode = execution_mode
    with pytest.raises(ValueError, match="prefill_mode"):
        lm.generate_until([SimpleNamespace(args=("a", {}))])


def test_context_window_override():
    lm = adapter()
    short = WhiteMatterHarnessLM(model=lm.model, tokenizer=lm.tokenizer, batch_size=2, max_length=8)
    assert short.max_length == 8
    assert short._prepare_full_ids(list(range(10)), [3])[0] == list(range(3, 10)) + [3]
    with pytest.raises(ValueError, match="max_length"):
        WhiteMatterHarnessLM(model=lm.model, tokenizer=lm.tokenizer, batch_size=2, max_length=1)


def test_compiler_limit_covers_mixed_eval_shapes(monkeypatch):
    from torch._dynamo import config as dynamo_config

    monkeypatch.setattr(dynamo_config, "recompile_limit", 8)
    configure_eval_compiler()
    assert dynamo_config.recompile_limit >= 64


def test_generation_truncation_empty_context_and_request_order():
    lm = adapter()
    options = {"max_gen_toks": 2, "logits_processor": [ForcedTokens(30, lm.tok_encode("Z"))]}
    requests = [SimpleNamespace(args=(text, options)) for text in ["", "a" * 50]]
    assert lm.generate_until(requests) == ["ZZ", "ZZ"]


@pytest.mark.parametrize("family", ["white_matter", "fusedkv"])
def test_generation_restores_order_across_options_and_honors_eos(family):
    lm = adapter(family=family)
    requests = [
        SimpleNamespace(args=("a", {"max_gen_toks": 3, "logits_processor": [ForcedTokens(1, [1])]})),
        SimpleNamespace(args=("abc", {"max_gen_toks": 2,
                                      "logits_processor": [ForcedTokens(3, lm.tok_encode("Z"))]})),
    ]
    assert lm.generate_until(requests) == ["", "ZZ"]


@pytest.mark.parametrize("options", [{"max_gen_toks": 32}, {"num_beams": 2}, {"until": [""]}])
def test_generation_rejects_unsupported_options(options):
    lm = adapter()
    with pytest.raises(ValueError):
        lm.generate_until([SimpleNamespace(args=("a", options))])


@pytest.mark.parametrize("family", ["white_matter", "fusedkv"])
def test_stop_strings_do_not_match_across_prompt_boundary(family):
    lm = adapter(family=family)
    request = SimpleNamespace(args=("X", {"max_gen_toks": 2, "until": ["XY"],
                                         "logits_processor": [ForcedTokens(1, lm.tok_encode("YZ"))]}))
    assert lm.generate_until([request]) == ["YZ"]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("prefill", ["autoregressive", "cyclic", "jacobi"])
@pytest.mark.parametrize("documents", [False, True])
@torch.inference_mode()
def test_gpu_generation_dynamic_padding_matches_sdpa(prefill, documents, monkeypatch):
    """Compare FA decoding against SDPA on the same padded/document-aware cache inputs."""
    import white_matter.ops.flash_attention as flash

    torch.manual_seed(81)
    lm = adapter(prefill, "cuda")
    if documents:
        lm.model.config.document_separator_token_id = lm.tok_encode("X")[0]
    middle = "X" if documents else "a"
    prompts = ["a" * 64 + middle + "b" * length for length in (62, 63)]
    options = {"max_gen_toks": 2, "logits_processor": [ForcedTokens(128, lm.tok_encode("Z"))]}
    requests = [SimpleNamespace(args=(prompt, options)) for prompt in prompts]
    logits, positions, fa_calls = [], [], []

    def record(module, args, output):
        logits.append(output.logits.detach().clone())
        positions.append(output.past_key_values.position.clone())
        assert output.past_key_values.capacity is None
        assert (output.past_key_values.document_ids == -2).sum() == 1

    handle = lm.model.register_forward_hook(record)
    fa_decode = flash.flash_attention_decode

    def record_fa(*args, **kwargs):
        fa_calls.append(1)
        return fa_decode(*args, **kwargs)

    monkeypatch.setattr(flash, "flash_attention_decode", record_fa)
    try:
        assert lm.generate_until(requests) == ["ZZ", "ZZ"]
        assert fa_calls, "generation must exercise the actual FlashAttention cached kernel"
        expected_logits, expected_positions = logits[:], positions[:]
        logits.clear()
        positions.clear()
        monkeypatch.setattr(lm.model.config, "_attn_implementation", "sdpa")
        for layer in lm.model.model.decoder.block.layers:
            monkeypatch.setattr(layer.self_attn, "attention_implementation", "sdpa")
            if prefill == "jacobi":
                monkeypatch.setattr(layer.self_attn, "_force_jacobi_reference", True)
        assert lm.generate_until(requests) == ["ZZ", "ZZ"]
    finally:
        handle.remove()
    assert len(logits) == len(expected_logits) == 2
    for actual, expected in zip(logits, expected_logits, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-3)
    for actual, expected in zip(positions, expected_positions, strict=True):
        torch.testing.assert_close(actual, expected)
    expected_end = [64, 63] if documents else [129, 128]
    assert positions[-1].flatten().tolist() == expected_end

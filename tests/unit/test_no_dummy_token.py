"""No-dummy defaults and equivalence across feedback execution schedules."""

import copy

import pytest
import torch

from tests.unit.test_model_serialization import config
from white_matter import WhiteMatterConfig, WhiteMatterForCausalLM


@pytest.mark.parametrize("dummy", [False, True])
@pytest.mark.parametrize("mode", ["jacobi", "cyclic", "autoregressive"])
def test_modes_converge_to_recurrence_and_preserve_gradients(dummy, mode):
    cfg = config()
    cfg.use_dummy_token = dummy
    cfg.num_passes = 7
    block = WhiteMatterForCausalLM(cfg).model.decoder.block.double()
    reference = copy.deepcopy(block)
    docs = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 2, 2]])
    inputs = torch.randn(2, 5, cfg.hidden_size, dtype=torch.float64)
    probe = torch.randn_like(inputs)
    records = []
    for current, kind in [(reference, "autoregressive"), (block, mode)]:
        x = inputs.clone().requires_grad_()
        with torch.compiler.set_stance("force_eager"):
            if kind == "jacobi":
                y = current.forward_jacobi(x, document_ids=docs)
            elif kind == "cyclic":
                y = current(x, cyclic_groups=2, document_ids=docs)[0]
            else:
                y = current.forward_autoregressive(x, document_ids=docs)
            (y * probe).sum().backward()
        records.append((y, x.grad, {n: p.grad for n, p in current.named_parameters()}))
    torch.testing.assert_close(*records, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("dummy", [False, True])
def test_cached_jacobi_continuation(dummy):
    cfg = config()
    cfg.use_dummy_token = dummy
    cfg.execution_mode = cfg.prefill_mode = "jacobi"
    cfg.num_passes = 6
    cfg.document_separator_token_id = None
    model = WhiteMatterForCausalLM(cfg).eval()
    ids = torch.randint(1, cfg.vocab_size, (2, 7))
    with torch.no_grad():
        prefix = model(ids[:, :3], use_cache=True)
        cache = prefix.past_key_values
        block = model.model.decoder.block
        state = tuple(
            t.unflatten(1, (cfg.num_kv_channels, cfg.num_key_value_heads))
            for t in (cache.layers[0].keys, cache.layers[0].values)
        )
        x = model.get_input_embeddings()(ids[:, 3:])
        expected, _ = block.forward_recurrent(x, initial_state=state)
        actual = model(ids[:, 3:], past_key_values=cache, use_cache=True)
        torch.testing.assert_close(actual.logits, model.lm_head(model.model.norm(expected)), rtol=1e-5, atol=1e-6)
        assert cache.layers[0].keys.shape[-2] == 7 + int(dummy)


def test_default_and_serialization():
    cfg = WhiteMatterConfig()
    assert cfg.use_dummy_token is False
    assert WhiteMatterConfig.from_dict(cfg.to_dict()).use_dummy_token is False
    cfg = config()
    block = WhiteMatterForCausalLM(cfg).model.decoder.block
    assert block.dummy_token is None
    assert "dummy_token" not in block.state_dict()


@pytest.mark.parametrize("dummy", [False, True])
@pytest.mark.parametrize("surrounding", [False, True])
@torch.no_grad()
def test_cached_jacobi_with_document_boundaries(dummy, surrounding):
    cfg = config()
    cfg.use_dummy_token = dummy
    cfg.execution_mode = cfg.prefill_mode = "jacobi"
    cfg.num_passes = 7
    cfg.document_separator_token_id = None
    if surrounding:
        cfg.num_pre_layers = cfg.num_post_layers = 1
        cfg.num_hidden_layers += 2
    model = WhiteMatterForCausalLM(cfg).eval()
    reference = copy.deepcopy(model)
    reference.config.prefill_mode = "autoregressive"
    reference.config.execution_mode = "autoregressive"
    ids = torch.randint(1, cfg.vocab_size, (2, 7))
    docs = torch.tensor([[0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 2, 2]])
    mask = torch.ones_like(ids)
    for current in (model, reference):
        prefix = current(ids[:, :3], document_ids=docs[:, :3], attention_mask=mask[:, :3], use_cache=True)
        result = current(
            ids[:, 3:],
            document_ids=docs[:, 3:],
            attention_mask=mask[:, 3:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
        if current is model:
            actual = result.logits
        else:
            torch.testing.assert_close(actual, result.logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dummy", [False, True])
@torch.no_grad()
def test_cached_jacobi_after_right_padding(dummy):
    torch.manual_seed(1)
    cfg = config()
    cfg.use_dummy_token = dummy
    cfg.execution_mode = cfg.prefill_mode = "jacobi"
    cfg.num_passes = 7
    cfg.document_separator_token_id = None
    model = WhiteMatterForCausalLM(cfg).eval()
    reference = copy.deepcopy(model)
    reference.config.execution_mode = reference.config.prefill_mode = "autoregressive"
    ids = torch.tensor([[3, 4, 0, 0, 5, 6], [7, 8, 9, 10, 11, 12]])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    outputs = []
    for current in (model, reference):
        prefix = current(ids[:, :4], attention_mask=mask, use_cache=True)
        result = current(
            ids[:, 4:],
            attention_mask=torch.ones_like(ids[:, 4:]),
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
        outputs.append(result.logits)
    torch.testing.assert_close(*outputs, rtol=1e-5, atol=1e-6)

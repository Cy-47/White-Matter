from __future__ import annotations

import copy

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from evals.heldout import evaluate


def tiny_config(architecture="vanilla", **kwargs):
    return AutoConfig.for_model(
        architecture,
        vocab_size=101,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=10_000.0,
        eos_token_id=100,
        document_separator_token_id=100,
        num_kv_channels=2 if architecture == "white_matter" else None,
        **kwargs,
    )


def test_heldout_uses_the_public_document_masked_forward():
    torch.manual_seed(7)
    model = AutoModelForCausalLM.from_config(tiny_config()).eval()
    ids = torch.tensor([[1, 2, 100, 3, 4, 5]])
    with torch.inference_mode():
        expected = model(ids, labels=ids).loss
        loss_sum, tokens = evaluate(model, [{"input_ids": ids}], logits_chunk=2)
    assert tokens == 5
    torch.testing.assert_close(torch.tensor(loss_sum / tokens), expected)


def test_return_dict_defaults_to_config():
    model = AutoModelForCausalLM.from_config(tiny_config(return_dict=False)).eval()
    assert isinstance(model(torch.tensor([[1, 2]])), tuple)


def test_logits_to_keep_preserves_last_token_logits():
    model = AutoModelForCausalLM.from_config(tiny_config()).eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = model(ids).logits[:, -1:]
        actual = model(ids, logits_to_keep=1).logits
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("option", ["output_hidden_states", "output_attentions"])
def test_unsupported_outputs_are_explicit(option):
    model = AutoModelForCausalLM.from_config(tiny_config()).eval()
    with pytest.raises(NotImplementedError):
        model(torch.tensor([[1, 2]]), **{option: True})


def test_short_cyclic_prompt_matches_causally_padded_prompt():
    model = AutoModelForCausalLM.from_config(tiny_config("white_matter", cyclic_groups=4)).eval()
    ids = torch.tensor([[1, 2, 3]])
    padded = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]])
    with torch.inference_mode():
        expected = model(padded, attention_mask=mask).logits[:, :3]
        actual = model(ids).logits
        generated = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=2, do_sample=False)
    torch.testing.assert_close(actual, expected)
    assert generated.shape == (1, 5)


@pytest.mark.parametrize("include_top_output", [False, True])
def test_exact_ar_training_recomputation_matches_uncheckpointed_reference(include_top_output):
    from training.forward import TrainingForward
    from training.losses import checkpointed_linear_cross_entropy, lm_cross_entropy_from_hidden
    from white_matter.modules.documents import document_ids_from_eos

    torch.manual_seed(93)
    model = AutoModelForCausalLM.from_config(
        tiny_config("white_matter", execution_mode="autoregressive", include_top_output=include_top_output)
    ).train()
    # Exercise a learned boundary state away from RMSNorm's zero-input singular scale.
    with torch.no_grad():
        model.model.decoder.block.dummy_token.normal_(std=0.02)
    reference = copy.deepcopy(model)
    ids = torch.tensor([[1, 100, 2, 3, 4, 100, 5], [6, 7, 100, 8, 100, 9, 10]])

    def run(model, checkpointed):
        inputs = model.get_input_embeddings()(ids)
        inputs.retain_grad()
        if checkpointed:
            normalized = TrainingForward(model, checkpoint_chunk_size=2, external_ce=True)(inputs, 1, 1, token_ids=ids)
            loss = checkpointed_linear_cross_entropy(normalized, ids, model.lm_head, token_chunk_size=3)
        else:
            hidden = model.model.decoder.block.forward_autoregressive(
                inputs, document_ids=document_ids_from_eos(ids, 100), checkpoint_chunk_size=0
            )
            normalized = model.model.norm(hidden)
            loss = lm_cross_entropy_from_hidden(ids, hidden=hidden, final_norm=model.model.norm, lm_head=model.lm_head)
        loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                gradients[name] = parameter.grad
        return normalized.detach(), loss.detach(), inputs.grad, gradients

    expected, actual = run(reference, False), run(model, True)
    for index in range(3):
        torch.testing.assert_close(actual[index], expected[index], rtol=1e-5, atol=2e-6)
    assert actual[3].keys() == expected[3].keys()
    for name in expected[3]:
        torch.testing.assert_close(actual[3][name], expected[3][name], rtol=2e-5, atol=2e-6, msg=name)


@pytest.mark.parametrize("execution_mode", ["cyclic", "autoregressive"])
def test_white_matter_decoder_forward_uses_configured_schedule(execution_mode):
    model = AutoModelForCausalLM.from_config(tiny_config("white_matter", cyclic_groups=2)).eval()
    model.config.execution_mode = execution_mode
    inputs = model.get_input_embeddings()(torch.tensor([[1, 2, 3, 4]]))
    decoder = model.model.decoder
    with torch.inference_mode():
        expected = (
            decoder.block.forward_autoregressive(inputs)
            if execution_mode == "autoregressive"
            else decoder.block.forward(inputs, num_passes=2, cyclic_groups=2)[0]
        )
        actual = decoder(inputs, num_passes=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("execution_mode", ["cyclic", "autoregressive"])
@pytest.mark.parametrize("training_forward", [False, True])
def test_public_and_training_forward_execute_the_decoder(execution_mode, training_forward):
    """Decoder-level processing must run before the feedback block."""
    from training.forward import TrainingForward

    model = AutoModelForCausalLM.from_config(
        tiny_config("white_matter", execution_mode=execution_mode, cyclic_groups=2)
    ).eval()
    ids = torch.tensor([[1, 2, 100, 3, 4]])
    wrapper = TrainingForward(model)

    def run():
        if training_forward:
            return wrapper(model.get_input_embeddings()(ids), 2, 0, token_ids=ids, compute_ce=False)
        return model(ids).logits

    calls = []

    def process_decoder_input(module, inputs):
        calls.append(module)
        return (inputs[0] + 0.125,)

    with torch.inference_mode():
        original = run()
        handle = model.model.decoder.register_forward_pre_hook(process_decoder_input)
        try:
            actual = run()
        finally:
            handle.remove()
    assert calls == [model.model.decoder]
    assert not torch.equal(actual, original)


@pytest.mark.parametrize("family", ["white_matter", "lckv", "vanilla", "fusedkv"])
@pytest.mark.parametrize("causal_lm", [False, True])
def test_hf_embedding_access_replacement_and_resize(family, causal_lm):
    from transformers import AutoModel

    factory = AutoModelForCausalLM if causal_lm else AutoModel
    model = factory.from_config(tiny_config(family)).eval()
    embeddings = torch.nn.Embedding(101, 32)
    original = embeddings.weight.detach().clone()
    model.set_input_embeddings(embeddings)
    assert model.get_input_embeddings() is embeddings
    if causal_lm:
        assert model.get_output_embeddings().weight is embeddings.weight
    resized = model.resize_token_embeddings(109, mean_resizing=False)
    assert model.get_input_embeddings() is resized
    assert model.config.vocab_size == 109
    torch.testing.assert_close(resized.weight[:101], original, rtol=0, atol=0)
    with torch.inference_mode():
        output = model(torch.tensor([[1, 108]]))
    if causal_lm:
        assert model.get_output_embeddings().weight is resized.weight
        assert output.logits.shape == (1, 2, 109)
    else:
        assert output.last_hidden_state.shape == (1, 2, 32)


@pytest.mark.parametrize("separator", [None, 99, 100])
@pytest.mark.parametrize("execution_mode", ["cyclic", "autoregressive"])
def test_training_uses_public_document_separator_policy(separator, execution_mode):
    from training.forward import TrainingForward

    config = tiny_config("white_matter", execution_mode=execution_mode, cyclic_groups=2)
    config.document_separator_token_id = separator
    model = AutoModelForCausalLM.from_config(config).train()
    ids = torch.tensor([[1, 100, 2, 99, 3, 4]])
    expected = model(ids, use_cache=False).logits
    hidden = TrainingForward(model)(
        model.get_input_embeddings()(ids), config.num_passes, config.num_passes, token_ids=ids, compute_ce=False
    )
    actual = model.lm_head(model.model.norm(hidden))
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    actual_gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    expected.square().sum().backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(actual_gradients[name], parameter.grad, msg=name)


def test_ar_graph_requires_document_separator():
    from training.forward import TrainingForward

    config = tiny_config("white_matter", execution_mode="autoregressive")
    config.document_separator_token_id = None
    model = AutoModelForCausalLM.from_config(config)
    with pytest.raises(ValueError, match="document_separator_token_id"):
        TrainingForward(model).capture_ar_graph(torch.zeros(1, 2, config.hidden_size))

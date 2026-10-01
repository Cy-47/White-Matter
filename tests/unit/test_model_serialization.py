import pytest
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from white_matter import WhiteMatterConfig, WhiteMatterForCausalLM


def config():
    return WhiteMatterConfig(
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
        num_kv_channels=2,
        cyclic_groups=2,
        num_passes=2,
    )


@pytest.mark.parametrize("include_top_output", [False, True])
@pytest.mark.parametrize("execution_mode", ["cyclic", "autoregressive"])
def test_causal_lm_round_trip_preserves_weights_and_execution(tmp_path, execution_mode, include_top_output):
    cfg = config()
    cfg.include_top_output = include_top_output
    cfg.execution_mode = execution_mode
    cfg.prefill_mode = execution_mode
    original = WhiteMatterForCausalLM(cfg).eval()
    original.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    assert loaded.config.include_top_output is include_top_output
    assert loaded.model.decoder.block.kv_pool.num_layers == cfg.num_hidden_layers + int(include_top_output)
    assert loaded.config.execution_mode == execution_mode
    assert loaded.config.prefill_mode == execution_mode
    assert loaded.lm_head.weight is loaded.get_input_embeddings().weight
    ids = torch.tensor([[1, 2, 100, 3, 4, 5]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids).logits, original(ids).logits, rtol=1e-5, atol=1e-6)
        expected = original(ids[:, :3], use_cache=True)
        actual = loaded(ids[:, :3], use_cache=True)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            loaded(ids[:, 3:], past_key_values=actual.past_key_values).logits,
            original(ids[:, 3:], past_key_values=expected.past_key_values).logits,
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.parametrize("architecture", ["white_matter", "lckv", "vanilla", "fusedkv"])
@pytest.mark.parametrize("causal_lm_export", [False, True])
def test_backbone_auto_model_round_trip(tmp_path, architecture, causal_lm_export):
    cfg = {
        "vocab_size": 101,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "rope_theta": 10_000.0,
        "eos_token_id": 100,
        "document_separator_token_id": 100,
    }
    if architecture == "white_matter":
        cfg.update(num_kv_channels=2, cyclic_groups=2, num_passes=2)
    elif architecture == "lckv":
        cfg.update(num_pre_layers=1, num_post_layers=1, num_passes=2)
    model = AutoModel.from_config(AutoConfig.for_model(architecture, **cfg)).eval()
    if causal_lm_export:
        lm = AutoModelForCausalLM.from_config(model.config)
        lm.model.load_state_dict(model.state_dict())
        lm.save_pretrained(tmp_path)
    else:
        model.save_pretrained(tmp_path)
    if architecture == "lckv":
        assert model.config.execution_mode == "jacobi"
    loaded = AutoModel.from_pretrained(tmp_path).eval()
    if architecture == "lckv":
        assert loaded.config.execution_mode == "jacobi"
    assert type(loaded) is type(model)
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids).last_hidden_state, model(ids).last_hidden_state, rtol=1e-5, atol=1e-6)

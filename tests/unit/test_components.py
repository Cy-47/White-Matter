"""Independent composition, surrounding layers, and optional dependency boundaries."""

import copy
import subprocess
import sys

import pytest
import torch

from training.forward import TrainingForward
from white_matter import WhiteMatterConfig, WhiteMatterForCausalLM
from white_matter.models.decoder_utils import prepare_attention_inputs, run_feedforward_layers
from white_matter.modules.documents import document_ids_from_eos


def test_eager_jacobi_matches_sdpa_and_cannot_read_future_tokens():
    from white_matter.blocks import FeedbackDecoderLayer, WhiteMatterBlock
    from white_matter.layers import WhiteMatterAttention
    from white_matter.modules import GatedMLP, KVPool, RotaryEmbedding

    torch.manual_seed(73)
    layers = [FeedbackDecoderLayer(32, WhiteMatterAttention(32, 4, 8), GatedMLP(32, 64)) for i in range(2)]
    reference = WhiteMatterBlock(layers, KVPool(32, 2, 8, 3, 1), RotaryEmbedding(8)).double()
    eager = copy.deepcopy(reference)
    for layer in eager.layers:
        layer.self_attn.attention_implementation = "eager"
    inputs = torch.randn(2, 8, 32, dtype=torch.float64)

    def run(block):
        x = inputs.clone().requires_grad_(True)
        output = block.forward_jacobi(x, num_passes=3, num_gradient_passes=2)
        output.square().mean().backward()
        gradients = {name: p.grad for name, p in block.named_parameters()}
        assert all(gradient is not None for gradient in gradients.values())
        return output.detach(), x.grad, gradients

    actual, expected = run(eager), run(reference)
    # Eager attention deliberately computes softmax in FP32.
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    changed = inputs.clone()
    changed[:, 1:] += 10
    with torch.no_grad():
        output = eager.forward_jacobi(changed, num_passes=3)
    torch.testing.assert_close(output[:, 0], actual[0][:, 0], rtol=0, atol=0)


def test_standalone_block_trains_without_optional_dependencies():
    from pathlib import Path

    example = Path(__file__).parents[2] / "examples/feedback_block.py"
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys, runpy
from importlib.abc import MetaPathFinder
class BlockOptional(MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'transformers', 'tilelang', 'flash_attn'}:
            raise AssertionError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
example = sys.argv[1]
sys.argv = [example, '--no-compile']
runpy.run_path(example, run_name='__main__')
""",
            str(example),
        ],
        check=True,
    )


@pytest.mark.parametrize("execution_mode", ["cyclic", "autoregressive"])
def test_feedforward_layers_match_explicit_composition_and_all_gradients(execution_mode):
    torch.manual_seed(41)
    config = WhiteMatterConfig(
        vocab_size=101,
        eos_token_id=100,
        document_separator_token_id=100,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=4,
        num_pre_layers=1,
        num_post_layers=1,
        num_kv_channels=2,
        num_passes=3,
        cyclic_groups=2,
        execution_mode=execution_mode,
    )
    model = WhiteMatterForCausalLM(config)
    reference = copy.deepcopy(model)
    ids = torch.tensor([[1, 2, 100, 3, 4, 100, 5, 6]])
    documents = document_ids_from_eos(ids, 100)

    def run(model, explicit):
        x = model.get_input_embeddings()(ids)
        x.retain_grad()
        decoder = model.model.decoder
        if explicit:
            attention_args = prepare_attention_inputs(x, documents, decoder.rotary_emb, "sdpa")
            hidden = run_feedforward_layers(decoder.pre_layers, x, **attention_args)
            if execution_mode == "cyclic":
                hidden, _ = decoder.block(
                    hidden, cyclic_groups=2, num_passes=3, num_gradient_passes=2, document_ids=documents
                )
            else:
                hidden = decoder.block.forward_autoregressive(hidden, document_ids=documents)
            hidden = run_feedforward_layers(decoder.post_layers, hidden, **attention_args)
        else:
            hidden = TrainingForward(model)(x, 3, 2, token_ids=ids, compute_ce=False)
        model.lm_head(model.model.norm(hidden)).square().mean().backward()
        gradients = {}
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            gradients[name] = param.grad
        return hidden, x.grad, gradients

    actual, expected = run(model, False), run(reference, True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize(
    "options", [{"num_passes": 0}, {"num_gradient_passes": -1}, {"num_gradient_passes": 4}, {"cyclic_groups": 0}]
)
def test_block_rejects_invalid_iteration_counts(options):
    from white_matter.blocks import FeedbackDecoderLayer, WhiteMatterBlock
    from white_matter.layers import WhiteMatterAttention
    from white_matter.modules import GatedMLP, KVPool, RotaryEmbedding

    layers = [FeedbackDecoderLayer(32, WhiteMatterAttention(32, 4, 8), GatedMLP(32, 64))]
    block = WhiteMatterBlock(layers, KVPool(32, 2, 8, 2, 1), RotaryEmbedding(8, 10_000.0), num_passes=3)
    with pytest.raises(ValueError, match=next(iter(options))):
        block(torch.randn(1, 16, 32), **options)


def test_lckv_block_initializes_and_trains_without_an_hf_owner():
    from white_matter.blocks import FeedbackDecoderLayer, LCKVBlock
    from white_matter.layers import WhiteMatterAttention
    from white_matter.modules import GatedMLP, KVPool, RotaryEmbedding
    from white_matter.modules.routing import FixedSourceMixer

    layers = [
        FeedbackDecoderLayer(32, WhiteMatterAttention(32, 4, 8, strict_causal=True), GatedMLP(32, 64)) for i in range(2)
    ]
    pool = KVPool(32, 2, 8, 2, 1, mixer=FixedSourceMixer(2))
    block = LCKVBlock(layers, pool, RotaryEmbedding(8, 10_000.0), num_passes=3)
    assert all(p.requires_grad for p in block.parameters())
    assert not any("router" in n or "dummy" in n for n, _ in block.named_parameters())
    x = torch.randn(1, 8, 32, requires_grad=True)
    output = block(x, num_gradient_passes=2, document_ids=torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]]))
    output.square().mean().backward()
    assert x.grad is not None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())


def test_cyclic_all_gradient_passes_do_not_build_a_discarded_pool(monkeypatch):
    from examples.feedback_block import make_block

    block = make_block()
    builds = []
    original = block.kv_pool.project_sequence

    def record_build(*args, **kwargs):
        builds.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(block.kv_pool, "project_sequence", record_build)
    x = torch.randn(2, 16, 32, requires_grad=True)
    documents = torch.arange(16).expand(2, -1) // 5
    output, state = block(x, cyclic_groups=4, num_passes=2, num_gradient_passes=2, document_ids=documents)
    assert state is None
    # One initial pool, then one refresh for each group in each pass.
    assert len(builds) == 1 + 4 * 2
    output.square().mean().backward()
    # Backward rematerializes each group, but not a discarded initial pool.
    assert len(builds) == 1 + 2 * 4 * 2
    assert x.grad is not None
    assert all(parameter.grad is not None for parameter in block.parameters())


@pytest.mark.parametrize("with_dummy", [False, True])
@torch.no_grad()
def test_inference_pool_bounds_projection_memory_and_preserves_positions(monkeypatch, with_dummy):
    from white_matter.modules import KVPool, RotaryEmbedding

    torch.manual_seed(27)
    pool = KVPool(32, 2, 8, 4, 2).double()
    for parameter in pool.parameters():
        parameter.add_(torch.randn_like(parameter) * 0.02)
    # A broadcast layer stack, ragged final chunk and different positions per batch.
    stacked = torch.randn(2, 1031, 1, 32, dtype=torch.float64).expand(-1, -1, 4, -1)
    dummy = torch.randn(32, dtype=torch.float64) if with_dummy else None
    positions = torch.arange(1031 + int(with_dummy)).repeat(2, 1)
    positions[1] %= 73
    rope = RotaryEmbedding(8)(stacked, positions)
    expected = pool.project_sequence(stacked, rope, dummy_token=dummy)
    sizes = []
    project = pool._project

    def record(chunk):
        sizes.append(chunk.shape[0] * chunk.shape[1])
        return project(chunk)

    concatenate = torch.cat

    def no_full_kv_concat(tensors, dim=0, **kwargs):
        assert not (dim == 3 and tensors[0].ndim == 5), "projected chunks must write directly into final KV"
        return concatenate(tensors, dim=dim, **kwargs)

    monkeypatch.setattr(torch, "cat", no_full_kv_concat)
    monkeypatch.setattr(pool, "_project", record)
    actual = pool.eval().project_sequence(stacked, rope, dummy_token=dummy)
    assert len(sizes) == 3
    assert max(sizes) <= 1024 + stacked.shape[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_inference_pool_stores_the_keys_consumed_by_attention():
    from white_matter.modules import KVPool, RotaryEmbedding

    pool = KVPool(32, 2, 8, 4, 2).cuda()
    stacked = torch.randn(2, 17, 4, 32, device="cuda", dtype=torch.bfloat16)
    rope = RotaryEmbedding(8).cuda()(stacked, torch.arange(17, device="cuda")[None])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = pool.project_sequence(stacked, rope)
        actual = pool.eval().project_sequence(stacked, rope)
    assert expected[0].dtype == expected[1].dtype == torch.bfloat16
    assert actual[0].dtype == actual[1].dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

"""HF mixed-dtype loading and the real inference path preserve matrix rounding."""
import copy

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from white_matter.models.generation import DecodeGraph
from white_matter.modules import KVPool


def make_model(family):
    family_options = {
        'white_matter': dict(num_kv_channels=2, num_passes=3, cyclic_groups=4, prefill_mode='cyclic'),
        'lckv': dict(num_passes=3, prefill_mode='jacobi'),
    }.get(family, {})
    config = AutoConfig.for_model(
        family, vocab_size=101, hidden_size=128, intermediate_size=192,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        residual_dtype='bf16', eos_token_id=100, document_separator_token_id=None,
        **family_options,
    )
    torch.manual_seed(37)
    model = AutoModelForCausalLM.from_config(config).eval()
    # Nontrivial gains and routers expose incorrect low-precision loading.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * .013)
    return model


@pytest.mark.parametrize('family', ['white_matter', 'vanilla', 'lckv', 'fusedkv'])
def test_hf_bf16_loading_preserves_non_matrix_parameters(tmp_path, family):
    model = make_model(family)
    matrices = {id(p) for m in model.modules() if isinstance(m, torch.nn.Linear)
                for p in m.parameters(recurse=False)}
    matrices.update(id(p) for m in model.modules() if isinstance(m, KVPool)
                    for p in (m.k_proj_weight, m.v_proj_weight))
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    expected = {}
    for name, param in model.named_parameters():
        expected[name] = param.to(torch.bfloat16) if id(param) in matrices else param
        torch.testing.assert_close(dict(loaded.named_parameters())[name], expected[name], rtol=0, atol=0)
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    loaded.save_pretrained(tmp_path / 'mixed')
    restored = AutoModelForCausalLM.from_pretrained(tmp_path / 'mixed', dtype=torch.bfloat16)
    for name, param in restored.named_parameters():
        torch.testing.assert_close(param, expected[name], rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('family', ['white_matter', 'vanilla'])
@torch.no_grad()
def test_bf16_loaded_inference_matches_fp32_masters(tmp_path, family):
    original = make_model(family)
    original.save_pretrained(tmp_path)
    models = [AutoModelForCausalLM.from_pretrained(
        tmp_path, dtype=dtype, attn_implementation='flash_attention_2',
    ).cuda().eval() for dtype in (torch.float32, torch.bfloat16)]
    ids = torch.randint(1, 99, (2, 521), device='cuda')
    caches = [m.allocate_inference_cache(528) for m in models]
    # Isolate storage/autocast equivalence from compilation. Different compiler
    # graphs can reorder FP32 reductions; both storage modes are checked against
    # full FP32 in the compiled accuracy test below.
    with torch.compiler.set_stance('force_eager'):
        outputs = [m(ids, past_key_values=c, use_cache=True).logits for m, c in zip(models, caches, strict=True)]
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        graphs = [DecodeGraph(m, c) for m, c in zip(models, caches, strict=True)]
        for token in torch.randint(1, 99, (2, 5), device='cuda').split(1, dim=1):
            torch.testing.assert_close(*(g(token) for g in graphs), rtol=0, atol=0)
        for a, b in zip(caches[0].layers, caches[1].layers, strict=True):
            torch.testing.assert_close((a.keys, a.values), (b.keys, b.values), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_pool_fusions_match_unfused_and_skip_training(dtype):
    from white_matter.modules import RotaryEmbedding

    torch.manual_seed(19)
    pool = KVPool(192, 2, 96, 4, 2, router_layer_stride=2).cuda()
    for parameter in pool.parameters():
        parameter.add_(torch.randn_like(parameter) * .02)
    reference = copy.deepcopy(pool)  # train mode deliberately keeps the unfused path
    x = torch.randn(2, 37, 4, 192, device='cuda', dtype=dtype)
    rope = RotaryEmbedding(96).cuda()(x, torch.arange(37, device='cuda')[None])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        expected = reference.project_sequence(x, rope)
        actual = pool.eval().project_sequence(x, rope)
    torch.testing.assert_close(actual, tuple(t.bfloat16() for t in expected), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('family', ['white_matter', 'vanilla'])
@pytest.mark.parametrize('num_splits', [0, 2])
@pytest.mark.parametrize('parameter_dtype', [torch.float32, torch.bfloat16])
@torch.no_grad()
@torch._dynamo.config.patch(fail_on_recompile_limit_hit=True)
def test_compiled_inference_preserves_fp32_accuracy(family, num_splits, parameter_dtype, monkeypatch, tmp_path):
    import contextlib
    import flash_attn
    from white_matter.models import modeling_base
    from white_matter.blocks._execution import cyclic

    # Each case compiles an independent model and cache configuration.
    torch.compiler.reset()
    native_decode = flash_attn.flash_attn_with_kvcache

    def decode(*args, **kwargs):
        assert kwargs.get('num_splits', 0) == num_splits
        return native_decode(*args, **kwargs)

    monkeypatch.setattr(flash_attn, 'flash_attn_with_kvcache', decode)

    original = make_model(family)
    if parameter_dtype == torch.bfloat16:
        original.save_pretrained(tmp_path)
        eager = AutoModelForCausalLM.from_pretrained(
            tmp_path, dtype=parameter_dtype, attn_implementation='flash_attention_2',
        ).cuda().eval()
    else:
        eager = original.cuda()
    eager.config._attn_implementation = 'flash_attention_2'
    for module in eager.modules():
        if hasattr(module, 'num_splits'):
            module.num_splits = num_splits
        if hasattr(module, 'attention_implementation'):
            module.attention_implementation = 'flash_attention_2'
    # Widen the same effective weights; BF16 storage rounding is a separate error.
    reference = copy.deepcopy(eager).float()
    compiled = copy.deepcopy(eager)
    reference.config.residual_dtype = 'fp32'
    reference.config._attn_implementation = 'sdpa'
    for module in reference.modules():
        if hasattr(module, 'attention_implementation'):
            module.attention_implementation = 'sdpa'
            module._force_cyclic_reference = True
    ids = torch.randint(1, 99, (2, 128), device='cuda')
    caches = [model.allocate_inference_cache(136) for model in (eager, compiled, reference)]
    compiled(ids, past_key_values=caches[1], use_cache=True)
    caches[1].reset()
    # Vanilla must have no breaks; WM prepares its cached cyclic schedule in Python.
    compiled.compile(fullgraph=family == 'vanilla', options={
        'emulate_precision_casts': True, 'reorder_for_locality': False,
    })

    def full_precision(tokens):
        with monkeypatch.context() as patch, torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            patch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
            patch.setattr(modeling_base, 'model_autocast_context', lambda _: contextlib.nullcontext())
            patch.setattr(cyclic, 'model_autocast_context', lambda _: contextlib.nullcontext())
            return reference(tokens, past_key_values=caches[2], use_cache=True).logits

    def check(a, b, fp32):
        # RMS reduction order differs at ~1 FP32 ULP and can change BF16 rounding.
        # Bound that difference and independently guard accuracy vs FP32.
        norm = fp32.float().norm().clamp_min(1e-12)
        assert (a.float() - b.float()).norm() / norm < .01
        eager_error = (b.float() - fp32.float()).norm() / norm
        assert (a.float() - fp32.float()).norm() / norm <= eager_error * 1.05 + 1e-5

    expected = eager(ids, past_key_values=caches[0], use_cache=True).logits
    actual = compiled(ids, past_key_values=caches[1], use_cache=True).logits
    check(actual, expected, full_precision(ids))
    graph = DecodeGraph(compiled, caches[1])
    for token in ids[:, :4].split(1, dim=1):
        expected = eager(token, past_key_values=caches[0], use_cache=True).logits
        check(graph(token), expected, full_precision(token))
    for actual, expected in zip(caches[1].layers, caches[0].layers, strict=True):
        for a, b in zip((actual.keys, actual.values), (expected.keys, expected.values), strict=True):
            assert (a.float() - b.float()).norm() / b.float().norm() < .01


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('with_dummy', [False, True])
@torch.no_grad()
def test_chunked_pool_direct_writes_match_concatenation(with_dummy):
    from white_matter.modules import RotaryEmbedding

    torch.manual_seed(28)
    pool = KVPool(128, 1, 64, 4, 2).cuda().eval()
    x = torch.randn(2, 1031, 4, 128, device='cuda', dtype=torch.bfloat16)
    dummy = torch.randn(128, device='cuda') if with_dummy else None
    offset = int(with_dummy)
    rope = RotaryEmbedding(64).cuda()(x, torch.arange(x.shape[1] + offset, device='cuda')[None])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        chunks = []
        for start in range(0, x.shape[1], 512):
            end = min(start + 512, x.shape[1])
            positions = tuple(t[:, start + offset if start else 0:end + offset] for t in rope)
            chunks.append(pool.project_sequence(x[:, start:end], positions, dummy_token=dummy if start == 0 else None))
        expected = tuple(torch.cat(parts, dim=3) for parts in zip(*chunks, strict=True))
        actual = pool.project_sequence(x, rope, dummy_token=dummy)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

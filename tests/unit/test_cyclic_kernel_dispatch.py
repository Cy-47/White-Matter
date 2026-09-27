"""Compilation policy tests that do not require TileLang or a CUDA device."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

from white_matter.ops.cyclic_attention._tilelang import registration


@pytest.fixture
def compiler(monkeypatch):
    calls = []
    tilelang = ModuleType("tilelang")
    language = ModuleType("tilelang.language")
    language.dynamic = lambda name, dtype="int32": (name, dtype)
    tilelang.language = language
    tilelang.PassConfigKey = SimpleNamespace(TL_ENABLE_FAST_MATH="fast_math")

    def compile_program(program, **kwargs):
        kernel = object()
        calls.append((program, kwargs, kernel))
        return kernel

    tilelang.compile = compile_program
    monkeypatch.setitem(sys.modules, "tilelang", tilelang)
    monkeypatch.setitem(sys.modules, "tilelang.language", language)
    module = ModuleType(f"{registration.__package__}.forward")
    module.build_program = lambda *args, **kwargs: (args, kwargs)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: [(8, 0), (8, 6), (9, 0)][index])
    registration._get_kernel.cache_clear()
    registration._device_capability.cache_clear()
    yield calls
    registration._get_kernel.cache_clear()
    registration._device_capability.cache_clear()


def _tensor(shape, device=0):
    return SimpleNamespace(shape=shape, device=torch.device("cuda", device))


def _forward(device=0, batch=1, query_length=64, kv_length=256, block=64):
    query = _tensor((batch, 8, query_length, 128), device)
    key = _tensor((batch, 2, kv_length, 128), device)
    return registration._kernel_for("fwd", query, key, key, 4, block)


def test_runtime_shapes_and_equivalent_requested_tiles_reuse_kernel(compiler):
    kernel = _forward(block=32)
    assert _forward(batch=3, query_length=17, kv_length=511, block=64) is kernel
    assert len(compiler) == 1
    assert registration._get_kernel.cache_info().hits == 1
    args, strides = compiler[0][0]
    assert args[:2] == (("batch", "int32"), ("kv_length", "int32"))
    assert args[5] == ("query_length", "int32")
    assert args[7:10] == (64, 128, 2)
    assert strides["key_strides"] == (
        ("key_stride_0", "int64"),
        ("key_stride_1", "int64"),
        ("key_stride_2", "int64"),
        1,
    )


def test_device_capability_selects_independent_tiles_and_target(compiler):
    a100 = _forward(device=0)
    a6000 = _forward(device=1)
    assert a100 is not a6000
    assert _forward(device=0) is a100
    assert _forward(device=1) is a6000
    assert len(compiler) == 2
    assert compiler[0][0][0][7:10] == (64, 128, 2)
    assert compiler[1][0][0][7:10] == (64, 64, 2)
    assert compiler[0][1]["target"] == {"kind": "cuda", "arch": "sm_80"}
    assert compiler[1][1]["target"] == {"kind": "cuda", "arch": "sm_86"}


def test_fallback_tuning_preserves_actual_compilation_target(compiler):
    _forward(device=2)
    assert compiler[0][0][0][7:10] == (64, 64, 2)
    assert compiler[0][1]["target"] == {"kind": "cuda", "arch": "sm_90"}


@pytest.mark.parametrize("layout", ["offset", "channel_stride", "row_stride", "singleton_stride"])
def test_unaligned_native_views_are_normalized(layout):
    if layout == "offset":
        value = torch.randn(1 + 2 * 3 * 128, dtype=torch.bfloat16)[1:].view(1, 2, 3, 128)
        assert value.is_contiguous()
        assert value.data_ptr() % 16 != 0
    elif layout == "channel_stride":
        value = torch.randn(1, 2, 3, 256, dtype=torch.bfloat16)[..., ::2]
    elif layout == "singleton_stride":
        value = torch.randn(1, 2, 3, 128, dtype=torch.bfloat16).as_strided((1, 2, 3, 128), (1, 384, 128, 1))
        assert value.is_contiguous()
    else:
        value = torch.randn(1, 2, 3, 129, dtype=torch.bfloat16)[..., :128]
    normalized = registration._normalize_native_kv(value, value.dtype)
    assert normalized.is_contiguous()
    assert normalized.data_ptr() % 16 == 0
    assert normalized.stride(-1) == 1
    assert all(stride % 8 == 0 for stride in normalized.stride()[:3])
    torch.testing.assert_close(normalized, value, rtol=0, atol=0)


def test_aligned_native_capacity_views_remain_copy_free():
    value = torch.randn(2, 3, 2, 17, 128, dtype=torch.bfloat16)[:, 1, :, :13, :]
    assert not value.is_contiguous()
    assert registration._normalize_native_kv(value, value.dtype) is value


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int32])
def test_contiguous_normalization_clones_misaligned_singleton_views(dtype):
    value = torch.arange(129, dtype=torch.float32).to(dtype)[1:].view(1, 1, 1, 128)
    assert value.is_contiguous()
    assert value.data_ptr() % 16 != 0
    normalized = registration._contiguous_in_dtype(value, dtype)
    assert normalized.data_ptr() % 16 == 0
    assert normalized.is_contiguous()
    torch.testing.assert_close(normalized, value, rtol=0, atol=0)
    assert registration._contiguous_in_dtype(normalized, dtype) is normalized


def test_key_value_pipeline_stages_are_capped(monkeypatch):
    monkeypatch.setitem(
        registration._TILE_TUNING_BY_CC[(8, 0)], 128, {"dkv": (64, 64, 4), "dq": (64, 64, 4), "fwd": (64, 128, 4)}
    )
    assert registration._kernel_tile_sizes(128, 64, "dkv", (8, 0)) == (64, 64, 2)
    assert registration._kernel_tile_sizes(128, 64, "dq", (8, 0)) == (64, 64, 4)


def test_document_forward_retains_its_own_tuning():
    assert registration._kernel_tile_sizes(128, 64, "fwd", (8, 6)) == (64, 64, 2)
    assert registration._kernel_tile_sizes(128, 64, "doc_fwd", (8, 6)) == (64, 32, 2)

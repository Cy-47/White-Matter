"""Registered TileLang cyclic-attention operators for compile-safe autograd."""

from __future__ import annotations

import functools

import torch

# Validated tiles for A6000 (99 KiB shared memory) and A100 (163 KiB).
# Ragged queries use predicated 64-row tiles: a 32-row fallback fails layout inference.
_TILE_TUNING_BY_CC = {
    (8, 6): {  # A6000 / sm_86, 99 KiB shared-memory limit
        # D:   {kernel: (block_M, block_N, num_stages)}
        128: {"fwd": (64, 64, 2), "doc_fwd": (64, 32, 2), "dq": (64, 32, 2), "dkv": (64, 32, 1)},
        # D=96 needs block_N=32 to fit the shared-memory limit.
        96: {"fwd": (64, 32, 2), "dq": (64, 32, 1), "dkv": (32, 32, 1)},
    },
    (8, 0): {  # A100 / sm_80, 163 KiB shared-memory limit
        # The larger shared memory permits wider tiles. The key/value backward
        # pipeline uses at most two stages because its Q-loop has a variable
        # trip count; a third stage can read beyond the valid range.
        128: {"fwd": (64, 128, 2), "dq": (64, 64, 2), "dkv": (64, 64, 2)},
    },
}
_FALLBACK_CC = (8, 6)


@functools.cache
def _device_capability(device_index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


def _kernel_tile_sizes(D: int, requested_block_M: int, kernel: str, capability: tuple[int, int]):
    """Resolve measured tiles before caching a compiled specialization."""
    if kernel in {"doc_dq", "doc_dkv"}:
        return 32, 32, 1
    table = _TILE_TUNING_BY_CC.get(capability, _TILE_TUNING_BY_CC[_FALLBACK_CC])
    tune = table.get(D)
    if tune is None:
        return requested_block_M, 64, 2
    block_M, block_N, stages = tune[kernel if kernel in tune else kernel.removeprefix("doc_")]
    return block_M, block_N, min(stages, 2) if kernel == "dkv" else stages


_RUNTIME_RESIDUE_BUFFERS: dict[tuple[str, int], tuple[torch.Tensor, ...]] = {}


def _runtime_residue_buffer(device: torch.device, K_stride: int, residue: int) -> torch.Tensor:
    """Return a persistent device scalar without allocating on the hot path.

    Specializing cyclic kernels on ``residue`` would recompile them for each
    chunk even when their tensor shapes are unchanged. A one-element device
    buffer makes residue a runtime kernel input.
    Build all residues for a stride together so the first chunk is the only
    one that allocates, and keep the views alive for CUDA-graph replay.
    """
    if not 0 <= residue < K_stride:
        raise ValueError(f"cyclic residue must be in [0, {K_stride}), got {residue}")
    key = (str(device), K_stride)
    buffers = _RUNTIME_RESIDUE_BUFFERS.get(key)
    if buffers is None:
        values = torch.arange(K_stride, dtype=torch.int32, device=device)
        buffers = tuple(values.split(1))
        _RUNTIME_RESIDUE_BUFFERS[key] = buffers
    return buffers[residue]


@functools.lru_cache(maxsize=768)
def _get_kernel(kind, HQ, HKV, D, K_stride, tiles, capability):
    """Compile structural variants; dimensions and outer KV strides stay runtime."""
    from importlib import import_module

    import tilelang
    import tilelang.language as T

    module_name = {
        "fwd": "forward",
        "dq": "backward_query",
        "dkv": "backward_key_value",
        "doc_fwd": "document_forward",
        "doc_dq": "document_backward_query",
        "doc_dkv": "document_backward_key_value",
    }[kind]
    block_M, block_N, stages = tiles
    B, T_kv, Q_LEN = (T.dynamic(name) for name in ("batch", "kv_length", "query_length"))
    strides = {}
    if kind == "fwd":
        # Infer outer strides from each tensor, including padded cache capacity.
        # Storage normalization guarantees unit element stride and vector alignment.
        strides = {
            "key_strides": (*[T.dynamic(f"key_stride_{i}", "int64") for i in range(3)], 1),
            "value_strides": (*[T.dynamic(f"value_stride_{i}", "int64") for i in range(3)], 1),
        }
    program = import_module(f"{__package__}.{module_name}").build_program(
        B,
        T_kv,
        HQ,
        HKV,
        D,
        Q_LEN,
        K_stride,
        block_M,
        block_N,
        stages,
        128,
        **strides,
    )
    return tilelang.compile(
        program,
        target={"kind": "cuda", "arch": f"sm_{capability[0]}{capability[1]}"},
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


def _kernel_for(kind, Q, K, V, K_stride, block_M):
    capability = _device_capability(Q.device.index)
    tiles = _kernel_tile_sizes(Q.shape[-1], block_M, kind, capability)
    # Q is BHQD here; all runtime dimensions are excluded from the cache key.
    with torch.cuda.device(Q.device):
        return _get_kernel(kind, Q.shape[1], K.shape[1], Q.shape[-1], K_stride, tiles, capability)


def _forward_impl(Q, K, V, metadata, K_stride, residue, block_M):
    QSeg, KSeg, QStart, KQEnd = metadata or (None,) * 4
    B, HQ, Q_LEN, D = Q.shape
    # Projections produce/consume BQHD. Keep that storage across attention;
    # the public operator still exposes BHQD views, including to autograd.
    query = Q
    Q = _contiguous_in_dtype(Q.transpose(1, 2), Q.dtype)
    # Plain prefill reads native cache views directly, including capacity padding
    # and channel selection. Unaligned layouts are normalized before dispatch.
    normalize = _normalize_native_kv if metadata is None else _contiguous_in_dtype
    K, V = (normalize(x, Q.dtype) for x in (K, V))
    output = torch.empty_like(Q)
    lse = torch.empty((B, HQ, Q_LEN), dtype=torch.float32, device=Q.device)
    segments = () if QSeg is None else tuple(_contiguous_in_dtype(x, torch.int32) for x in (QSeg, KSeg, QStart))
    kind = "doc_fwd" if segments else "fwd"
    kernel = _kernel_for(kind, query, K, V, K_stride, block_M)
    kernel(Q, K, V, *segments, _runtime_residue_buffer(Q.device, K_stride, residue), output, lse)
    return output.transpose(1, 2), lse


def _backward_impl(Q, K, V, metadata, Out, Lse, dOut, K_stride, residue, block_M):
    from .backward_preprocess import backward_preprocess

    QSeg, KSeg, QStart, KQEnd = metadata or (None,) * 4
    Q, K, V, dOut = (_contiguous_in_dtype(x, Q.dtype) for x in (Q, K, V, dOut))
    # Hoist the row reduction to save shared memory in both backward kernels.
    # Keep Out's noncontiguous view: only this reduction consumes it.
    D_pre = backward_preprocess(dOut, Out.to(Q.dtype))
    dQ, dK, dV = (torch.empty_like(x) for x in (Q, K, V))
    segments = ()
    if QSeg is not None:
        QSeg, KSeg, QStart, KQEnd = (_contiguous_in_dtype(x, torch.int32) for x in (QSeg, KSeg, QStart, KQEnd))
        segments = (QSeg, KSeg)
    residue_buffer = _runtime_residue_buffer(Q.device, K_stride, residue)
    for kind, bound, outputs in (("dq", QStart, (dQ,)), ("dkv", KQEnd, (dK, dV))):
        kernel = _kernel_for("doc_" + kind if segments else kind, Q, K, V, K_stride, block_M)
        metadata = (*segments, bound) if segments else ()
        kernel(Q, K, V, dOut, D_pre, Lse, *metadata, residue_buffer, *outputs)
    return dQ, dK, dV


@torch.library.custom_op("white_matter::cyclic_attn_fwd", mutates_args=())
def cyclic_attn_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    K_stride: int,
    residue: int,
    block_M: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _forward_impl(Q, K, V, None, K_stride, residue, block_M)


@cyclic_attn_fwd.register_fake
def _fake_fwd(Q: torch.Tensor, *_: object, **__: object) -> tuple[torch.Tensor, torch.Tensor]:
    B, HQ, Q_LEN, D = Q.shape
    Out = torch.empty((B, Q_LEN, HQ, D), dtype=Q.dtype, device=Q.device).transpose(1, 2)
    Lse = torch.empty((B, HQ, Q_LEN), dtype=torch.float32, device=Q.device)
    return Out, Lse


@torch.library.custom_op("white_matter::cyclic_attn_bwd", mutates_args=())
def cyclic_attn_bwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Out: torch.Tensor,
    Lse: torch.Tensor,
    dOut: torch.Tensor,
    K_stride: int,
    residue: int,
    block_M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _backward_impl(Q, K, V, None, Out, Lse, dOut, K_stride, residue, block_M)


@cyclic_attn_bwd.register_fake
def _fake_bwd(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, *_: object, **__: object
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # MUST mirror the real op's output metadata EXACTLY (contiguous, Q.dtype) —
    # see cyclic_attn_bwd, which does K = K.contiguous().to(Q.dtype) before
    # dK = empty_like(K). If the fake returned empty_like(original K/V) instead
    # (wrong dtype/strides when K/V != Q), inductor compiles the downstream
    # gather over dK/dV with the wrong element size/stride -> CUDA illegal
    # memory access under torch.compile (eager never consults the fake).
    Qc = Q.contiguous()
    Kc = K.contiguous().to(Qc.dtype)
    Vc = V.contiguous().to(Qc.dtype)
    return torch.empty_like(Qc), torch.empty_like(Kc), torch.empty_like(Vc)


def _setup_ctx(ctx, inputs, output):
    *tensors, ctx.K_stride, ctx.residue, ctx.block_M = inputs
    ctx.save_for_backward(*tensors, *output)


def _backward(ctx, dOut, dLse):
    tensors = ctx.saved_tensors
    # Three Q/K/V inputs, optionally four document tensors, and two outputs.
    operator = cyclic_attn_doc_bwd if len(tensors) == 9 else cyclic_attn_bwd
    gradients = operator(*tensors, dOut, ctx.K_stride, ctx.residue, ctx.block_M)
    return (*gradients, *((None,) * (len(tensors) - 2)))


cyclic_attn_fwd.register_autograd(_backward, setup_context=_setup_ctx)


# Document kernels consume explicit segment IDs and backward bounds, retaining
# the globally visible dummy. Ragged metadata loads are clamped in the kernels.


def _normalize_native_kv(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Retain aligned cache views; normalize layouts unsafe for vector loads."""
    if x.dtype != dtype:
        x = x.to(dtype)
    if x.stride(-1) == 1 and x.data_ptr() % 16 == 0 and all(stride % 8 == 0 for stride in x.stride()[:3]):
        return x
    return x.clone(memory_format=torch.contiguous_format)


def _contiguous_in_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return contiguous, 16-byte-aligned storage in ``dtype``.

    The document-masked training path invokes these custom ops more than a
    thousand times per step. Production Q/K/V and the hoisted segment buffers
    already satisfy both conditions, so unconditional ``.contiguous().to()``
    paid thousands of Python/dispatcher calls that returned the input unchanged.
    Keep the public op's conversion behavior for unusual callers, but branch
    before dispatch on the hot path.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != dtype:
        x = x.to(dtype)
    return x if x.data_ptr() % 16 == 0 else x.clone()


@torch.library.custom_op("white_matter::cyclic_attn_doc_fwd", mutates_args=())
def cyclic_attn_doc_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    QSeg: torch.Tensor,
    KSeg: torch.Tensor,
    QStart: torch.Tensor,
    KQEnd: torch.Tensor,
    K_stride: int,
    residue: int,
    block_M: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _forward_impl(Q, K, V, (QSeg, KSeg, QStart, KQEnd), K_stride, residue, block_M)


cyclic_attn_doc_fwd.register_fake(_fake_fwd)


@torch.library.custom_op("white_matter::cyclic_attn_doc_bwd", mutates_args=())
def cyclic_attn_doc_bwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    QSeg: torch.Tensor,
    KSeg: torch.Tensor,
    QStart: torch.Tensor,
    KQEnd: torch.Tensor,
    Out: torch.Tensor,
    Lse: torch.Tensor,
    dOut: torch.Tensor,
    K_stride: int,
    residue: int,
    block_M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _backward_impl(Q, K, V, (QSeg, KSeg, QStart, KQEnd), Out, Lse, dOut, K_stride, residue, block_M)


cyclic_attn_doc_bwd.register_fake(_fake_bwd)


cyclic_attn_doc_fwd.register_autograd(_backward, setup_context=_setup_ctx)

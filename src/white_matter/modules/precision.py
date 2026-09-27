"""Residual precision and CUDA BF16 autocast for model execution."""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

import torch

# Only changed by the inference-only precision scope; default training is unchanged.
_FP32_INFERENCE = False


@contextmanager
def fp32_inference() -> Iterator[None]:
    """Disable nested model autocast for an eager FP32 reference evaluation.

    The caller must also select reference attention and FP32 model parameters.
    This process-local scope is intended for isolated evaluation workers.
    """
    global _FP32_INFERENCE
    if torch.is_grad_enabled():
        raise RuntimeError("FP32 reference scope is inference-only")
    previous = _FP32_INFERENCE
    matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    _FP32_INFERENCE = True
    try:
        with torch.autocast("cuda", enabled=False):
            yield
    finally:
        _FP32_INFERENCE = previous
        torch.set_float32_matmul_precision(matmul_precision)


def residual_activation_dtype(device: torch.device | str, residual_dtype: str) -> torch.dtype:
    if residual_dtype not in {"fp32", "bf16"}:
        raise ValueError("residual_dtype must be 'fp32' or 'bf16'")
    return (
        torch.bfloat16
        if not _FP32_INFERENCE and residual_dtype == "bf16" and torch.device(device).type == "cuda"
        else torch.float32
    )


def model_autocast_context(device: torch.device | str) -> AbstractContextManager[None]:
    return (
        torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True)
        if torch.device(device).type == "cuda" and not _FP32_INFERENCE
        else nullcontext()
    )


def cast_residual(tensor: torch.Tensor, *, residual_dtype: str) -> torch.Tensor:
    return tensor.to(residual_activation_dtype(tensor.device, residual_dtype))

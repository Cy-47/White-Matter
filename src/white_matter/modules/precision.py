"""Residual precision and CUDA BF16 autocast for model execution."""

from contextlib import AbstractContextManager, nullcontext

import torch


def residual_activation_dtype(device: torch.device | str, residual_dtype: str) -> torch.dtype:
    if residual_dtype not in {"fp32", "bf16"}:
        raise ValueError("residual_dtype must be 'fp32' or 'bf16'")
    return torch.bfloat16 if residual_dtype == "bf16" and torch.device(device).type == "cuda" else torch.float32


def model_autocast_context(device: torch.device | str) -> AbstractContextManager[None]:
    return (
        torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True)
        if torch.device(device).type == "cuda"
        else nullcontext()
    )


def cast_residual(tensor: torch.Tensor, *, residual_dtype: str) -> torch.Tensor:
    return tensor.to(residual_activation_dtype(tensor.device, residual_dtype))

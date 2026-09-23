"""Training's FP32 master parameters, BF16 autocast, and attention policy."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from white_matter.modules.precision import residual_activation_dtype


def configure_precision(device: Any) -> None:
    if torch.device(device).type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    # Leave room for specializations across feedback shapes and gradient states.
    torch._dynamo.config.recompile_limit = max(64, torch._dynamo.config.recompile_limit)


def attention_kernel_context(device: str | torch.device):
    """Allow Flash/Efficient SDPA on CUDA; leave CPU dispatch unchanged."""
    return (
        sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
        if torch.device(device).type == "cuda"
        else nullcontext()
    )


def assert_precision_contract(
    model: nn.Module,
    decoder_input: torch.Tensor,
    *,
    residual_dtype: str,
    require_gradients: bool = False,
) -> None:
    """Check residual dtype, FP32 masters, and the first backward's gradients."""
    expected = residual_activation_dtype(decoder_input.device, residual_dtype)
    if decoder_input.dtype != expected:
        raise RuntimeError(f"decoder activation dtype mismatch: expected={expected} actual={decoder_input.dtype}")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dtype != torch.float32:
            raise RuntimeError(f"{name}: expected FP32 master parameter, got {parameter.dtype}")
        if require_gradients:
            if parameter.grad is None:
                raise RuntimeError(f"{name}: missing trainable parameter gradient")
            if parameter.grad.dtype != parameter.dtype:
                raise RuntimeError(f"{name}: gradient dtype differs from its master parameter")


def describe_precision_policy(device: str | torch.device, residual_dtype: str) -> str:
    cuda = torch.device(device).type == "cuda"
    return (
        f"parameters/grads={torch.float32} "
        f"activations={residual_activation_dtype(device, residual_dtype)} "
        f"autocast={'bf16' if cuda else 'off'} tf32={'on' if cuda else 'off'}"
    )

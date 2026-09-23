"""Fused preprocessing shared by both cyclic-attention backward paths."""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_dot(dout, out, delta, N: tl.constexpr, H: tl.constexpr, Q: tl.constexpr, D: tl.constexpr,
             S_B: tl.constexpr, S_H: tl.constexpr, S_Q: tl.constexpr, BLOCK_D: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * 8 + tl.arange(0, 8)
    dims = tl.arange(0, BLOCK_D)
    # Forward stores BQHD; read its BHQD view without copying it.
    offsets = rows // (H * Q) * S_B + rows // Q % H * S_H + rows % Q * S_Q
    mask = (rows[:, None] < N) & (dims[None, :] < D)
    x = tl.load(dout + rows[:, None] * D + dims[None, :], mask, other=0).to(tl.float32)
    y = tl.load(out + offsets[:, None] + dims[None, :], mask, other=0).to(tl.float32)
    tl.store(delta + rows, tl.sum(x * y, axis=1), rows < N)


def backward_preprocess(dout: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Compute sum(dout * out, -1) in FP32; dout is contiguous, out has unit inner stride."""
    B, H, Q, D = dout.shape
    delta = torch.empty((B, H, Q), device=dout.device, dtype=torch.float32)
    _row_dot[(triton.cdiv(B * H * Q, 8),)](
        dout, out, delta, B * H * Q, H, Q, D, *out.stride()[:3],
        BLOCK_D=triton.next_power_of_2(D), num_warps=4,
    )
    return delta

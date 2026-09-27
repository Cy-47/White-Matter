"""Fused preprocessing shared by both cyclic-attention backward paths."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["Q", "S_B", "S_H", "S_Q"])
def _row_dot(
    dout,
    out,
    delta,
    H: tl.constexpr,
    Q,
    D: tl.constexpr,
    S_B,
    S_H,
    S_Q,
    BLOCK_D: tl.constexpr,
):
    queries = tl.program_id(0).to(tl.int64) * 8 + tl.arange(0, 8)
    head = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    rows = (batch * H + head) * Q + queries
    dims = tl.arange(0, BLOCK_D)
    # Forward stores BQHD; read its BHQD view without copying it.
    offsets = batch * S_B + head * S_H + queries * S_Q
    mask = (queries[:, None] < Q) & (dims[None, :] < D)
    x = tl.load(dout + rows[:, None] * D + dims[None, :], mask, other=0).to(tl.float32)
    y = tl.load(out + offsets[:, None] + dims[None, :], mask, other=0).to(tl.float32)
    tl.store(delta + rows, tl.sum(x * y, axis=1), queries < Q)


def backward_preprocess(dout: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Compute sum(dout * out, -1) in FP32; dout is contiguous, out has unit inner stride."""
    B, H, Q, D = dout.shape
    delta = torch.empty((B, H, Q), device=dout.device, dtype=torch.float32)
    _row_dot[(triton.cdiv(Q, 8), H, B)](
        dout,
        out,
        delta,
        H,
        Q,
        D,
        *out.stride()[:3],
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=4,
    )
    return delta

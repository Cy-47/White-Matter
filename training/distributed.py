"""Manual data-parallel reduction after local microbatch clipping.

Clipping and averaging do not commute. The trainer clips before accumulating
and reducing, then applies a final global clip before the optimizer update.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Iterable

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# A full FP32 gradient vector can occupy several GiB. Limiting coalescing
# allocations to 64 MiB avoids that transient peak while retaining efficient
# NCCL collectives. Parameters larger than the cap are reduced in contiguous
# views, without a flat copy or an oversized collective.
GRAD_REDUCE_BUCKET_CAP_BYTES = 64 << 20


def setup_distributed(timeout: timedelta = timedelta(hours=2)) -> tuple[int, int, int, str, bool]:
    """(rank, world, local_rank, device, is_rank0) from torchrun env;
    (0, 1, 0, "cuda:0", True) when run standalone.

    The generous NCCL timeout covers rank-0-only phases (eval, checkpoint
    I/O) during which the other ranks sit at a barrier.
    """
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda_available = torch.cuda.is_available()
    if world > 1 and not cuda_available:
        raise RuntimeError("NCCL distributed training requires CUDA")
    if cuda_available:
        # Bind before process-group construction so NCCL and barrier() share an
        # explicit, unambiguous rank-to-device mapping from their first call.
        torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group(
            backend="nccl",
            timeout=timeout,
            device_id=torch.device("cuda", local_rank),
        )
    return rank, world, local_rank, f"cuda:{local_rank}", rank == 0


def initialize_all_reduce(device: str) -> None:
    """Initialize NCCL's real-data all-reduce path before model activations.

    Tiny scalar collectives do not initialize every NCCL protocol buffer. If
    the first substantial all-reduce follows a memory-bound backward, NCCL can
    fail to reserve that persistent workspace even though the gradient itself
    is reduced in place. A single production-sized collective before model
    construction makes that memory part of the true training footprint.
    """
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    elements = max(1, GRAD_REDUCE_BUCKET_CAP_BYTES // torch.float32.itemsize)
    buffer = torch.zeros(elements, device=device, dtype=torch.float32)
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)


def all_reduce_grads(
    params: Iterable[torch.nn.Parameter],
    world_size: int,
    *,
    validate_presence: bool = False,
) -> None:
    """Average gradients in bounded, dtype/device-compatible buckets.

    Reduced buckets become gradient storage; oversized contiguous gradients
    reduce in place through bounded views. Validate rankwise presence before
    the first reduction to prevent mismatched collectives.
    """
    if dist.is_available() and dist.is_initialized():
        process_group_world_size = dist.get_world_size()
        if world_size != process_group_world_size:
            raise ValueError(
                f"all_reduce_grads world_size={world_size} does not match "
                f"initialized process-group world size {process_group_world_size}"
            )
    if world_size <= 1:
        return

    param_list = list(params)
    if not param_list:
        return
    if validate_presence:
        # A different unused-parameter set on two ranks would make the
        # subsequent tensor sequence disagree and can deadlock NCCL. Production
        # validates the first step (and first resumed step), before any
        # differently shaped floating-point collective can begin.
        presence = torch.tensor(
            [int(param.grad is not None) for param in param_list],
            device=param_list[0].device,
            dtype=torch.int32,
        )
        dist.all_reduce(presence, op=dist.ReduceOp.SUM)
        mismatch = ((presence != 0) & (presence != world_size)).nonzero().flatten()
        if mismatch.numel():
            indices = mismatch[:16].cpu().tolist()
            raise RuntimeError(f"gradient presence differs across ranks at parameter indices {indices}")
    gradient_groups: dict[
        tuple[torch.device, torch.dtype],
        list[torch.nn.Parameter],
    ] = {}
    for parameter in param_list:
        gradient = parameter.grad
        if gradient is None:
            continue
        # Dict insertion order makes the collective sequence deterministic from
        # the already-authenticated model traversal while keeping dtype/device
        # compatible tensors in the same flat buffer.
        key = (gradient.device, gradient.dtype)
        gradient_groups.setdefault(key, []).append(parameter)
    if not gradient_groups:
        return

    def reduce_bucket(bucket_params: list[torch.nn.Parameter]) -> None:
        bucket_grads: list[torch.Tensor] = []
        for parameter in bucket_params:
            gradient = parameter.grad
            assert gradient is not None
            bucket_grads.append(gradient)
        bucket_bytes = sum(gradient.numel() * gradient.element_size() for gradient in bucket_grads)
        if len(bucket_grads) == 1 and bucket_bytes > GRAD_REDUCE_BUCKET_CAP_BYTES:
            gradient = bucket_grads[0]
            if not gradient.is_contiguous():
                raise RuntimeError("oversized gradients must be contiguous for bounded flat-view reduction")
            flat = gradient.view(-1)
            chunk_elements = max(
                1,
                GRAD_REDUCE_BUCKET_CAP_BYTES // gradient.element_size(),
            )
            for chunk in flat.split(chunk_elements):
                dist.all_reduce(chunk, op=dist.ReduceOp.SUM)
                chunk.div_(world_size)
            bucket_params[0].grad = flat.view_as(gradient)
            return
        flat = _flatten_dense_tensors(bucket_grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        reduced_grads = _unflatten_dense_tensors(flat, bucket_grads)
        for param, reduced in zip(
            bucket_params,
            reduced_grads,
            strict=True,
        ):
            # The views retain ``flat`` after this function returns, so the
            # optimizers consume the reduced bucket without a copy.
            param.grad = reduced

    for group in gradient_groups.values():
        bucket: list[torch.nn.Parameter] = []
        bucket_bytes = 0
        for parameter in group:
            gradient = parameter.grad
            assert gradient is not None
            gradient_bytes = gradient.numel() * gradient.element_size()
            if bucket and bucket_bytes + gradient_bytes > GRAD_REDUCE_BUCKET_CAP_BYTES:
                reduce_bucket(bucket)
                bucket = []
                bucket_bytes = 0
            bucket.append(parameter)
            bucket_bytes += gradient_bytes
            if bucket_bytes >= GRAD_REDUCE_BUCKET_CAP_BYTES:
                reduce_bucket(bucket)
                bucket = []
                bucket_bytes = 0
        if bucket:
            reduce_bucket(bucket)


def broadcast_params(params: Iterable[torch.nn.Parameter], src: int = 0) -> None:
    """Broadcast initialization weights from src in one coalesced allocation."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    tensors: list[torch.Tensor] = [p.data for p in params]
    if not tensors:
        return
    flat = _flatten_dense_tensors(tensors)
    dist.broadcast(flat, src=src)
    for orig, new in zip(
        tensors,
        _unflatten_dense_tensors(flat, tensors),
        strict=True,
    ):
        orig.copy_(new)

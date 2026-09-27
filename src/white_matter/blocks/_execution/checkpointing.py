"""Exact packed autoregressive reverse-mode recomputation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from white_matter.blocks.white_matter import WhiteMatterBlock


class _OffloadedPackedARCheckpoint(torch.autograd.Function):
    """Full-BPTT packed AR with one final cache endpoint offloaded to CPU.

    Append-only state makes every earlier boundary a view of that endpoint.
    Backward replays chunks in reverse, carrying the K/V adjoint between them.
    Parameters are explicit autograd inputs; accumulated gradients are returned
    to ordinary AccumulateGrad/DDP, never written into parameter.grad here.
    """

    @staticmethod
    def forward(
        ctx: Any,
        block: WhiteMatterBlock,
        chunk_size: int,
        backward_batch_size: int,
        split_state_vjp: bool,
        x: torch.Tensor,
        K_initial: torch.Tensor,
        V_initial: torch.Tensor,
        K_dummy: torch.Tensor,
        V_dummy: torch.Tensor,
        q_pos: torch.Tensor,
        valid_start: torch.Tensor,
        reset_after: torch.Tensor,
        *parameters: torch.Tensor,
    ) -> torch.Tensor:
        ctx.block = block
        ctx.chunk_size = int(chunk_size)
        ctx.backward_batch_size = int(backward_batch_size)
        ctx.split_state_vjp = bool(split_state_vjp)
        ctx.parameters = parameters
        ctx.device_type = x.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled(x.device.type)
        ctx.autocast_dtype = torch.get_autocast_dtype(x.device.type) if ctx.autocast_enabled else None
        ctx.save_for_backward(
            x,
            K_initial,
            V_initial,
            K_dummy,
            V_dummy,
            q_pos,
            valid_start,
            reset_after,
        )

        from .autoregressive import _forward_packed_chunk

        # Function.forward has no tape: the shared driver writes a fixed-capacity
        # cache, while backward replay uses its differentiable functional appends.
        output, K_state, V_state = _forward_packed_chunk(
            block,
            x,
            K_initial,
            V_initial,
            K_dummy,
            V_dummy,
            q_pos,
            valid_start,
            reset_after,
        )

        # One endpoint contains every earlier prefix.  Offload exactly once;
        # chunking now changes recomputation granularity without multiplying
        # host cache storage by the number of chunks.
        ctx.boundary = (
            K_state.detach().to(device="cpu"),
            V_state.detach().to(device="cpu"),
        )
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        x, K_initial, V_initial, K_dummy, V_dummy, q_pos, valid_start, reset_after = ctx.saved_tensors
        boundary = cast(tuple[torch.Tensor, torch.Tensor], ctx.boundary)
        grad_x = torch.empty_like(x)
        grad_K_initial, grad_V_initial = torch.empty_like(K_initial), torch.empty_like(V_initial)
        grad_K_dummy, grad_V_dummy = torch.zeros_like(K_dummy), torch.zeros_like(V_dummy)
        parameter_grads: list[torch.Tensor | None] = [None] * len(ctx.parameters)
        batch_size = ctx.backward_batch_size if ctx.backward_batch_size > 0 else x.shape[0]
        joint_chunks = not ctx.split_state_vjp and ctx.chunk_size > 1

        # Shard independent rows to bound device-resident K/V adjoints.
        for batch_start in range(0, x.shape[0], batch_size):
            batch = slice(batch_start, min(x.shape[0], batch_start + batch_size))
            grad_K_next = grad_V_next = None
            # Joint chunks reuse one endpoint transfer, avoiding quadratic PCIe traffic.
            endpoint = tuple(t[batch].to(x.device) for t in boundary) if joint_chunks else None
            for start in reversed(range(0, x.shape[1], ctx.chunk_size)):
                end = min(x.shape[1], start + ctx.chunk_size)
                joint = not ctx.split_state_vjp and end - start > 1
                length = K_initial.shape[3] + (start if joint else end)
                if joint:
                    K_value, V_value = (t[..., :length, :] for t in cast(tuple, endpoint))
                else:
                    K_value, V_value = (t[batch, :, :, :length].to(x.device) for t in boundary)

                if joint:
                    grads = _recompute_vjp(
                        ctx,
                        (x[batch, start:end], K_value, V_value, K_dummy[batch], V_dummy[batch]),
                        (q_pos[batch, start:end], valid_start[batch, start:end], reset_after[batch, start:end]),
                        (grad_output[batch, start:end], grad_K_next, grad_V_next),
                        (0, 1, 2, 3, 4),
                        ctx.parameters,
                        joint=True,
                    )
                    grad_x[batch, start:end] = cast(torch.Tensor, grads[0])
                    grad_K_next, grad_V_next = grads[1:3]
                    _accumulate([grad_K_dummy[batch], grad_V_dummy[batch]], grads[3:5])
                    _accumulate(parameter_grads, grads[5:])
                    del grads, K_value, V_value
                    continue

                # Token VJPs need only a view of each exact prefix. Split mode
                # recomputes V separately after releasing the K/x/parameter graph.
                for token in reversed(range(start, end)):
                    prefix = K_initial.shape[3] + token
                    slots = torch.arange(prefix, device=x.device).view(1, prefix)
                    metadata = (
                        q_pos[batch, token],
                        slots >= valid_start[batch, token : token + 1],
                        reset_after[batch, token],
                    )
                    inputs = (
                        x[batch, token : token + 1],
                        K_value[..., :prefix, :],
                        V_value[..., :prefix, :],
                        K_dummy[batch],
                        V_dummy[batch],
                    )
                    adjoints = (
                        grad_output[batch, token : token + 1],
                        None if grad_K_next is None else grad_K_next[..., -1:, :],
                        None if grad_V_next is None else grad_V_next[..., -1:, :],
                    )
                    wrt = (0, 1, 3) if ctx.split_state_vjp else (0, 1, 2, 3, 4)
                    grads = _recompute_vjp(ctx, inputs, metadata, adjoints, wrt, ctx.parameters)
                    grad_x[batch, token : token + 1] = cast(torch.Tensor, grads[0])
                    grad_K = cast(torch.Tensor, grads[1])
                    if grad_K_next is not None:
                        grad_K.add_(grad_K_next[..., :-1, :])
                    _accumulate([grad_K_dummy[batch], grad_V_dummy[batch]], grads[3:5])
                    _accumulate(parameter_grads, grads[5:])
                    grad_V = grads[2]
                    del grads
                    if ctx.split_state_vjp:
                        grads = _recompute_vjp(ctx, inputs, metadata, adjoints, (2, 4), ())
                        grad_V = grads[2]
                        _accumulate([grad_V_dummy[batch]], grads[4:5])
                        del grads
                    if grad_V_next is not None:
                        cast(torch.Tensor, grad_V).add_(grad_V_next[..., :-1, :])
                    grad_K_next, grad_V_next = grad_K, grad_V
                    del inputs, adjoints, grad_K, grad_V, metadata, slots
                del K_value, V_value

            grad_K_initial[batch] = cast(torch.Tensor, grad_K_next)
            grad_V_initial[batch] = cast(torch.Tensor, grad_V_next)
            del grad_K_next, grad_V_next, endpoint
        ctx.boundary = None
        return (
            None,
            None,
            None,
            None,
            grad_x,
            grad_K_initial,
            grad_V_initial,
            grad_K_dummy,
            grad_V_dummy,
            None,
            None,
            None,
            *parameter_grads,
        )


def _accumulate(total: list[torch.Tensor | None], gradients: tuple[torch.Tensor | None, ...]) -> None:
    for index, gradient in enumerate(gradients):
        if gradient is not None:
            if total[index] is None:
                total[index] = gradient
            else:
                cast(torch.Tensor, total[index]).add_(gradient)


def _recompute_vjp(
    ctx: Any,
    values: tuple[torch.Tensor, ...],
    metadata: tuple[torch.Tensor, ...],
    adjoints: tuple[torch.Tensor | None, ...],
    wrt: tuple[int, ...],
    parameters: tuple[torch.Tensor, ...],
    *,
    joint: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    """Recompute once; scope releases the graph before the next state VJP.

    values = (x, K, V, reset_K, reset_V). Metadata and successor adjoints are
    explicit so replay cannot capture a later token's masks or loop state.
    Only requested leaves require gradients, preserving split mode's memory bound.
    """
    from .autoregressive import _forward_packed_chunk

    with torch.enable_grad():
        inputs = tuple(value.detach().requires_grad_(i in wrt) for i, value in enumerate(values))
        with torch.autocast(ctx.device_type, dtype=ctx.autocast_dtype, enabled=ctx.autocast_enabled):
            if joint:
                outputs = _forward_packed_chunk(ctx.block, *inputs, *metadata)
            else:
                # Use the instance method to retain the production compile boundary.
                outputs = ctx.block._packed_autoregressive_step(*inputs, *metadata)
        grads = torch.autograd.grad(
            outputs,
            tuple(inputs[i] for i in wrt) + parameters,
            tuple(
                torch.zeros_like(output) if grad is None else grad
                for output, grad in zip(outputs, adjoints, strict=True)
            ),
            allow_unused=True,
        )
    result: list[torch.Tensor | None] = [None] * len(values)
    for index, gradient in zip(wrt, grads[: len(wrt)], strict=True):
        result[index] = gradient
    return (*result, *grads[len(wrt) :])

from __future__ import annotations

import torch

import training.distributed as wm_dist
from training.distributed import all_reduce_grads


def test_bounded_gradient_reduce_is_exact_and_optimizer_compatible(monkeypatch) -> None:
    world_size = 4
    monkeypatch.setattr(wm_dist, "GRAD_REDUCE_BUCKET_CAP_BYTES", 32)
    collective_sizes: list[int] = []

    def fake_all_reduce(tensor, *, op, async_op=False):
        assert op == torch.distributed.ReduceOp.SUM
        assert async_op is False
        collective_sizes.append(tensor.numel())
        tensor.mul_(world_size)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    parameters = [torch.nn.Parameter(torch.zeros(elements)) for elements in (5, 3, 6, 12, 2)]
    for index, parameter in enumerate(parameters):
        parameter.grad = torch.arange(parameter.numel(), dtype=parameter.dtype).view_as(parameter) + index
    expected = [parameter.grad.clone() for parameter in parameters]
    original_storage = [parameter.grad.untyped_storage().data_ptr() for parameter in parameters]

    all_reduce_grads(parameters, world_size)

    assert collective_sizes == [8, 6, 8, 4, 2]
    for parameter, expected_gradient in zip(parameters, expected, strict=True):
        torch.testing.assert_close(parameter.grad, expected_gradient, rtol=0, atol=0)
    reduced_storage = [parameter.grad.untyped_storage().data_ptr() for parameter in parameters]
    assert reduced_storage[0] == reduced_storage[1]
    assert reduced_storage[2:] == original_storage[2:]

    optimizer = torch.optim.AdamW(parameters, lr=1.0e-3, foreach=True)
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in parameters)


def test_gradient_reduce_rejects_rank_dependent_presence(monkeypatch) -> None:
    calls: list[torch.dtype] = []

    def fake_all_reduce(tensor, *, op, async_op=False):
        assert op == torch.distributed.ReduceOp.SUM
        assert async_op is False
        calls.append(tensor.dtype)
        tensor.copy_(torch.tensor([2, 1], dtype=tensor.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    parameters = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)

    try:
        all_reduce_grads(parameters, 2, validate_presence=True)
    except RuntimeError as error:
        assert "parameter indices [1]" in str(error)
    else:
        raise AssertionError("rank-dependent gradients were accepted")
    assert calls == [torch.int32]

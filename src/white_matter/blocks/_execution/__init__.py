"""Execution helpers shared by feedback schedules."""

import torch


def resolve_passes(default: int, num_passes: int | None, num_gradient_passes: int | None) -> tuple[int, int]:
    num_passes = default if num_passes is None else num_passes
    if type(num_passes) is not int or num_passes < 1:
        raise ValueError("num_passes must be a positive integer")
    if num_gradient_passes is None:
        num_gradient_passes = num_passes if torch.is_grad_enabled() else 0
    if type(num_gradient_passes) is not int or not 0 <= num_gradient_passes <= num_passes:
        raise ValueError("num_gradient_passes must be between zero and num_passes")
    return num_passes, num_gradient_passes

"""Numerical assertions shared by end-to-end GPU tests."""

import torch


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    aggregate_rtol: float | None = None,
    cosine: float = 0.999,
) -> None:
    """Accept elementwise agreement or a bounded aggregate BF16 difference."""
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as error:
        delta = (actual.float() - expected.float()).flatten()
        reference = expected.float().flatten()
        relative_l2 = delta.norm() / reference.norm().clamp_min(torch.finfo(torch.float32).tiny)
        similarity = torch.nn.functional.cosine_similarity(actual.float().flatten(), reference, dim=0)
        reference_max = reference.abs().max()
        assert (
            aggregate_rtol is not None
            and relative_l2 <= aggregate_rtol
            and similarity >= cosine
            and delta.abs().max() <= aggregate_rtol * reference_max + atol
        ), (
            f"max_abs={delta.abs().max().item():.6g}, relative_l2={relative_l2.item():.6g}, "
            f"cosine={similarity.item():.6g}; {error}"
        )


def assert_gradient_maps_close(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    **kwargs: float,
) -> None:
    assert actual.keys() == expected.keys()
    failures = []
    for name in expected:
        try:
            assert_close(actual[name], expected[name], **kwargs)
        except AssertionError as error:
            failures.append(f"{name}: {error}")
    assert not failures, "gradient mismatches:\n" + "\n".join(failures)

"""Execution policy for compiled workloads and explicit eager diagnostics."""

import argparse
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import torch


def add_compile_argument(parser: argparse.ArgumentParser, name: str = "--compile") -> None:
    parser.add_argument(
        name,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile tensor execution (default); disabling it enables eager diagnostics with a warning.",
    )


def warn_eager(reason: str = "explicit eager diagnostics") -> None:
    warnings.warn(
        f"Eager execution selected: {reason}. Compilation is disabled, including internal helpers; "
        "runtime and memory use can differ from compiled execution.",
        RuntimeWarning,
        stacklevel=2,
    )


@contextmanager
def execution_policy(compiled: bool = True, *, reason: str = "explicit eager diagnostics") -> Iterator[None]:
    """Keep eager opt-outs visible and fail on compiler fallback in compiled runs."""
    if not compiled:
        warn_eager(reason)
    with (
        # Model scales and normalization epsilons are configuration constants.
        # PyTorch 2.12's nested-region capture cannot reliably lift SymFloat
        # module attributes across repeated calls with dynamic tensor shapes.
        torch._dynamo.config.patch(suppress_errors=False, fail_on_recompile_limit_hit=True, specialize_float=True),
        torch.compiler.set_stance("default" if compiled else "force_eager"),
    ):
        yield

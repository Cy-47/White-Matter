"""Signature-preserving aliases for untyped PyTorch 2.12 decorators."""

from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar, cast

import torch

_F = TypeVar("_F", bound=Callable[..., Any])

# These aliases only provide the signatures missing from PyTorch's stubs.
compiler_disable = cast(Callable[[_F], _F], torch.compiler.disable)
dynamo_disable = cast(Callable[[_F], _F], torch._dynamo.disable)
dynamo_disable_nonrecursive = cast(Callable[[_F], _F], partial(torch._dynamo.disable, recursive=False))

"""Signature-preserving aliases for untyped PyTorch 2.12 decorators."""

from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar, cast

import torch

_F = TypeVar("_F", bound=Callable[..., Any])

# These aliases only provide the signatures missing from PyTorch's stubs.
compiler_assume_constant_result = cast(Callable[[_F], _F], torch.compiler.assume_constant_result)
nested_compile_region = cast(Callable[[_F], _F], torch.compiler.nested_compile_region)
compiler_disable = cast(Callable[[_F], _F], torch.compiler.disable)

# Skip host scheduling while the enclosing compiler still captures child tensor calls.
eager_loop = cast(Callable[[_F], _F], partial(torch.compiler.disable, recursive=False))

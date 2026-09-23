"""Selective checkpoints for inexpensive activation reconstruction."""

from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

import torch
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

_T = TypeVar("_T")
_GEMMS = {torch.ops.aten.mm.default, torch.ops.aten.bmm.default, torch.ops.aten.addmm.default}
_ATTENTION = {
    "white_matter.cyclic_attn_fwd.default",
    "white_matter.cyclic_attn_doc_fwd.default",
}


def _policy(_context: object, operation: object, *_args: object, **_kwargs: object) -> CheckpointPolicy:
    expensive = operation in _GEMMS or str(operation) in _ATTENTION
    return CheckpointPolicy.MUST_SAVE if expensive else CheckpointPolicy.MUST_RECOMPUTE


_context = partial(create_selective_checkpoint_contexts, _policy)


def checkpoint_pointwise(function: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Save matrix products and reconstruct the surrounding pointwise work."""
    return checkpoint(
        function, *args, use_reentrant=False, preserve_rng_state=False, context_fn=_context, **kwargs,
    )

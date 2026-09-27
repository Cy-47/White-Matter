"""Jacobi feedback execution with explicit document inputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, overload

import torch
from torch.utils.checkpoint import checkpoint

from white_matter.modules.precision import model_autocast_context

from . import resolve_passes

if TYPE_CHECKING:
    from white_matter.blocks.lckv import LCKVBlock
    from white_matter.blocks.white_matter import WhiteMatterBlock


@overload
def forward_jacobi(
    self: WhiteMatterBlock | LCKVBlock,
    x: torch.Tensor,
    *,
    num_passes: int | None = None,
    num_gradient_passes: int | None = None,
    document_ids: torch.Tensor | None = None,
    on_pass: Callable[[int, torch.Tensor], None] | None = None,
    output_final_state: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def forward_jacobi(
    self: WhiteMatterBlock | LCKVBlock,
    x: torch.Tensor,
    *,
    num_passes: int | None = None,
    num_gradient_passes: int | None = None,
    document_ids: torch.Tensor | None = None,
    on_pass: Callable[[int, torch.Tensor], None] | None = None,
    output_final_state: Literal[True],
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: ...


@overload
def forward_jacobi(
    self: WhiteMatterBlock | LCKVBlock,
    x: torch.Tensor,
    *,
    num_passes: int | None = None,
    num_gradient_passes: int | None = None,
    document_ids: torch.Tensor | None = None,
    on_pass: Callable[[int, torch.Tensor], None] | None = None,
    output_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: ...


def forward_jacobi(
    self: WhiteMatterBlock | LCKVBlock,
    x: torch.Tensor,
    *,
    num_passes: int | None = None,
    num_gradient_passes: int | None = None,
    document_ids: torch.Tensor | None = None,
    on_pass: Callable[[int, torch.Tensor], None] | None = None,
    output_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    n_iter, num_gradient_passes = resolve_passes(self.num_passes, num_passes, num_gradient_passes)
    if on_pass is not None and torch.is_grad_enabled():
        raise ValueError("pass observation is inference-only")
    detached_passes = n_iter - num_gradient_passes
    q_pos_emb, k_pos_emb = self._prepare_rope(x, document_ids)
    states = x.unsqueeze(1).expand(-1, len(self.layers), -1, -1).contiguous()
    with torch.no_grad(), model_autocast_context(x.device):
        for index in range(detached_passes):
            states, hidden = self.jacobi_pass(x, states, q_pos_emb, k_pos_emb, document_ids)
            if on_pass is not None:
                on_pass(index + 1, hidden)
    # Detached passes produce fresh state tensors; differentiated passes still read live x.
    for _ in range(num_gradient_passes):
        if getattr(self, "checkpoint_jacobi_passes", False) and torch.is_grad_enabled():
            # Every data-dependent input is explicit for backward recomputation.
            states, hidden = checkpoint(
                self.jacobi_pass,
                x,
                states,
                q_pos_emb,
                k_pos_emb,
                document_ids,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            states, hidden = self.jacobi_pass(x, states, q_pos_emb, k_pos_emb, document_ids)
    if output_final_state:
        # The last sweep's layer inputs define the frozen prompt memory used
        # by subsequent autoregressive tokens.
        keys, values = self.kv_pool.project_sequence(
            states.transpose(1, 2).contiguous(),
            k_pos_emb,
            dummy_token=getattr(self, "dummy_token", None),
        )
        return hidden, (keys, values)
    return hidden

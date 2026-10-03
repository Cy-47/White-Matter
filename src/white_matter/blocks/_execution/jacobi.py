"""Jacobi feedback execution with explicit document inputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, overload

import torch
from torch.utils.checkpoint import checkpoint

from white_matter.modules.documents import document_position_ids
from white_matter.modules.precision import model_autocast_context
from white_matter.ops import prepare_strict_causal_metadata

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
    past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
    position_ids: torch.Tensor | None = None,
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
    past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
    position_ids: torch.Tensor | None = None,
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
    past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
    position_ids: torch.Tensor | None = None,
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
    past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
    position_ids: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    n_iter, num_gradient_passes = resolve_passes(self.num_passes, num_passes, num_gradient_passes)
    if on_pass is not None and torch.is_grad_enabled():
        raise ValueError("pass observation is inference-only")
    detached_passes = n_iter - num_gradient_passes
    use_dummy = self.use_dummy_token
    prefix = 0
    if past_key_values is not None:
        prefix = past_key_values[0].shape[-2] - int(use_dummy)
        if prefix < 0:
            raise ValueError("past_key_values is missing its dummy slot")
        if position_ids is None:
            position_ids = (
                document_position_ids(document_ids)[:, prefix:]
                if document_ids is not None
                else torch.arange(prefix, prefix + x.shape[1], device=x.device)[None]
            )
        q_pos_emb = k_pos_emb = self.rotary_emb(x, position_ids + int(use_dummy))
    else:
        q_pos_emb, k_pos_emb = self._prepare_rope(x, document_ids)
    metadata = (
        None
        if document_ids is None
        else prepare_strict_causal_metadata(
            document_ids,
            x.shape[1],
            query_start=prefix,
            use_dummy_token=use_dummy,
        )
    )
    states = x.unsqueeze(1).expand(-1, self.kv_pool.num_layers, -1, -1).contiguous()
    # Jacobi training stays in the outer graph to preserve BF16 backward rounding.
    execute_detached_pass = self.jacobi_pass
    if num_gradient_passes == 0:
        execute_detached_pass = getattr(self, "inference_jacobi_pass", self.jacobi_pass)
    with torch.no_grad(), model_autocast_context(x.device):
        for index in range(detached_passes):
            states, hidden = execute_detached_pass(x, states, q_pos_emb, k_pos_emb, past_key_values, metadata)
            if on_pass is not None:
                on_pass(index + 1, hidden)
    # Detached passes produce fresh state tensors; differentiated passes still read live x.
    for _ in range(num_gradient_passes):
        if self.checkpoint_jacobi_passes and torch.is_grad_enabled():
            # Every data-dependent input is explicit for backward recomputation.
            states, hidden = checkpoint(
                self.jacobi_pass,
                x,
                states,
                q_pos_emb,
                k_pos_emb,
                past_key_values,
                metadata,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            states, hidden = self.jacobi_pass(x, states, q_pos_emb, k_pos_emb, past_key_values, metadata)
    if output_final_state:
        # The last sweep's layer inputs define the frozen prompt memory used
        # by subsequent autoregressive tokens.
        keys, values = self.kv_pool.project_sequence(
            states.transpose(1, 2).contiguous(),
            k_pos_emb,
            dummy_token=self.dummy_token if past_key_values is None else None,
        )
        if past_key_values is not None:
            keys, values = (
                torch.cat((past, new), dim=3) for past, new in zip(past_key_values, (keys, values), strict=True)
            )
        return hidden, (keys, values)
    return hidden

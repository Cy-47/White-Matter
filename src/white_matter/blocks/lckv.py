"""Fixed-source, strictly causal Jacobi feedback region."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import torch
from torch import nn

from white_matter.modules.documents import document_position_ids
from white_matter.modules.kv_pool import KVPool
from white_matter.modules.precision import model_autocast_context
from white_matter.modules.rotary import PositionEmbedding
from white_matter.ops import StrictCausalMetadata, prepare_strict_causal_metadata

from ._execution import jacobi, resolve_passes
from .decoder_layer import FeedbackDecoderLayer, run_feedback_layers


class LCKVBlock(nn.Module):
    use_dummy_token = False
    dummy_token: None = None
    checkpoint_jacobi_passes = False

    def __init__(
        self,
        layers: Sequence[FeedbackDecoderLayer],
        kv_pool: KVPool,
        rotary_emb: PositionEmbedding,
        *,
        num_passes: int = 9,
    ) -> None:
        super().__init__()
        if not layers or len(layers) != kv_pool.num_layers or kv_pool.num_kv_channels != 1:
            raise ValueError("LCKV requires matching layers and a single KV source")
        self.layers = nn.ModuleList(layers)
        self.kv_pool = kv_pool
        self.rotary_emb = rotary_emb
        self.num_passes = num_passes
        for layer in layers:
            if not layer.self_attn.strict_causal:
                raise ValueError("LCKV layers must use strictly causal attention")

    def _prepare_rope(
        self, x: torch.Tensor, document_ids: torch.Tensor | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        positions = (
            torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            if document_ids is None
            else document_position_ids(document_ids)
        )
        embeddings = self.rotary_emb(x, positions)
        return embeddings, embeddings

    def jacobi_pass(
        self,
        x_in: torch.Tensor,
        layer_hidden_states: torch.Tensor,
        q_pos_emb: tuple[torch.Tensor, torch.Tensor],
        k_pos_emb: tuple[torch.Tensor, torch.Tensor],
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
        metadata: StrictCausalMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if past_key_values is not None:
            raise ValueError("cached Jacobi continuation requires WhiteMatterBlock")
        keys, values = self.kv_pool.project_sequence(layer_hidden_states.transpose(1, 2).contiguous(), k_pos_emb)
        hidden, states = run_feedback_layers(
            self.layers, x_in, (keys[:, 0],), (values[:, 0],), q_pos_emb, metadata=metadata
        )
        return torch.stack(states, dim=1), hidden

    @overload
    def forward(
        self,
        x: torch.Tensor,
        *,
        num_passes: int | None = None,
        num_gradient_passes: int | None = None,
        document_ids: torch.Tensor | None = None,
        output_final_state: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        x: torch.Tensor,
        *,
        num_passes: int | None = None,
        num_gradient_passes: int | None = None,
        document_ids: torch.Tensor | None = None,
        output_final_state: Literal[True],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: ...

    def forward(
        self,
        x: torch.Tensor,
        *,
        num_passes: int | None = None,
        num_gradient_passes: int | None = None,
        document_ids: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Training retains the full-stack reference, including its gradients.
        if self.training or torch.is_grad_enabled() or num_gradient_passes not in (None, 0):
            return jacobi.forward_jacobi(
                self,
                x,
                num_passes=num_passes,
                num_gradient_passes=num_gradient_passes,
                document_ids=document_ids,
                output_final_state=output_final_state,
            )
        passes, _ = resolve_passes(self.num_passes, num_passes, num_gradient_passes)
        q_pos_emb, k_pos_emb = self._prepare_rope(x, document_ids)
        metadata = None if document_ids is None else prepare_strict_causal_metadata(document_ids, x.shape[1])
        source = x
        with torch.no_grad(), model_autocast_context(x.device):
            for _ in range(passes):
                # Only the input to the last feedback layer feeds the fixed source.
                expanded = source.unsqueeze(2).expand(-1, -1, len(self.layers), -1)
                keys, values = self.kv_pool.project_sequence(expanded, k_pos_emb)
                del expanded
                hidden = x
                for layer in self.layers:
                    source = hidden
                    hidden = layer(hidden, keys[:, 0], values[:, 0], q_pos_emb, metadata=metadata)
                del keys, values
            if output_final_state:
                expanded = source.unsqueeze(2).expand(-1, -1, len(self.layers), -1)
                return hidden, self.kv_pool.project_sequence(expanded, k_pos_emb)
        return hidden

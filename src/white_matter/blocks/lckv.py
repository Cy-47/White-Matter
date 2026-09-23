"""Fixed-source, strictly causal Jacobi feedback region."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from white_matter.modules.documents import document_position_ids
from white_matter.modules.kv_pool import KVPool
from white_matter.modules.rotary import PositionEmbedding

from ._execution import jacobi
from .decoder_layer import FeedbackDecoderLayer, run_feedback_layers


class LCKVBlock(nn.Module):
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
        document_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = self.kv_pool.project_sequence(
            layer_hidden_states.transpose(1, 2).contiguous(), k_pos_emb
        )
        hidden, states = run_feedback_layers(
            self.layers, x_in, (keys[:, 0],), (values[:, 0],), q_pos_emb, document_ids=document_ids
        )
        return torch.stack(states, dim=1), hidden

    forward = jacobi.forward_jacobi

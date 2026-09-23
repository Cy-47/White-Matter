"""Compiled training loss over the same decoder used by the public model."""

import torch
from torch import nn

from training.losses import lm_cross_entropy_from_hidden
from white_matter.modules.documents import document_ids_from_eos


class TrainingForward(nn.Module):
    def __init__(self, model, *, checkpoint_chunk_size: int = 0, external_ce: bool = False):
        super().__init__()
        self.model = model
        self.checkpoint_chunk_size = checkpoint_chunk_size
        self.external_ce = external_ce
        object.__setattr__(self, "ar_graph", None)
        self.ar_graph_shape = None

    def capture_ar_graph(self, sample_hidden):
        from training.compile import capture_autoregressive_graph

        if self.checkpoint_chunk_size:
            raise ValueError("AR graph capture cannot be combined with activation checkpointing")
        if self.model.config.document_separator_token_id is None:
            raise ValueError("AR graph capture requires document_separator_token_id")
        # The model remains the sole registered owner of decoder parameters.
        object.__setattr__(self, "ar_graph", capture_autoregressive_graph(self.model.model.decoder, sample_hidden))
        self.ar_graph_shape = sample_hidden.shape

    @torch.compiler.disable
    def _replay_ar_graph(self, hidden, document_ids):
        return self.ar_graph(hidden, document_ids)

    def forward(self, hidden_states, num_passes: int, num_gradient_passes: int, *, token_ids, compute_ce=True):
        separator = self.model.config.document_separator_token_id
        document_ids = None if separator is None else document_ids_from_eos(token_ids, separator)
        if self.ar_graph is not None and hidden_states.shape == self.ar_graph_shape and self.training:
            hidden_states = self._replay_ar_graph(hidden_states, document_ids)
        else:
            hidden_states = self.model.model.decoder(
                hidden_states,
                num_passes=num_passes,
                num_gradient_passes=num_gradient_passes,
                document_ids=document_ids,
                checkpoint_chunk_size=self.checkpoint_chunk_size,
            )
        if not compute_ce:
            return hidden_states
        hidden_states = hidden_states.to(self.model.model.norm.weight.dtype)
        if self.external_ce:
            return self.model.model.norm(hidden_states)
        return lm_cross_entropy_from_hidden(
            token_ids, hidden=hidden_states, final_norm=self.model.model.norm, lm_head=self.model.lm_head
        )

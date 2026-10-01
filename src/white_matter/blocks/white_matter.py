"""Feedback region composed from ordinary PyTorch modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from white_matter._typing import compiler_disable
from white_matter.modules.documents import document_position_ids, feedback_document_mask
from white_matter.modules.kv_pool import KVPool
from white_matter.modules.rotary import PositionEmbedding

from ._execution import autoregressive, cyclic, jacobi
from ._execution.metadata import prepare_feedback_metadata
from .decoder_layer import FeedbackDecoderLayer, run_feedback_layers


class WhiteMatterBlock(nn.Module):
    """Cross-layer feedback with explicit layers, pool, and positional encoding.

    Supplied components own their initialization. Default PyTorch construction
    is valid without a Hugging Face post_init or any trainer setup.
    By default the pool needs L+1 sources: the block input and all L outputs.
    Set include_top_output=False with an L-source pool for paper reproduction.
    """

    def __init__(
        self,
        layers: Sequence[FeedbackDecoderLayer],
        kv_pool: KVPool,
        rotary_emb: PositionEmbedding,
        *,
        num_passes: int = 3,
        checkpoint_jacobi_passes: bool = False,
        include_top_output: bool = True,
    ) -> None:
        super().__init__()
        if type(include_top_output) is not bool:
            raise ValueError("include_top_output must be boolean")
        if not layers or len(layers) + int(include_top_output) != kv_pool.num_layers:
            raise ValueError("pool source depth must equal feedback depth plus include_top_output")
        self.include_top_output = include_top_output
        self.num_layers = len(layers)
        self.num_kv_channels = kv_pool.num_kv_channels
        self.num_passes = num_passes
        self.checkpoint_jacobi_passes = checkpoint_jacobi_passes
        self._cyclic_schedule_cache: dict[tuple[int, int, str], tuple[list[torch.Tensor], torch.Tensor]] = {}
        self._cyclic_rope_cache: dict[
            tuple[int, int, str, torch.dtype],
            tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]],
        ] = {}
        self.dummy_token = nn.Parameter(torch.zeros(kv_pool.hidden_size))
        self.layers = nn.ModuleList(layers)
        self.kv_pool = kv_pool
        self.rotary_emb = rotary_emb

    def _prepare_rope(
        self, x: torch.Tensor, document_ids: torch.Tensor | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        device = x.device
        # Slot 0 holds the learned dummy token; real token t is stored at slot t+1.
        if document_ids is None:
            q_pos = torch.arange(1, T + 1, device=device).unsqueeze(0)
            k_pos = torch.arange(T + 1, device=device).unsqueeze(0)
        else:
            # Real tokens start at position 1 behind the dummy token.
            q_pos = document_position_ids(document_ids) + 1
            k_pos = torch.cat([q_pos.new_zeros(B, 1), q_pos], dim=1)
        return self.rotary_emb(x, q_pos), self.rotary_emb(x, k_pos)

    forward = cyclic.forward_cyclic
    forward_jacobi = jacobi.forward_jacobi
    cyclic_pass = cyclic.run_pass

    def jacobi_pass(
        self,
        x_in: torch.Tensor,
        layer_hidden_states: torch.Tensor,
        q_pos_emb: tuple[torch.Tensor, torch.Tensor],
        k_pos_emb: tuple[torch.Tensor, torch.Tensor],
        document_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project (B,S,T,D) mixer sources, then sweep with fixed K/V."""
        K, V = self.kv_pool.project_sequence(
            layer_hidden_states.transpose(1, 2).contiguous(), k_pos_emb, dummy_token=self.dummy_token
        )
        # Preselect contiguous channel views once, shared by all their readers.
        keys = tuple(k[..., :-1, :].contiguous() for k in K.unbind(1))
        values = tuple(v[..., :-1, :].contiguous() for v in V.unbind(1))
        metadata = None
        if document_ids is not None:
            slots = torch.arange(x_in.shape[1], device=x_in.device)
            metadata = prepare_feedback_metadata(document_ids, [slots], x_in.shape[1])[0]
        hidden, states = run_feedback_layers(
            self.layers,
            x_in,
            keys,
            values,
            q_pos_emb,
            document_ids=document_ids,
            metadata=metadata,
            jacobi=True,
            include_top_output=self.include_top_output,
        )
        return torch.stack(states, dim=1), hidden

    def _autoregressive_step(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Differentiable token sweep and functional append for recurrent training."""
        output, keys, values = self._run_token_layers(
            x,
            state,
            self.rotary_emb(x, position_ids),
            attention_mask,
        )
        return output, (
            torch.cat((state[0], keys.transpose(0, 1)), dim=3),
            torch.cat((state[1], values.transpose(0, 1)), dim=3),
        )

    forward_autoregressive = compiler_disable(autoregressive.forward_autoregressive)

    def _packed_autoregressive_step(
        self,
        x_new: torch.Tensor,
        K_channel_past: torch.Tensor,
        V_channel_past: torch.Tensor,
        K_dummy: torch.Tensor,
        V_dummy: torch.Tensor,
        q_pos: torch.Tensor,
        valid_mask: torch.Tensor,
        reset_after: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Packed AR computation before the recurrent cache concatenation.

        Keeping the new one-slot K/V separate lets reverse-mode apply the
        concatenation VJP algebraically, avoiding full-prefix K/V outputs in the
        token-wise backward hot path.  All data-dependent inputs remain explicit.
        """
        B, _, _, N, _ = K_channel_past.shape
        q_pos_emb = self.rotary_emb(x_new, q_pos.view(B, 1).to(device=x_new.device, dtype=torch.long))
        add_mask = x_new.new_zeros((B, 1, 1, N)).masked_fill(
            ~valid_mask.view(B, 1, 1, N).to(x_new.device), torch.finfo(x_new.dtype).min
        )
        output, K_new, V_new = self._run_token_layers(x_new, (K_channel_past, V_channel_past), q_pos_emb, add_mask)
        reset_mask = reset_after.to(device=x_new.device, dtype=torch.bool).view(B, 1, 1, 1, 1)
        return (
            output,
            torch.where(reset_mask, K_dummy, K_new.transpose(0, 1)),
            torch.where(reset_mask, V_dummy, V_new.transpose(0, 1)),
        )

    def _run_token_layers(
        self,
        x: torch.Tensor,
        channels: tuple[torch.Tensor, torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        *,
        cache_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read the committed KV channels, run the layers, and project the new token."""
        x, layer_inputs = run_feedback_layers(
            self.layers,
            x,
            channels[0].unbind(1),
            channels[1].unbind(1),
            position_embeddings,
            include_top_output=self.include_top_output,
            decode_key_mask=attention_mask,
            cache_seqlens=cache_seqlens,
        )
        keys, values = self.kv_pool.project_token(layer_inputs, position_embeddings)
        return x, keys, values

    def _project_dummy(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the learned dummy token at position zero: (k,B,H,1,d)."""
        dummy = self.dummy_token.view(1, 1, -1).expand(batch_size, 1, -1)
        positions = torch.zeros(1, 1, device=dummy.device, dtype=torch.long)
        return self.kv_pool.project_token([dummy] * self.kv_pool.num_layers, self.rotary_emb(dummy, positions))

    def forward_recurrent(
        self,
        x: torch.Tensor,
        *,
        initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Continue exact AR from (B,k,H,N,d) K/V, including the leading dummy token.

        Positions use WM's convention (first real token is 1). An optional
        boolean mask has shape (B,T,N+T); it includes the dummy token and new tokens.
        Alternatively, document_ids labels all past/new token slots (B,N-1+T),
        excluding the dummy token; negative IDs mark padding. Document masks are built
        one token at a time to avoid a quadratic prefill allocation.
        Only committed prefix slots are read. Returned state includes every
        new token and retains its gradient history; callers own its lifetime.
        """
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError("recurrent inputs must be nonempty (B,T,D) tensors")
        if initial_state is None:
            keys, values = self._project_dummy(x.shape[0])
            initial_state = keys.transpose(0, 1), values.transpose(0, 1)
        keys, values = initial_state
        if keys.ndim != 5 or keys.shape != values.shape or keys.shape[:2] != (x.shape[0], self.num_kv_channels):
            raise ValueError("initial_state requires matching (B,k,H,N,d) K/V")
        if keys.shape[-2] < 1:
            raise ValueError("initial_state must include the dummy token")
        prefix = keys.shape[-2]
        if document_ids is not None and (
            document_ids.shape != (x.shape[0], prefix - 1 + x.shape[1])
            or document_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("document_ids must label the past and new token slots, excluding the dummy token")
        if document_ids is not None and position_ids is None:
            raise ValueError("document-aware recurrence requires explicit document-relative position_ids")
        if position_ids is None:
            position_ids = torch.arange(prefix, prefix + x.shape[1], device=x.device).unsqueeze(0)
        if position_ids.shape not in {(1, x.shape[1]), x.shape[:2]}:
            raise ValueError("position_ids must have shape (1,T) or (B,T)")
        if attention_mask is not None and (
            attention_mask.shape != (x.shape[0], x.shape[1], prefix + x.shape[1]) or attention_mask.dtype != torch.bool
        ):
            raise ValueError("attention_mask must be boolean (B,T,N+T), including the dummy token")
        outputs = []
        for t in range(x.shape[1]):
            token = x[:, t : t + 1]
            keep = None
            mask = None
            if attention_mask is not None:
                keep = attention_mask[:, t : t + 1, : prefix + t].unsqueeze(1)
            if document_ids is not None:
                document = document_ids[:, prefix - 1 + t : prefix + t]
                document_keep = feedback_document_mask(document_ids[:, : prefix + t - 1], document)[:, None, None]
                keep = document_keep if keep is None else keep & document_keep
            if keep is not None:
                mask = x.new_zeros(keep.shape).masked_fill(~keep, torch.finfo(x.dtype).min)
            hidden, (keys, values) = self._autoregressive_step(token, (keys, values), position_ids[:, t : t + 1], mask)
            outputs.append(hidden)
        return torch.cat(outputs, dim=1), (keys, values)

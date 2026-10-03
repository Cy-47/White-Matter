"""Feedback region composed from ordinary PyTorch modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from white_matter._typing import eager_loop, nested_compile_region
from white_matter.modules.documents import document_position_ids, feedback_document_mask
from white_matter.modules.kv_pool import KVPool
from white_matter.modules.rotary import PositionEmbedding
from white_matter.ops import StrictCausalMetadata

from ._execution import autoregressive, cyclic, jacobi
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
        use_dummy_token: bool = False,
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
        if type(use_dummy_token) is not bool:
            raise ValueError("use_dummy_token must be boolean")
        self.use_dummy_token = use_dummy_token
        self.dummy_token = nn.Parameter(torch.zeros(kv_pool.hidden_size)) if use_dummy_token else None
        for layer in layers:
            layer.self_attn.use_dummy_token = use_dummy_token
        self.layers = nn.ModuleList(layers)
        self.kv_pool = kv_pool
        self.rotary_emb = rotary_emb

    def _prepare_rope(
        self, x: torch.Tensor, document_ids: torch.Tensor | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        device = x.device
        offset = int(self.use_dummy_token)
        # Real positions follow the optional leading dummy slot.
        if document_ids is None:
            q_pos = torch.arange(offset, T + offset, device=device).unsqueeze(0)
            k_pos = torch.arange(T + offset, device=device).unsqueeze(0)
        else:
            # Document-relative real positions follow the optional dummy.
            q_pos = document_position_ids(document_ids) + offset
            k_pos = torch.cat([q_pos.new_zeros(B, offset), q_pos], dim=1)
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
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
        metadata: StrictCausalMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project (B,S,T,D) mixer sources, then sweep with fixed K/V."""
        K, V = self.kv_pool.project_sequence(
            layer_hidden_states.transpose(1, 2).contiguous(),
            k_pos_emb,
            dummy_token=self.dummy_token if past_key_values is None else None,
        )
        prefix_length = 0
        if past_key_values is not None:
            prefix_length = past_key_values[0].shape[-2] - int(self.use_dummy_token)
            K, V = (torch.cat((past, new), dim=3) for past, new in zip(past_key_values, (K, V), strict=True))
        # Preselect contiguous channel views once, shared by all their readers.
        keys = tuple(k.contiguous() for k in K.unbind(1))
        values = tuple(v.contiguous() for v in V.unbind(1))
        hidden, states = run_feedback_layers(
            self.layers,
            x_in,
            keys,
            values,
            q_pos_emb,
            metadata=metadata,
            jacobi=True,
            prefix_length=prefix_length,
            include_top_output=self.include_top_output,
        )
        return torch.stack(states, dim=1), hidden

    inference_jacobi_pass = nested_compile_region(jacobi_pass)

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

    forward_autoregressive = autoregressive.forward_autoregressive

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
            ~valid_mask.view(B, 1, 1, N).to(x_new.device), float("-inf")
        )
        output, K_new, V_new = self._run_token_layers(x_new, (K_channel_past, V_channel_past), q_pos_emb, add_mask)
        if not self.use_dummy_token:
            return output, K_new.transpose(0, 1), V_new.transpose(0, 1)
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
        static_cache: bool = False,
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
            static_cache=static_cache,
            committed_prefix=True,
        )
        keys, values = self.kv_pool.project_token(layer_inputs, position_embeddings)
        return x, keys, values

    def _initial_kv(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize (k,B,H,N,d) history; N is one with a dummy, otherwise zero."""
        if self.dummy_token is None:
            source = self.kv_pool.k_proj_weight
            empty = source.new_empty(batch_size, 0, self.kv_pool.hidden_size)
            positions = torch.empty(1, 0, device=source.device, dtype=torch.long)
            return self.kv_pool.project_token([empty] * self.kv_pool.num_layers, self.rotary_emb(empty, positions))
        dummy = self.dummy_token.view(1, 1, -1).expand(batch_size, 1, -1)
        positions = torch.zeros(1, 1, device=dummy.device, dtype=torch.long)
        return self.kv_pool.project_token([dummy] * self.kv_pool.num_layers, self.rotary_emb(dummy, positions))

    @eager_loop
    def forward_recurrent(
        self,
        x: torch.Tensor,
        *,
        initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Continue exact AR from (B,k,H,N,d) K/V, including the optional leading dummy token.

        Positions start at int(use_dummy_token). An optional
        boolean mask has shape (B,T,N+T); it includes any dummy slot and new tokens.
        Alternatively, document_ids labels all real past/new token slots (B,N-int(use_dummy_token)+T),
        excluding the dummy token; negative IDs mark padding. Document masks are built
        one token at a time to avoid a quadratic prefill allocation.
        Only committed prefix slots are read. Returned state includes every
        new token and retains its gradient history; callers own its lifetime.
        """
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError("recurrent inputs must be nonempty (B,T,D) tensors")
        if initial_state is None:
            keys, values = self._initial_kv(x.shape[0])
            initial_state = keys.transpose(0, 1), values.transpose(0, 1)
        keys, values = initial_state
        if keys.ndim != 5 or keys.shape != values.shape or keys.shape[:2] != (x.shape[0], self.num_kv_channels):
            raise ValueError("initial_state requires matching (B,k,H,N,d) K/V")
        offset = int(self.use_dummy_token)
        if keys.shape[-2] < offset:
            raise ValueError("initial_state must include the dummy token")
        prefix = keys.shape[-2]
        if document_ids is not None and (
            document_ids.shape != (x.shape[0], prefix - offset + x.shape[1])
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
            raise ValueError("attention_mask must be boolean (B,T,N+T), including any dummy slot")
        outputs = []
        for t in range(x.shape[1]):
            token = x[:, t : t + 1]
            keep = None
            mask = None
            if attention_mask is not None:
                keep = attention_mask[:, t : t + 1, : prefix + t].unsqueeze(1)
            if document_ids is not None:
                document = document_ids[:, prefix - offset + t : prefix - offset + t + 1]
                document_keep = feedback_document_mask(
                    document_ids[:, : prefix + t - offset], document, use_dummy_token=self.use_dummy_token
                )[:, None, None]
                keep = document_keep if keep is None else keep & document_keep
            if keep is not None:
                mask = x.new_zeros(keep.shape).masked_fill(~keep, float("-inf"))
            hidden, (keys, values) = self._autoregressive_step(token, (keys, values), position_ids[:, t : t + 1], mask)
            outputs.append(hidden)
        return torch.cat(outputs, dim=1), (keys, values)

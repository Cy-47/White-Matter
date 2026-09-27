"""Fan et al. feedback memory: one static mixture and one shared KV source."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from white_matter.modules.rotary import RotaryEmbedding, rotate_half

from .decoder_layer import FeedbackDecoderLayer


class FeedbackMemory(nn.Module):
    """Project a softmax mixture of the embedding and all layer outputs."""

    def __init__(self, hidden_size: int, num_layers: int, num_kv_heads: int, head_dim: int, eps: float) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers + 1))
        self.key = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.value = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.key_norm = nn.RMSNorm(head_dim, eps=eps)
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(
        self,
        states: list[torch.Tensor],
        position: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(states) != self.layer_logits.numel():
            raise ValueError("feedback memory requires the embedding and every layer output")
        weights = self.layer_logits.softmax(0).to(states[0].dtype)
        memory = sum(weight * state for weight, state in zip(weights, states, strict=True))
        shape = (memory.shape[0], memory.shape[1], self.num_kv_heads, self.head_dim)
        key = self.key_norm(self.key(memory).view(shape)).transpose(1, 2)
        value = self.value(memory).view(shape).transpose(1, 2)
        cos, sin = position
        key = key * cos.unsqueeze(1) + rotate_half(key) * sin.unsqueeze(1)
        return key.to(value.dtype), value


class FeedbackTransformerBlock(nn.Module):
    """Sequential token sweep; a token publishes its KV after all layers finish."""

    def __init__(self, layers: nn.ModuleList, memory: FeedbackMemory, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        self.layers = layers
        self.memory = memory
        self.rotary_emb = rotary_emb

    def token_step(
        self,
        token: torch.Tensor,
        keys: torch.Tensor | None,
        values: torch.Tensor | None,
        position: torch.Tensor,
        *,
        key_mask: torch.Tensor | None = None,
        cache_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rope = self.rotary_emb(token, position)
        states = [token]
        hidden = token
        for module in self.layers:
            layer = cast(FeedbackDecoderLayer, module)
            if keys is None:
                # The paper's attention context contains strictly earlier tokens.
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            else:
                assert values is not None
                hidden = layer(
                    hidden,
                    keys,
                    values,
                    rope,
                    decode_key_mask=key_mask,
                    cache_seqlens=cache_lengths,
                    committed_prefix=True,
                )
            states.append(hidden)
        key, value = self.memory(states, rope)
        return hidden, key, value

    def forward_reference(
        self,
        x: torch.Tensor,
        *,
        document_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable token sweep using a functional, growing prefix."""
        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError("expected nonempty (B,T,D) inputs")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise NotImplementedError("feedback reference currently requires unpadded inputs")
        keys = values = None
        outputs = []
        positions = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        if document_ids is not None:
            from white_matter.modules.documents import document_position_ids

            positions = document_position_ids(document_ids)
        for t in range(x.shape[1]):
            mask = None
            if document_ids is not None and t:
                keep = document_ids[:, :t] == document_ids[:, t : t + 1]
                mask = x.new_zeros((x.shape[0], 1, 1, t)).masked_fill(~keep[:, None, None], float("-inf"))
            out, key, value = self.token_step(
                x[:, t : t + 1],
                keys,
                values,
                positions[:, t : t + 1],
                key_mask=mask,
            )
            outputs.append(out)
            keys = key if keys is None else torch.cat((keys, key), dim=-2)
            values = value if values is None else torch.cat((values, value), dim=-2)
        assert keys is not None
        assert values is not None
        return torch.cat(outputs, dim=1), keys, values

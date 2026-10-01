"""Static consumer checks; mypy verifies both valid and rejected compositions.

Expected errors use narrow ignores. warn_unused_ignores makes the check fail
if a regression to Any stops rejecting an invalid call. Do not execute this file.
"""

from typing import assert_type

import torch
from torch import nn

from white_matter.blocks import FeedbackDecoderLayer, LCKVBlock, WhiteMatterBlock
from white_matter.layers import FeedbackAttention, WhiteMatterAttention
from white_matter.modules import FeedForward, GatedMLP, KVMixer, KVPool, PositionEmbedding, RotaryEmbedding
from white_matter.modules.routing import FixedSourceMixer
from white_matter.ops import CyclicAttentionMetadata, cyclic_attention, prepare_cyclic_attention_metadata


class CustomMLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sin()


class IncompatibleMLP(nn.Module):
    def forward(self, x: str) -> str:
        return x


def check_contracts(x: torch.Tensor, segments: torch.Tensor) -> None:
    attention = WhiteMatterAttention(32, 4, 8)
    attention_contract: FeedbackAttention = attention
    mlp_contract: FeedForward = GatedMLP(32, 64)
    custom_mlp_contract: FeedForward = CustomMLP()
    rope_contract: PositionEmbedding = RotaryEmbedding(8)
    mixer_contract: KVMixer = FixedSourceMixer(2)
    layer = FeedbackDecoderLayer(32, attention_contract, custom_mlp_contract)
    FeedbackDecoderLayer(32, attention_contract, mlp_contract)
    pool = KVPool(32, 2, 8, 1, 1)
    block = WhiteMatterBlock([layer], pool, rope_contract, include_top_output=False)
    KVPool(32, 2, 8, 2, 1, mixer=mixer_contract)
    hidden, state = block.forward(x, cyclic_groups=2)
    assert_type(hidden, torch.Tensor)
    assert_type(state, tuple[torch.Tensor, torch.Tensor] | None)
    assert_type(
        block.forward(x, cyclic_groups=2, output_final_state=True),
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None],
    )
    assert_type(block.forward_recurrent(x), tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]])
    assert_type(block.forward_jacobi(x), torch.Tensor)
    assert_type(block.forward_autoregressive(x), torch.Tensor)
    lckv_layer = FeedbackDecoderLayer(32, WhiteMatterAttention(32, 4, 8, strict_causal=True), CustomMLP())
    assert_type(LCKVBlock([lckv_layer], pool, rope_contract).forward(x), torch.Tensor)
    assert_type(rope_contract.forward(x, segments), tuple[torch.Tensor, torch.Tensor])
    assert_type(prepare_cyclic_attention_metadata(segments, segments), CyclicAttentionMetadata)
    assert_type(cyclic_attention(x, x, x, backend="reference"), torch.Tensor)

    WhiteMatterBlock([layer], pool, rope_contract, num_passes="3")  # type: ignore[arg-type]
    WhiteMatterBlock([attention], pool, rope_contract)  # type: ignore[list-item]
    FeedbackDecoderLayer(32, CustomMLP(), custom_mlp_contract)  # type: ignore[arg-type]
    FeedbackDecoderLayer(32, attention_contract, IncompatibleMLP())  # type: ignore[arg-type]
    WhiteMatterBlock([layer], pool, CustomMLP())  # type: ignore[arg-type]
    KVPool(32, 2, 8, 1, 1, mixer=CustomMLP())  # type: ignore[arg-type]
    block.forward_jacobi(x, num_gradient_passes="2")  # type: ignore[call-overload]
    block.forward_autoregressive(x, checkpoint_chunk_size="2")  # type: ignore[arg-type]
    cyclic_attention(x, x, x, backend="flash")  # type: ignore[arg-type]
    cyclic_attention(x, x, x, metadata=segments)  # type: ignore[arg-type]
    prepare_cyclic_attention_metadata([0, 1], segments)  # type: ignore[arg-type]

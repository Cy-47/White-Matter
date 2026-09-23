"""Compose a feedback region without Transformers, then optimize a custom loss.

Run from a checkout with the core installed: python examples/feedback_block.py
Replace the supplied attention/MLP modules to experiment with layer computation;
new state representations or schedules also require an appropriate block update.
"""

import torch

from white_matter.blocks import FeedbackDecoderLayer, WhiteMatterBlock
from white_matter.layers import WhiteMatterAttention
from white_matter.modules import GatedMLP, KVPool, RotaryEmbedding


def make_block(hidden_size=32, num_layers=4, num_kv_channels=2):
    layers = [
        FeedbackDecoderLayer(
            hidden_size, WhiteMatterAttention(hidden_size, 4, 8), GatedMLP(hidden_size, hidden_size * 2)
        )
        for i in range(num_layers)
    ]
    pool = KVPool(hidden_size, 2, 8, num_layers, num_kv_channels, router_prior="cyclic:0.25", router_layer_stride=2)
    return WhiteMatterBlock(layers, pool, RotaryEmbedding(8, 10_000.0), num_passes=3)


def main():
    torch.manual_seed(7)
    block = make_block()
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)
    # In a host model these tensors can come from embeddings or an encoder.
    conditioning = torch.randn(2, 16, 32, requires_grad=True)
    documents = torch.tensor([[0] * 8 + [1] * 8, [0] * 4 + [1] * 12])
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        conditioning.grad = None
        output, _ = block(conditioning, cyclic_groups=4, num_passes=3, num_gradient_passes=2, document_ids=documents)
        loss = output.square().mean()
        loss.backward()
        assert conditioning.grad is not None
        assert all(p.grad is not None for p in block.parameters())
        optimizer.step()
        print(f"loss={loss.item():.6f}")

    # Carry exact AR state across calls for a continuous sequence, without HF.
    with torch.no_grad():
        first, state = block.forward_recurrent(conditioning[:, :8])
        last, state = block.forward_recurrent(conditioning[:, 8:], initial_state=state)
        assert torch.cat((first, last), dim=1).shape == conditioning.shape
        assert state[0].shape[-2] == conditioning.shape[1] + 1  # dummy token plus real tokens


if __name__ == "__main__":
    main()

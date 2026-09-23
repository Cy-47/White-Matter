"""Standalone grouped-query cyclic attention with explicit document metadata."""

import torch

from white_matter.ops import cyclic_attention, prepare_cyclic_attention_metadata


def main():
    torch.manual_seed(7)
    query = torch.randn(1, 4, 8, 32, requires_grad=True)
    key = torch.randn(1, 2, 16, 32, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    # Query i belongs to slot 2*i. Key slot zero is a shared initial source.
    token_segments = torch.tensor([[0] * 8 + [1] * 8])
    metadata = prepare_cyclic_attention_metadata(
        token_segments[:, ::2], torch.cat((token_segments.new_full((1, 1), -1), token_segments[:, :-1]), dim=1)
    )
    output = cyclic_attention(query, key, value, query_stride=2, metadata=metadata)
    output.square().mean().backward()
    print(f"output={tuple(output.shape)}, gradients={[x.grad.norm().item() for x in (query, key, value)]}")


if __name__ == "__main__":
    main()

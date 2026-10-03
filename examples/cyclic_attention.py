"""Standalone grouped-query cyclic attention with explicit document metadata."""

import argparse

import torch

from white_matter.compilation import add_compile_argument, execution_policy
from white_matter.ops import cyclic_attention, prepare_cyclic_attention_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "tilelang"), default="reference")
    add_compile_argument(parser)
    args = parser.parse_args()
    with execution_policy(args.compile):
        run(args)


def run(args):
    device = "cuda" if args.backend == "tilelang" else "cpu"
    dtype = torch.bfloat16 if args.backend == "tilelang" else torch.float32
    torch.manual_seed(7)
    query = torch.randn(1, 4, 8, 64, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(1, 2, 16, 64, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    # Query i belongs to slot 2*i. Key slot zero is a shared initial source.
    token_segments = torch.tensor([[0] * 8 + [1] * 8], device=device)
    metadata = prepare_cyclic_attention_metadata(
        token_segments[:, ::2], torch.cat((token_segments.new_full((1, 1), -1), token_segments[:, :-1]), dim=1)
    )
    attention = torch.compile(cyclic_attention, fullgraph=True) if args.compile else cyclic_attention
    output = attention(query, key, value, query_stride=2, metadata=metadata, backend=args.backend)
    output.square().mean().backward()
    print(f"output={tuple(output.shape)}, gradients={[x.grad.norm().item() for x in (query, key, value)]}")


if __name__ == "__main__":
    main()

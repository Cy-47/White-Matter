"""Independent paper equations and cyclic scheduling, including every gradient."""

import copy
from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

from white_matter.blocks import FeedbackDecoderLayer, WhiteMatterBlock
from white_matter.layers import WhiteMatterAttention
from white_matter.modules import GatedMLP, KVPool, RotaryEmbedding


def paper_pool(pool, states, rope, dummy):
    """Literal per-channel equations; no Router or KVPool forward methods."""
    states = torch.cat((dummy[None, None, None].expand(states.shape[0], 1, pool.num_layers, -1), states), 1)
    outputs = []
    for branch in ("k", "v"):
        router = getattr(pool.mixer, f"{branch}_router")
        source = F.rms_norm(states, (pool.hidden_size,), eps=pool.rms_norm_eps)
        source = source * getattr(pool, f"pre_mix_{branch}_weight")
        selected = sorted(range(pool.num_layers - 1, -1, -router.layer_stride))
        context = source[:, :, selected].flatten(2)
        weights = F.linear(context, router.linear.weight, router.linear.bias)
        weights = weights.unflatten(-1, (pool.num_kv_channels, pool.num_layers))
        channels = []
        for channel in range(pool.num_kv_channels):
            mixed = (source * weights[:, :, channel, :, None]).sum(2)
            mixed = F.rms_norm(mixed, (pool.hidden_size,), eps=pool.mix_norm_eps)
            mixed = mixed * pool.post_mix[f"{branch}_gain"][channel]
            projected = F.linear(mixed, getattr(pool, f"{branch}_proj_weight")[channel])
            projected = projected.unflatten(-1, (pool.num_key_value_heads, pool.head_dim)).transpose(1, 2)
            if branch == "k":
                projected = F.rms_norm(projected, (pool.head_dim,), eps=pool.rms_norm_eps)
                projected = projected * pool.k_norm_weight[channel]
                first, second = projected.chunk(2, -1)
                projected = projected * rope[0][:, None] + torch.cat((-second, first), -1) * rope[1][:, None]
            channels.append(projected)
        outputs.append(torch.stack(channels, 1))
    return tuple(outputs)


def paper_cyclic(block, x, documents, *, passes, gradient_passes, groups):
    """Rebuild the entire pool from layer states before each group.

    This reference uses explicit visibility, without compact caches, kernel
    metadata, in-place KV publication, pass shortcuts, or checkpointing.
    """
    qrope, krope = block._prepare_rope(x, documents)
    states = x[:, :, None].expand(-1, -1, block.kv_pool.num_layers, -1)
    output = torch.zeros_like(x)
    length = x.shape[1]
    keys = torch.arange(length + 1)
    for iteration in range(passes):
        if iteration == passes - gradient_passes and iteration:
            states = states.detach()
        with torch.no_grad() if iteration < passes - gradient_passes else nullcontext():
            for offset in range(groups):
                slots = torch.arange(offset, length, groups)
                key, value = paper_pool(block.kv_pool, states, krope, block.dummy_token)
                keep = keys[None] <= slots[:, None]
                if documents is not None:
                    same = documents[:, slots, None] == documents[:, None, :]
                    keep = keep & torch.cat((torch.ones_like(same[:, :, :1]), same), -1)
                hidden, fresh = x[:, slots], []
                for index, layer in enumerate(block.layers):
                    fresh.append(hidden)
                    channel = index % block.num_kv_channels
                    hidden = layer(
                        hidden,
                        key[:, channel],
                        value[:, channel],
                        tuple(t[:, slots] for t in qrope),
                        decode_key_mask=keep.unsqueeze(-3),
                    )
                if block.include_top_output:
                    fresh.append(hidden)
                states = states.index_copy(1, slots, torch.stack(fresh, 2))
                output = output.index_copy(1, slots, hidden)
    return output


@pytest.mark.parametrize("include_top_output", [False, True])
@pytest.mark.parametrize("channels", [1, 2, 4])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("gradient_passes", [2, 3])
def test_cyclic_matches_paper_equations_and_all_gradients(channels, packed, gradient_passes, include_top_output):
    torch.manual_seed(608)
    block = WhiteMatterBlock(
        [FeedbackDecoderLayer(16, WhiteMatterAttention(16, 2, 8), GatedMLP(16, 24)) for _ in range(4)],
        KVPool(16, 1, 8, 4 + int(include_top_output), channels, router_layer_stride=2),
        RotaryEmbedding(8),
        include_top_output=include_top_output,
        use_dummy_token=True,
    ).double()
    # Nonzero routing weights, normalization gains, and boundary states are
    # essential: step-zero priors alone cannot test content-dependent routing.
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.03)
    reference = copy.deepcopy(block)
    inputs = torch.randn(2, 11, 16, dtype=torch.float64)
    probe = torch.randn_like(inputs)
    documents = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], [0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4]]) if packed else None
    records = []
    # Disable compilation, but retain the production selective checkpoints so
    # their recomputation and all explicit document inputs are exercised.
    with torch.compiler.set_stance("force_eager"):
        for current, manual in ((reference, True), (block, False)):
            x = inputs.clone().requires_grad_()
            output = (
                paper_cyclic(current, x, documents, passes=3, gradient_passes=gradient_passes, groups=3)
                if manual
                else current(
                    x, num_passes=3, num_gradient_passes=gradient_passes, cyclic_groups=3, document_ids=documents
                )[0]
            )
            (output * probe).sum().backward()
            grads = {name: p.grad for name, p in current.named_parameters()}
            assert x.grad is not None
            assert all(g is not None for g in grads.values())
            records.append((output.detach(), x.grad, grads))
    torch.testing.assert_close(*records, rtol=1e-9, atol=1e-9)

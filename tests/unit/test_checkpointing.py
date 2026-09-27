"""Rematerialized AR must retain all input, dummy-token, reset and parameter gradients."""

import copy

import pytest
import torch

from examples.feedback_block import make_block


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    ("chunk_size", "split", "batch_size"), [(1, False, 0), (3, False, 1), (3, True, 0), (3, True, 1)]
)
def test_ar_checkpoint_matches_uncheckpointed(packed, chunk_size, split, batch_size):
    torch.manual_seed(83)
    block = make_block().double()
    reference = copy.deepcopy(block)
    inputs = torch.randn(2, 7, 32, dtype=torch.float64)
    probe = torch.randn_like(inputs)
    # Resets within and across chunk boundaries, with an uneven final chunk.
    docs = torch.tensor([[0, 0, 1, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2, 3]]) if packed else None
    results = []
    for current, checkpoint in ((reference, 0), (block, chunk_size)):
        x = inputs.clone().requires_grad_()
        hidden = current.forward_autoregressive(
            x,
            document_ids=docs,
            checkpoint_chunk_size=checkpoint,
            backward_batch_size=batch_size,
            split_state_vjp=split,
        )
        (hidden * probe).sum().backward()
        gradients = {n: p.grad for n, p in current.named_parameters()}
        assert x.grad is not None
        assert all(g is not None for g in gradients.values())
        results.append((hidden.detach(), x.grad, gradients))
    torch.testing.assert_close(*results, rtol=1e-10, atol=1e-10)

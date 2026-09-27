"""Checkpointed vocabulary projection preserves every explicit input gradient."""

import pytest
import torch

from training.losses import checkpointed_linear_cross_entropy


@pytest.mark.parametrize("bias", [False, True])
def test_checkpointed_loss_matches_dense_loss_and_all_gradients(bias):
    torch.manual_seed(61)
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    head = torch.nn.Linear(8, 13, bias=bias)
    labels = torch.randint(0, 13, (2, 7))
    actual = checkpointed_linear_cross_entropy(hidden, labels, head, token_chunk_size=3)
    expected = torch.nn.functional.cross_entropy(head(hidden[:, :-1]).reshape(-1, 13), labels[:, 1:].reshape(-1))
    inputs = (hidden, *head.parameters())
    actual_gradients = torch.autograd.grad(actual, inputs)
    expected_gradients = torch.autograd.grad(expected, inputs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradients, expected_gradients)

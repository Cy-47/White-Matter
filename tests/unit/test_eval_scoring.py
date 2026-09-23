import pytest
import torch
import torch.nn.functional as F

from evals.scoring import score_tokens


@pytest.mark.parametrize("chunk_size", [1, 4, 32])
def test_scoring_matches_dense_without_retaining_logits(chunk_size, monkeypatch):
    torch.manual_seed(17)
    hidden = torch.randn(11, 8, requires_grad=True)
    weight = torch.randn(19, 8, requires_grad=True)
    targets = torch.arange(11)
    logits = F.linear(hidden, weight)
    expected = logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
    linear = F.linear
    chunks = []

    def bounded_linear(inputs, weight):
        assert not torch.is_grad_enabled()
        chunks.append(inputs.shape[0])
        return linear(inputs, weight)

    monkeypatch.setattr(F, "linear", bounded_linear)
    scores, greedy = score_tokens(hidden, weight, targets, chunk_size=chunk_size)
    torch.testing.assert_close(scores, expected)
    torch.testing.assert_close(greedy, logits.argmax(-1) == targets)
    assert max(chunks) <= chunk_size
    assert not scores.requires_grad

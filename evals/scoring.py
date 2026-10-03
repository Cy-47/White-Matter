"""Token likelihoods with bounded vocabulary workspace."""

import torch
import torch.nn.functional as F


@torch.compile(dynamic=True, options={"emulate_precision_casts": True})
def _score_chunk(hidden, weight, target):
    logits = F.linear(hidden.to(weight.dtype), weight)
    return -F.cross_entropy(logits.float(), target, reduction="none"), logits.argmax(-1) == target


@torch.inference_mode()
def score_tokens(hidden, weight, targets, *, chunk_size=256):
    """Return target log probabilities and greedy matches for flat token inputs."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    scores = torch.empty(targets.shape, dtype=torch.float32, device=hidden.device)
    greedy = torch.empty_like(targets, dtype=torch.bool)
    for start in range(0, targets.numel(), chunk_size):
        stop = start + chunk_size
        scores[start:stop], greedy[start:stop] = _score_chunk(hidden[start:stop], weight, targets[start:stop])
    return scores, greedy

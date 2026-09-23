"""Token likelihoods with bounded vocabulary workspace."""

import torch
import torch.nn.functional as F


@torch.inference_mode()
def score_tokens(hidden, weight, targets, *, chunk_size=256):
    """Return target log probabilities and greedy matches for flat token inputs."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    scores = torch.empty(targets.shape, dtype=torch.float32, device=hidden.device)
    greedy = torch.empty_like(targets, dtype=torch.bool)
    for start in range(0, targets.numel(), chunk_size):
        stop = start + chunk_size
        logits = F.linear(hidden[start:stop].to(weight.dtype), weight)
        target = targets[start:stop]
        scores[start:stop] = -F.cross_entropy(logits.float(), target, reduction="none")
        greedy[start:stop] = logits.argmax(-1) == target
    return scores, greedy

"""Language-model losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def cce_linear_cross_entropy(
    hidden_normed: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Module,
) -> torch.Tensor:
    """Logit-free next-token CE with unfiltered, FP32-accumulated gradients."""
    from cut_cross_entropy import linear_cross_entropy

    return linear_cross_entropy(
        hidden_normed, lm_head.weight, labels,
        bias=getattr(lm_head, "bias", None), shift=True, impl="cce_exact",
    )


def lm_cross_entropy_from_hidden(
    labels: torch.Tensor,
    *,
    hidden: torch.Tensor,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
) -> torch.Tensor:
    """Next-token CE with norm/projection inside the caller's compiled region.

    Slice the label-less final position before the vocabulary projection.
    """
    shifted_logits = lm_head(final_norm(hidden)[:, :-1, :])
    return F.cross_entropy(shifted_logits.float().reshape(-1, shifted_logits.shape[-1]), labels[:, 1:].reshape(-1))


def checkpointed_linear_cross_entropy(
    hidden_normed: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    token_chunk_size: int = 4096,
) -> torch.Tensor:
    """Bound vocabulary logits by checkpointing flat token chunks.

    Recompute the same FP32 cross-entropy in backward. Activations, classifier
    parameters, and targets are explicit inputs, including across the AR boundary.
    """
    token_chunk_size = int(token_chunk_size)
    if token_chunk_size < 1:
        raise ValueError(f"token_chunk_size must be positive, got {token_chunk_size}")
    hidden = hidden_normed[:, :-1, :].reshape(-1, hidden_normed.shape[-1])
    targets = labels[:, 1:].reshape(-1)
    weight = lm_head.weight
    bias = getattr(lm_head, "bias", None)

    def chunk_loss(h, w, b, target):
        return F.cross_entropy(F.linear(h, w, b).float(), target, reduction="sum")

    losses: list[torch.Tensor] = []
    for start in range(0, targets.numel(), token_chunk_size):
        end = min(targets.numel(), start + token_chunk_size)
        losses.append(
            checkpoint(
                chunk_loss,
                hidden[start:end],
                weight,
                bias,
                targets[start:end],
                use_reentrant=False,
                preserve_rng_state=False,
                determinism_check="none",
            )
        )
    return torch.stack(losses).sum() / targets.numel()

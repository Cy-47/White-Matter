"""Scoped inference settings and token-weighted scoring shared by experiments."""

from contextlib import contextmanager, nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from evals.scoring import score_tokens
from white_matter.modules.precision import fp32_inference


@contextmanager
def execution(model, *, mode=None, passes=None, groups=None, precision="bf16"):
    """Restore all settings on exit; FP32 uses eager reference attention throughout."""
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if mode not in {None, "cyclic", "jacobi", "autoregressive"}:
        raise ValueError("unsupported execution mode")
    if any(value is not None and (type(value) is not int or value < 1) for value in (passes, groups)):
        raise ValueError("passes and groups must be positive integers")
    changes = []

    def set_value(obj, key, value):
        changes.append((obj, key, hasattr(obj, key), getattr(obj, key, None)))
        setattr(obj, key, value)

    try:
        for key, value in (("execution_mode", mode), ("num_passes", passes), ("cyclic_groups", groups)):
            if value is not None:
                set_value(model.config, key, value)
        if precision == "fp32":
            if any(p.dtype != torch.float32 for p in model.parameters()):
                raise ValueError("FP32 reference requires FP32 parameters")
            set_value(model.config, "_attn_implementation", "sdpa")
            from white_matter.layers.white_matter import WhiteMatterAttention

            for module in model.modules():
                if isinstance(module, WhiteMatterAttention):
                    set_value(module, "attention_implementation", "sdpa")
                    set_value(module, "_force_jacobi_reference", True)
                    set_value(module, "_force_cyclic_reference", True)
        with (fp32_inference() if precision == "fp32" else nullcontext()), (
            sdpa_kernel(SDPBackend.MATH) if precision == "fp32" else nullcontext()
        ):
            yield
    finally:
        for obj, key, existed, value in reversed(changes):
            if existed:
                setattr(obj, key, value)
            else:
                delattr(obj, key)


@torch.inference_mode()
def score_hidden(hidden, ids, weight, chunk_size=256) -> float:
    scores, _ = score_tokens(
        hidden[:, :-1].reshape(-1, hidden.shape[-1]), weight,
        ids[:, 1:].reshape(-1), chunk_size=chunk_size,
    )
    return float(-scores.double().sum())


@torch.inference_mode()
def evaluate(model, loader, *, num_passes=None, loss_backend="torch", chunk_size=256):
    """Return summed CE and target count without averaging batch perplexities."""
    if loss_backend not in {"torch", "cce"}:
        raise ValueError("loss_backend must be torch or cce")
    device = next(model.parameters()).device
    total, count = 0.0, 0
    model.eval()
    for batch in loader:
        ids = batch["input_ids"].to(device)
        hidden = model.model(ids, num_passes=num_passes, use_cache=False).last_hidden_state
        if loss_backend == "cce" and device.type == "cuda":
            from cut_cross_entropy import linear_cross_entropy
            from white_matter.modules.precision import model_autocast_context

            with model_autocast_context(device):
                total += float(linear_cross_entropy(
                    hidden, model.lm_head.weight, ids, shift=True, reduction="sum", impl="cce_exact",
                ).double())
        else:
            total += score_hidden(hidden, ids, model.lm_head.weight, chunk_size)
        count += ids.shape[0] * (ids.shape[1] - 1)
    if count == 0:
        raise ValueError("evaluation contains no prediction targets")
    return total, count

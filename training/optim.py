"""The shape-batched Muon optimizer used by the paper recipes."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz


def _batched_zeropower_via_newtonschulz(
    gradients: list[torch.Tensor],
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> list[torch.Tensor]:
    """Orthogonalize one equal-shape group with batched GEMMs."""
    if len(gradients) == 1:
        return [_zeropower_via_newtonschulz(gradients[0], ns_coefficients, ns_steps, eps)]

    # Allocate the BF16 workspace directly. Stacking in FP32 before casting
    # creates a second full-size temporary for the transformer matrix groups.
    workspace = torch.empty(
        (len(gradients), *gradients[0].shape),
        device=gradients[0].device,
        dtype=torch.bfloat16,
    )
    torch.stack(gradients, out=workspace)
    if ns_steps >= 100:
        raise ValueError("Newton--Schulz steps must be less than 100")
    workspace.div_(workspace.norm(dim=(-2, -1), keepdim=True).clamp(min=eps))
    a, b, c = ns_coefficients
    for _ in range(ns_steps):
        gram = torch.bmm(workspace, workspace.transpose(-2, -1))
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        workspace = torch.baddbmm(workspace, gram_update, workspace, beta=a)
    return list(workspace.unbind(0))


class BatchedMuon(torch.optim.Muon):
    """Muon with the paper implementation's shape-batched NS update.

    Parameters with equal orientation-independent shapes share batched GEMMs.
    The Nesterov blend reuses gradient storage, which is dead after ``step``.
    All other equations, parameter groups, state dictionaries, and learning-rate
    adjustment follow :class:`torch.optim.Muon`.
    """

    _orthogonalize = staticmethod(_batched_zeropower_via_newtonschulz)

    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        parameter_ids = [id(parameter) for group in self.param_groups for parameter in group["params"]]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("BatchedMuon requires each parameter to appear once")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            parameters: list[torch.Tensor] = []
            gradients: list[torch.Tensor] = []
            momentum_buffers: list[torch.Tensor] = []
            self._init_group(group, parameters, gradients, momentum_buffers)

            by_shape: dict[
                tuple[int, int],
                list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]],
            ] = defaultdict(list)
            for parameter, gradient, momentum_buffer in zip(parameters, gradients, momentum_buffers, strict=True):
                transposed = gradient.shape[0] > gradient.shape[1]
                by_shape[tuple(sorted(gradient.shape))].append((parameter, gradient, momentum_buffer, transposed))

            learning_rate = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            adjust_lr_fn = group["adjust_lr_fn"]

            for shape_group in by_shape.values():
                shape_gradients = [entry[1] for entry in shape_group]
                shape_buffers = [entry[2] for entry in shape_group]
                torch._foreach_lerp_(shape_buffers, shape_gradients, 1 - momentum)

                if group["nesterov"]:
                    # The gradients are not consumed after optimizer.step().
                    torch._foreach_lerp_(shape_gradients, shape_buffers, momentum)
                    momentum_updates = shape_gradients
                else:
                    momentum_updates = shape_buffers

                normalized_updates = [
                    update.T if entry[3] else update
                    for entry, update in zip(shape_group, momentum_updates, strict=True)
                ]
                normalized_orthogonal = self._orthogonalize(
                    normalized_updates,
                    group["ns_coefficients"],
                    int(group["ns_steps"]),
                    float(group["eps"]),
                )
                updates = [
                    update.T if entry[3] else update
                    for entry, update in zip(shape_group, normalized_orthogonal, strict=True)
                ]

                # Transpose-equivalent matrices share NS, but the original LR
                # adjustment depends on the parameter's untransposed shape.
                update_by_shape: dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
                for entry, update in zip(shape_group, updates, strict=True):
                    update_by_shape[tuple(entry[0].shape)].append((entry[0], update))

                for original_shape, entries in update_by_shape.items():
                    adjusted_lr = _adjust_lr(learning_rate, adjust_lr_fn, original_shape)
                    update_parameters = [entry[0] for entry in entries]
                    shape_updates = [entry[1] for entry in entries]
                    torch._foreach_mul_(update_parameters, 1 - learning_rate * weight_decay)
                    torch._foreach_add_(update_parameters, shape_updates, alpha=-adjusted_lr)

                # Unbound views retain the complete BF16 workspace. Release
                # them before allocating the next equal-shape batch.
                del normalized_orthogonal, updates, update_by_shape, update, entries, shape_updates

        return loss


class CompiledBatchedMuon(BatchedMuon):
    """Compile only Newton--Schulz while preserving its eager BF16 casts."""

    _orthogonalize = staticmethod(
        torch.compile(
            _batched_zeropower_via_newtonschulz,
            fullgraph=True,
            dynamic=False,
            options={"emulate_precision_casts": True},
        )
    )


@torch.no_grad()
def clip_grad_norm_if_needed_(
    parameters,
    max_norm: float,
    *,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """Match ``clip_grad_norm_`` but skip its unconditional multiply by one.

    PyTorch intentionally avoids a device-to-host conditional and always runs
    the foreach multiply, even when the clamped coefficient is exactly one.
    This training loop already synchronizes on the returned norm for logging and
    skip logic; using that same scalar here avoids writing the full multi-billion
    parameter gradient vector on ordinary non-clipping steps. Non-finite norms
    still execute PyTorch's clipping primitive so NaN/Inf propagation is
    unchanged.
    """
    parameter_list = list(parameters)
    grads = [parameter.grad for parameter in parameter_list if parameter.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads,
        norm_type=norm_type,
        error_if_nonfinite=False,
        foreach=None,
    )
    clip_coefficient = float(max_norm) / (total_norm + 1.0e-6)
    coefficient_value = float(clip_coefficient)
    if not math.isfinite(coefficient_value) or coefficient_value < 1.0:
        torch.nn.utils.clip_grads_with_norm_(
            parameter_list,
            max_norm=float(max_norm),
            total_norm=total_norm,
            foreach=None,
        )
    return total_norm


def post_clip_norm_for_monitoring(
    preclip_norm: torch.Tensor,
    max_norm: float,
) -> torch.Tensor:
    """Return the finite post-clip norm without hiding NaN/Inf failures.

    ``clamp_max`` is the inexpensive finite-path equivalent used when the
    just-computed mini-batch clip can be reused.  Applied unconditionally,
    however, it maps ``+inf`` to ``max_norm`` and would let the optimizer step
    gradients that the clipping multiply has already poisoned with NaNs.
    Preserve every non-finite value so the training loop's skip gate sees it.
    """
    return torch.where(
        torch.isfinite(preclip_norm),
        preclip_norm.clamp_max(float(max_norm)),
        preclip_norm,
    )


def learning_rate_multiplier(
    step: int,
    total: int,
    *,
    schedule: str,
    warmup_frac: float,
    floor_frac: float,
) -> float:
    """Constant LR, or warmup followed by cosine decay to a fixed floor."""
    schedule = str(schedule).strip().lower()
    if schedule == "constant":
        return 1.0
    if schedule == "cosine":
        warmup = max(1, int(total * warmup_frac))
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return floor_frac + (1.0 - floor_frac) * cosine
    raise ValueError(f"unknown lr_schedule={schedule!r}; expected 'cosine' or 'constant'")


def is_no_decay_parameter(name: str) -> bool:
    """Return whether a decoder parameter follows the no-decay rule.

    ``name`` is relative to ``model.model.decoder``. Pool normalization gains use
    stacked layouts, so shape alone cannot distinguish them from matrices that
    should receive weight decay.
    """
    if name.startswith("fusions."):
        return True
    if name.endswith(".weight") and (
        ".input_layernorm" in name or ".post_attention_layernorm" in name or ".q_norm" in name or ".k_norm" in name
    ):
        return True
    if name.endswith(
        (
            "pre_mix_k_weight",
            "pre_mix_v_weight",
            "pre_mix_weight",
            "k_norm_weight",
            "k_gain",
            "v_gain",
        )
    ):
        return True
    if name.endswith((".bias", "_bias")):
        return True
    return name.endswith("dummy_token")


def partition_optimizer_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """Partition a causal LM exactly as the production trainer does.

    Decoder parameters retain their ``named_parameters()`` traversal order.
    Embeddings and final normalization are appended to ``main_nodecay`` last.
    The tied LM head shares the embedding parameter and is counted once.
    """
    main_decay: list[nn.Parameter] = []
    main_nodecay: list[nn.Parameter] = []
    muon_main: list[nn.Parameter] = []

    for name, parameter in model.model.decoder.named_parameters():
        if not parameter.requires_grad:
            continue
        nodecay = is_no_decay_parameter(name)
        if not nodecay and parameter.dim() == 2:
            muon_main.append(parameter)
        else:
            (main_nodecay if nodecay else main_decay).append(parameter)

    main_nodecay.extend(p for p in (model.get_input_embeddings().weight, model.model.norm.weight) if p.requires_grad)
    return main_decay, main_nodecay, muon_main


@dataclass(frozen=True)
class TrainingOptimizers:
    """Optimizers and ordered parameter partition used by production training."""

    adamw: torch.optim.Optimizer
    muon: torch.optim.Optimizer | None
    parameters: list[nn.Parameter]
    base_lr: float


def build_optimizers(
    model: nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    muon_momentum: float,
    muon_ns_steps: int,
    device: str | torch.device,
    distributed_muon: bool = False,
    compiled: bool = True,
) -> TrainingOptimizers:
    """Build the exact production AdamW/Muon parameter groups and options."""

    main_decay, main_nodecay, muon_main = partition_optimizer_parameters(model)
    base_lr = float(base_lr)
    weight_decay = float(weight_decay)

    adamw_groups = [
        {
            "params": main_decay,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "main_decay",
        },
        {
            "params": main_nodecay,
            "lr": base_lr,
            "weight_decay": 0.0,
            "name": "main_nodecay",
        },
    ]
    adamw = torch.optim.AdamW(
        adamw_groups,
        betas=(float(adam_beta1), float(adam_beta2)),
        fused=str(device).startswith("cuda"),
    )

    muon = None
    if muon_main:
        inner_muon_class = CompiledBatchedMuon if compiled and torch.device(device).type == "cuda" else BatchedMuon
        muon_class = inner_muon_class
        sharding_options = {}
        if distributed_muon:
            import torch.distributed as dist
            from torch.distributed.optim import ZeroRedundancyOptimizer

            if not dist.is_initialized() or dist.get_world_size() < 2:
                raise ValueError("distributed_muon requires a process group with at least two ranks")
            muon_class = ZeroRedundancyOptimizer
            sharding_options = {"optimizer_class": inner_muon_class}
        muon = muon_class(
            [{"params": muon_main, "lr": base_lr, "name": "muon_main"}],
            momentum=float(muon_momentum),
            nesterov=True,
            weight_decay=weight_decay,
            ns_steps=muon_ns_steps,
            adjust_lr_fn="match_rms_adamw",
            **sharding_options,
        )

    parameters = main_decay + main_nodecay + muon_main
    optimizer_parameter_ids = [id(parameter) for parameter in parameters]
    trainable_parameter_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    if len(set(optimizer_parameter_ids)) != len(optimizer_parameter_ids):
        raise RuntimeError("optimizer traversal contains a duplicate parameter")
    if set(optimizer_parameter_ids) != set(trainable_parameter_ids):
        raise RuntimeError("optimizer union must match the unique trainable model-parameter identity set exactly")

    return TrainingOptimizers(
        adamw=adamw,
        muon=muon,
        parameters=parameters,
        base_lr=base_lr,
    )


def set_learning_rate(
    optimizers: TrainingOptimizers,
    *,
    step: int,
    total_steps: int,
    schedule: str,
    warmup_frac: float,
    floor_frac: float,
) -> float:
    """Assign every production optimizer-group LR for one schedule index."""

    current_lr = optimizers.base_lr * learning_rate_multiplier(
        step,
        total_steps,
        schedule=schedule,
        warmup_frac=warmup_frac,
        floor_frac=floor_frac,
    )
    active_optimizers = (optimizers.adamw,) if optimizers.muon is None else (optimizers.adamw, optimizers.muon)
    for optimizer in active_optimizers:
        for group in optimizer.param_groups:
            group["lr"] = current_lr
    return current_lr


def step_optimizers(optimizers: TrainingOptimizers) -> None:
    """Apply Muon then AdamW and release each disjoint gradient set promptly."""

    if optimizers.muon is not None:
        optimizers.muon.step()
        # AdamW cannot read Muon's gradients because the factory validates a
        # disjoint exhaustive partition. Release them before fused AdamW lazily
        # creates its moments on the first update.
        optimizers.muon.zero_grad(set_to_none=True)
    optimizers.adamw.step()
    optimizers.adamw.zero_grad(set_to_none=True)

"""Workload compilation and training CUDA graph capture."""

from __future__ import annotations

import warnings
from typing import Any

import torch

_COMPILE_OPTIONS = {"emulate_precision_casts": True}


def _packed_ar_backend(graph, inputs):
    from torch._dynamo.backends.registry import lookup_backend

    warnings.warn(
        "Packed autoregressive execution uses aot_eager: autograd graphs are captured, "
        "while tensor kernels remain eager to preserve the validated gradient behavior.",
        RuntimeWarning,
        stacklevel=2,
    )
    return lookup_backend("aot_eager")(graph, inputs)


class _AutoregressiveDecoder(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, hidden, document_ids):
        return self.decoder(hidden, document_ids=document_ids)


def capture_autoregressive_graph(decoder, sample_hidden: torch.Tensor) -> torch.nn.Module:
    """Capture fixed-shape AR forward/backward with live weights and document inputs."""
    if decoder.config.execution_mode != "autoregressive" or not sample_hidden.is_cuda:
        raise ValueError("AR graph capture requires an autoregressive decoder on CUDA")
    if decoder.config.num_pre_layers or decoder.config.num_post_layers:
        raise ValueError("AR graph capture currently requires a pure feedback decoder")
    sample_hidden = sample_hidden.detach().requires_grad_(True)
    documents = torch.zeros(sample_hidden.shape[:2], device=sample_hidden.device, dtype=torch.long)
    # Replay must execute fresh casts after each optimizer update.
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        graph = torch.cuda.make_graphed_callables(
            _AutoregressiveDecoder(decoder).train(),
            (sample_hidden, documents),
        )
    return graph


def patch_inductor_skip_repro_deepcopy() -> None:
    """Skip the unused FX graph deepcopy when Inductor repro is disabled."""
    import functools

    import torch._inductor.compile_fx as cfx
    from torch._dynamo import config as dcfg

    if getattr(cfx, "_repro_deepcopy_patched", False):
        return

    orig_wrap = cfx.wrap_compiler_debug

    @functools.wraps(orig_wrap)
    def wrap_compiler_debug(compiler_fn, compiler_name):
        return compiler_fn if dcfg.repro_after is None else orig_wrap(compiler_fn, compiler_name)

    cfx.wrap_compiler_debug = wrap_compiler_debug
    cfx._repro_deepcopy_patched = True


def configure_training_compilation(
    model: Any,
    *,
    ar_dynamic: bool = False,
) -> None:
    """Configure the packed-AR exception before compiling the enclosing workload.

    Repeated passes use nested regions in the model implementation. Packed AR
    checkpoints retain their separately validated AOTAutograd backend.
    """
    decoder = model.model.decoder
    block = getattr(decoder, "block", decoder)  # Baselines may own their layers directly.
    # Remove the dead per-compile FX-graph deepcopy before any torch.compile runs.
    patch_inductor_skip_repro_deepcopy()

    # EOS-packed exact-AR training uses AOTAutograd's eager backend for the
    # tensor-state step.  It removes the Python module/layer traversal and
    # pre-builds backward while preserving eager ATen CUDA kernels. Token-step
    # graph boundaries can still change BF16 recurrent-gradient accumulation
    # relative to a fully unrolled eager graph. Changing this backend requires
    # a production gradient comparison.
    if hasattr(block, "_packed_autoregressive_step"):
        block._packed_autoregressive_step = torch.compile(  # type: ignore[method-assign]
            block._packed_autoregressive_step,
            backend=_packed_ar_backend,
            fullgraph=False,
            dynamic=ar_dynamic,
            # Config overrides are thread-local; checkpoint backward otherwise
            # sees the default limit of 8 on its autograd worker.
            recompile_limit=max(64, torch._dynamo.config.recompile_limit),
        )


def compile_training_forward(module: torch.nn.Module) -> torch.nn.Module:
    """Compile the complete training loss with eager-compatible casts."""
    return torch.compile(
        module,
        fullgraph=False,
        dynamic=False,
        options=dict(_COMPILE_OPTIONS),
    )


def compile_evaluation(model: Any, *, eager_pass_loop: bool = False) -> None:
    """Compile evaluation, optionally keeping variable pass counts in Python."""
    patch_inductor_skip_repro_deepcopy()
    if eager_pass_loop:
        block = model.model.decoder.block
        for name in ("inference_jacobi_pass", "inference_pass", "cyclic_pass"):
            if hasattr(block, name):
                setattr(block, name, torch.compile(getattr(block, name), dynamic=True, options=dict(_COMPILE_OPTIONS)))
        # Skip the scheduling frames while allowing their compiled passes to run.
        for name in ("forward", "forward_jacobi"):
            if hasattr(block, name):
                setattr(block, name, torch.compiler.disable(getattr(block, name), recursive=False))
    # Likelihood scoring calls the backbone directly to bound vocabulary memory;
    # generation calls the complete LM. Nested calls trace into their caller.
    for entry in (model, model.model):
        entry.compile(dynamic=True, options=dict(_COMPILE_OPTIONS))

"""Shared production forward setup and clipped-gradient calculation."""

import torch
import torch.distributed as dist

from training.compile import compile_training_forward, configure_training_compilation
from training.distributed import all_reduce_grads
from training.forward import TrainingForward
from training.losses import cce_linear_cross_entropy, checkpointed_linear_cross_entropy
from training.optim import clip_grad_norm_if_needed_, post_clip_norm_for_monitoring
from training.precision import assert_precision_contract, attention_kernel_context
from white_matter.modules.precision import cast_residual, model_autocast_context, residual_activation_dtype


def prepare_training_forward(model, recipe, batch_size, *, compiled=True):
    device = model.get_input_embeddings().weight.device
    chunk_size = 16 if recipe.model.execution_mode == "autoregressive" and not recipe.ar_cuda_graph else 0
    if compiled:
        configure_training_compilation(model, ar_dynamic=recipe.model.execution_mode == "autoregressive")
    runner = TrainingForward(
        model, checkpoint_chunk_size=chunk_size, external_ce=recipe.loss_backend == "cce" or chunk_size > 0
    ).to(device)
    if recipe.ar_cuda_graph:
        with attention_kernel_context(device):
            runner.capture_ar_graph(
                torch.zeros(
                    batch_size,
                    recipe.data.sequence_length,
                    model.config.hidden_size,
                    device=device,
                    dtype=residual_activation_dtype(device, recipe.model.residual_dtype),
                )
            )
    return compile_training_forward(runner) if compiled else runner


def training_gradients(model, training_forward, optimizers, batches, recipe, *, world=1, check_gradients=False):
    """Consume one update's microbatches, returning loss/norm and clipped gradients.

    The caller owns data loading, nonfinite-update policy, and optimizer stepping.
    """
    device = model.get_input_embeddings().weight.device
    residual_dtype = recipe.model.residual_dtype
    decoder_input_dtype = residual_activation_dtype(device, residual_dtype)
    params = optimizers.parameters
    grad_accum_steps = recipe.gradient_accumulation_steps
    max_grad_norm = recipe.optimizer.max_gradient_norm
    iterative = recipe.model.execution_mode in {"cyclic", "jacobi"}
    num_gradient_passes = recipe.gradient_passes if iterative else 1
    num_passes = num_gradient_passes + (recipe.no_gradient_passes if iterative else 0)
    external_ce_enabled = training_forward.external_ce
    ar_ce_token_chunk_size = 4096
    batches = iter(batches)
    # Local clipping precedes accumulation; averaging and the final clip follow.
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    accumulation_scale = 1.0 / grad_accum_steps
    accumulated_grads = None
    microbatch_norm = None
    for microbatch in range(grad_accum_steps):
        model.zero_grad(set_to_none=True)
        input_ids = next(batches)
        with attention_kernel_context(device), model_autocast_context(device):
            decoder_inputs = None
            if check_gradients and microbatch == 0:
                embeddings = model.model.embed_tokens(input_ids)
                decoder_inputs = cast_residual(embeddings, residual_dtype=residual_dtype)
                if embeddings.dtype != torch.float32:
                    raise RuntimeError(f"trainable fp32 embedding produced a non-fp32 activation: {embeddings.dtype}")
                assert_precision_contract(
                    model,
                    decoder_inputs,
                    residual_dtype=residual_dtype,
                )
                del embeddings
            # Norm + head + CE run in the compiled decoder runner unless
            # an external logit-free/bounded loss was selected.
            value = training_forward(
                decoder_inputs,
                num_passes,
                num_gradient_passes,
                token_ids=input_ids,
            )
            if recipe.loss_backend == "cce":
                loss = cce_linear_cross_entropy(value, input_ids, model.lm_head)
            elif external_ce_enabled:
                loss = checkpointed_linear_cross_entropy(
                    value, input_ids, model.lm_head, token_chunk_size=ar_ce_token_chunk_size
                )
            else:
                loss = value
            del value
            del decoder_inputs

        # Clip this microbatch before accumulation and data-parallel averaging.
        loss.backward()
        microbatch_norm = clip_grad_norm_if_needed_(params, max_norm=float(recipe.optimizer.max_gradient_norm))
        if grad_accum_steps > 1:
            if accumulated_grads is None:
                accumulated_grads = [p.grad.detach().clone() if p.grad is not None else None for p in params]
            else:
                for index, parameter in enumerate(params):
                    if parameter.grad is not None:
                        accumulated_grads[index] = (
                            parameter.grad.detach().clone()
                            if accumulated_grads[index] is None
                            else accumulated_grads[index].add_(parameter.grad)
                        )
        if check_gradients and microbatch == 0:
            assert_precision_contract(
                model,
                torch.empty(0, device=device, dtype=decoder_input_dtype),
                residual_dtype=residual_dtype,
                require_gradients=True,
            )
        loss_sum.add_(loss.detach().to(dtype=torch.float64))

    # Mean loss for logging; gradient averaging follows below.
    mean_loss = loss_sum * accumulation_scale
    # Overlap the loss average with gradient reduction; synchronize only at logging.
    pending_loss = dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG, async_op=True) if world > 1 else None
    if grad_accum_steps > 1:
        # Write the mean of the per-micro-batch-clipped grads into .grad.
        model.zero_grad(set_to_none=True)
        assert accumulated_grads is not None
        for index, parameter in enumerate(params):
            if accumulated_grads[index] is not None:
                parameter.grad = accumulated_grads[index].mul_(accumulation_scale)
    if world > 1:
        # One standard model parameter walk covers the decoder and its
        # embedding/norm/head modules.
        all_reduce_grads(
            model.parameters(),
            world,
            # A rank-divergent active set must fail before differently
            # sized packed-gradient all-reduces can hang.
            validate_presence=check_gradients,
        )
    if pending_loss is not None:
        pending_loss.wait()
    loss_value = float(mean_loss)
    # With one local micro-batch, reuse the identical earlier clip norm.
    reuse_microbatch_norm = grad_accum_steps == 1 and world == 1 and microbatch_norm is not None
    if reuse_microbatch_norm:
        grad_norm = post_clip_norm_for_monitoring(
            microbatch_norm,
            float(recipe.optimizer.max_gradient_norm),
        )
    else:
        grad_norm = clip_grad_norm_if_needed_(params, max_norm=max_grad_norm)
    return loss_value, grad_norm

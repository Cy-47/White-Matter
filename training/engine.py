"""Distributed next-token training with fixed recipe settings and exact resume."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from training.checkpoint import (
    gather_rng_states_for_rank0,
    load_training_checkpoint,
    restore_rng_states,
    save_training_checkpoint,
    validate_resume_training_identity,
    resolved_config_sha256,
    training_source_sha256,
)
from training.data import _new_loader_iterator, _resume_sequential_loader
from training.distributed import (
    broadcast_params,
    setup_distributed,
    initialize_all_reduce,
)
from training.step import prepare_training_forward, training_gradients
from training.metrics import MetricsLogger
from training.optim import (
    build_optimizers,
    set_learning_rate,
    step_optimizers,
)
from training.recipes import TrainingRecipe, per_rank_batch_size


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train(recipe: TrainingRecipe, args: argparse.Namespace) -> None:

    attention_implementation = "flash_attention_2"
    compile_mode = "default"
    checkpoint_chunk_size = 16 if recipe.model.execution_mode == "autoregressive" and not recipe.ar_cuda_graph else 0
    cache_dir = Path(args.data_dir).expanduser().resolve()
    cfg = {
        "recipe_sha256": recipe.sha256,
        "runtime": {
            "attention_implementation": attention_implementation,
            "compile_mode": compile_mode,
            "ar_checkpoint_chunk_size": checkpoint_chunk_size,
        },
        "cache_dir": str(cache_dir),
    }
    training_source_digest = training_source_sha256()
    metrics_provenance = {
        "resolved_config_sha256": resolved_config_sha256(cfg),
        "training_source_sha256": training_source_digest,
    }
    results_path = Path(args.output_dir)
    full_checkpoint_path = results_path / "ckpt_full.pt"
    resume_checkpoint = (
        Path(args.resume_from_checkpoint).expanduser().resolve()
        if args.resume_from_checkpoint is not None
        else full_checkpoint_path
    )
    if args.resume_from_checkpoint is not None and not resume_checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_checkpoint}")
    if results_path.exists() and not results_path.is_dir():
        raise ValueError(f"results path is not a directory: {results_path}")
    if args.resume_from_checkpoint is not None and full_checkpoint_path.exists():
        if resume_checkpoint != full_checkpoint_path.resolve():
            raise ValueError(
                "output directory already has ckpt_full.pt; remove the explicit "
                "--resume-from-checkpoint or choose a new output directory"
            )
    rank, world, local_rank, device, is_rank0 = setup_distributed()
    initialize_all_reduce(device)
    seed = recipe.seed
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    from transformers import AutoModelForCausalLM

    from training.data import TokenCacheDataset, load_cache_metadata
    from training.precision import (
        configure_precision,
        describe_precision_policy,
    )

    residual_dtype = recipe.model.residual_dtype
    # Splits — read canonical spec from cache metadata.
    splits = load_cache_metadata(cache_dir)["splits"]
    n_train = int(splits["n_train"])
    n_val = int(splits["n_val"])
    n_test = int(splits["n_test"])
    if is_rank0:
        log(f"distributed: rank={rank} world={world} local_rank={local_rank} device={device}")
        log(describe_precision_policy(device, residual_dtype))
        log(f"split meta: n_train={n_train} n_val={n_val} n_test={n_test}")

    iterative = recipe.model.execution_mode in {"cyclic", "jacobi"}
    num_gradient_passes = recipe.gradient_passes if iterative else 1
    no_gradient_passes = recipe.no_gradient_passes if iterative else 0
    num_passes = num_gradient_passes + no_gradient_passes
    if is_rank0:
        log(f"recipe={recipe.name} gradient_passes={num_gradient_passes} no_gradient_passes={no_gradient_passes}")

    # Build model.
    configure_precision(device)
    recipe.model._attn_implementation = attention_implementation
    model = AutoModelForCausalLM.from_config(recipe.model).to(device=device, dtype=torch.float32)
    model.train()

    # Ensure embeddings and final normalization match across ranks.
    # On resume, checkpoint loading below restores these weights as well.
    if world > 1:
        broadcast_params((model.get_input_embeddings().weight, model.model.norm.weight), src=0)
        if is_rank0:
            log("distributed: broadcast embeddings and final normalization from rank 0")

    bs = per_rank_batch_size(recipe, world)
    seq_len = recipe.data.sequence_length
    training_forward = prepare_training_forward(model, recipe, bs)

    # Average only after local microbatch clipping; DDP would change this update.
    if world > 1 and is_rank0:
        log(f"distributed: using bounded manual grad-reduce (no DDP wrapper). world={world}")

    grad_accum_steps = recipe.gradient_accumulation_steps

    resume_config = {**cfg, "batch_size": bs, "world_size": world}
    resume_config_sha256 = resolved_config_sha256(resume_config)

    # Sequential token data. Packed-document boundaries are derived from EOS
    # inside TrainingForward for every architecture.
    train_ds = TokenCacheDataset(
        cache_dir,
        split="train",
        sequence_length=seq_len,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )
    sampler = (
        DistributedSampler(
            train_ds,
            num_replicas=world,
            rank=rank,
            shuffle=False,
            drop_last=True,
        )
        if world > 1
        else None
    )
    loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=False,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = max(
        1,
        (len(sampler) if sampler is not None else len(train_ds)) // (bs * grad_accum_steps),
    )
    if is_rank0:
        log(
            f"batch: per_rank={bs} global={bs * world} grad_accum={grad_accum_steps} "
            f"effective={bs * world * grad_accum_steps} steps_per_epoch={steps_per_epoch} "
            "data_order=sequential"
        )

    # Muon owns hidden 2D matrices; AdamW owns norms, biases, embeddings, and
    # the tied LM head. This is the paper's only optimizer partition.
    lr_base = recipe.optimizer.learning_rate
    optimizers = build_optimizers(
        model,
        base_lr=lr_base,
        weight_decay=recipe.optimizer.weight_decay,
        adam_beta1=recipe.optimizer.adam_beta1,
        adam_beta2=recipe.optimizer.adam_beta2,
        muon_momentum=recipe.optimizer.muon_momentum,
        muon_ns_steps=recipe.optimizer.muon_ns_steps,
        device=device,
        distributed_muon=recipe.optimizer.distributed_muon,
    )
    if is_rank0:
        for optimizer in (optimizers.adamw, optimizers.muon):
            if optimizer is not None:
                for group in optimizer.param_groups:
                    count = sum(p.numel() for p in group["params"])
                    log(f"optimizer group {group['name']}: {count:,} parameters")
    opt = optimizers.adamw
    opt_muon = optimizers.muon
    # The shared factory validates an exhaustive, duplicate-free partition.
    # This static order is also used by clipping and manual gradient reduction.

    max_grad_norm = recipe.optimizer.max_gradient_norm
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be positive and finite, got {max_grad_norm}")
    if is_rank0:
        log(f"global gradient clip={max_grad_norm:g}")
    warmup_frac = recipe.optimizer.warmup_fraction
    lr_schedule = "cosine"
    if is_rank0:
        log(
            f"lr schedule: {lr_schedule} peak={lr_base:.3e} "
            f"warmup_frac={warmup_frac:g} floor_frac={float(recipe.optimizer.minimum_lr_fraction):g}"
        )

    if is_rank0:
        results_path.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    metrics = MetricsLogger(
        results_path / "metrics.jsonl",
        enabled=is_rank0,
        constant_fields=metrics_provenance,
    )

    def save_model_for_eval(path: Path, *, step: int) -> None:
        """Save a standard Hugging Face directory with safetensors weights."""
        model.config.training_step = int(step)
        model.config.training_sequence_length = recipe.data.sequence_length
        model.config.recipe_name = str(recipe.name)
        model.config.recipe_sha256 = resume_config_sha256
        model.save_pretrained(path, safe_serialization=True)

    n_skipped = 0  # steps whose grad was non-finite → opt.step() skipped
    start_step = 0
    if resume_checkpoint.exists():
        checkpoint = load_training_checkpoint(resume_checkpoint)
        validate_resume_training_identity(
            checkpoint,
            current_config_sha256=resume_config_sha256,
            current_source_sha256=training_source_digest,
        )
        checkpoint_step = checkpoint.get("step")
        if type(checkpoint_step) is not int or not 0 <= checkpoint_step <= recipe.steps:
            raise ValueError(f"checkpoint step must be in [0, {recipe.steps}], got {checkpoint_step!r}")
        checkpoint_adamw_state = checkpoint.get("opt")
        if not isinstance(checkpoint_adamw_state, dict):
            raise ValueError("resume checkpoint is missing AdamW state")
        checkpoint_muon_state = checkpoint.get("opt_muon")
        if opt_muon is not None:
            if not isinstance(checkpoint_muon_state, dict):
                raise ValueError("Muon resume requires checkpoint opt_muon state")
        elif checkpoint_muon_state is not None:
            raise ValueError("checkpoint contains Muon state but the current optimizer does not")
        model.load_state_dict(checkpoint["model"], strict=True)
        opt.load_state_dict(checkpoint_adamw_state)
        del checkpoint_adamw_state
        if opt_muon is not None:
            assert isinstance(checkpoint_muon_state, dict)
            opt_muon.load_state_dict(checkpoint_muon_state)
        restore_rng_states(
            checkpoint,
            rng,
            rank=rank,
            world_size=world,
            cuda_device=local_rank,
        )
        n_skipped = int(checkpoint.get("n_skipped", 0))
        metrics.restore_rolling(checkpoint.get("rolling", {}))
        start_step = checkpoint_step
        # Truncate metrics.jsonl to the resumed trajectory (drop rows for
        # steps >= start_step that were written after the last checkpoint).
        metrics.truncate_to(start_step)
        if is_rank0:
            log(
                f"RESUME: ckpt_full.pt → continuing from step {start_step}/{recipe.steps} (model+AdamW{'+Muon' if opt_muon is not None else ''}+rng restored; metrics truncated to <{start_step}; n_skipped={n_skipped})"
            )
        # All model/optimizer/RNG state has been copied into its live owner. Drop
        # the multi-GB CPU payload on every rank before training resumes.
        del checkpoint
    last_full_checkpoint_step = start_step if resume_checkpoint.exists() else None
    if world > 1:
        dist.barrier()
    if is_rank0:
        log(
            f"begin training: {recipe.steps} steps  bs={bs} global_bs={bs * world} seq_len={seq_len}"
            + (f"  [RESUMED @ {start_step}]" if start_step else "")
        )

    t0 = time.monotonic()
    # Resume by batches, including partial epochs under gradient accumulation.
    iterator = None
    epoch = 0
    if start_step < recipe.steps:
        total_batches = start_step * grad_accum_steps
        iterator, epoch, skip_batches, batches_per_epoch = _resume_sequential_loader(
            loader,
            total_batches=total_batches,
            sampler=sampler,
        )
        if skip_batches > 0 and is_rank0:
            log(
                f"RESUME: fast-forwarded sequential loader by {skip_batches} batches "
                f"into epoch {epoch} ({skip_batches}/{batches_per_epoch})"
            )

    def _save_training_checkpoint(
        next_step: int,
        *,
        write_latest_model: bool,
    ) -> None:
        """Save one exact-resume checkpoint through the all-rank protocol."""
        adamw_state = opt.state_dict() if is_rank0 else None
        if opt_muon is not None and recipe.optimizer.distributed_muon:
            opt_muon.consolidate_state_dict(to=0)
        muon_state = opt_muon.state_dict() if opt_muon is not None and is_rank0 else None
        rng_states = gather_rng_states_for_rank0(
            rng,
            rank=rank,
            world_size=world,
            cuda_device=local_rank,
        )

        if is_rank0:
            assert adamw_state is not None
            assert rng_states is not None
            if write_latest_model:
                save_model_for_eval(
                    results_path / "latest",
                    step=next_step,
                )

            # ``next_step`` is the next update to run after a resume.
            save_training_checkpoint(
                results_path / "ckpt_full.pt",
                step=next_step,
                model=model,
                opt_state=adamw_state,
                rng=rng,
                n_skipped=n_skipped,
                rolling=metrics.rolling_state(),
                cfg_name=recipe.name,
                seed=seed,
                training_config_sha256=resume_config_sha256,
                training_source_sha256=training_source_digest,
                rng_states_by_rank=rng_states,
                opt_muon_state=muon_state,
                extra={"residual_dtype": residual_dtype},
            )
        if world > 1:
            dist.barrier()

    completed_steps = start_step
    gradients_checked = False
    for step in range(start_step, recipe.steps):
        completed_steps = step + 1
        # Recipe pass counts are fixed across ranks and micro-batches.
        check_gradients = not gradients_checked and num_gradient_passes > 0

        cur_lr = set_learning_rate(
            optimizers,
            step=step,
            total_steps=recipe.steps,
            schedule=lr_schedule,
            warmup_frac=warmup_frac,
            floor_frac=recipe.optimizer.minimum_lr_fraction,
        )

        def microbatches():
            nonlocal iterator, epoch
            for _ in range(grad_accum_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    iterator = _new_loader_iterator(loader)
                    batch = next(iterator)
                yield batch["input_ids"].to(device=device, non_blocking=True)

        loss_value, grad_norm = training_gradients(
            model, training_forward, optimizers, microbatches(), recipe, world=world, check_gradients=check_gradients,
        )
        if check_gradients:
            gradients_checked = True
            if is_rank0:
                log("precision/gradient audit PASS")
        # Never apply an optimizer update to non-finite gradients.
        gn = float(grad_norm)
        if not torch.isfinite(grad_norm):
            n_skipped += 1
            if is_rank0:
                log(f"step {step + 1}: non-finite grad_norm ({gn}); update skipped")
            metrics.log_raw({"step": step, "skipped": True, "grad_norm": str(gn)})
            model.zero_grad(set_to_none=True)
            continue
        step_optimizers(optimizers)

        # Cumulative rank-zero peaks include initialization and optimizer-state allocation.
        if is_rank0 and str(device).startswith("cuda"):
            rank0_cuda_max_allocated_mib = torch.cuda.max_memory_allocated(device) / 2**20
            rank0_cuda_max_reserved_mib = torch.cuda.max_memory_reserved(device) / 2**20
        else:
            rank0_cuda_max_allocated_mib = 0.0
            rank0_cuda_max_reserved_mib = 0.0

        row = metrics.log_step(
            {
                "step": step,
                "wall_s": time.monotonic() - t0,
                "lr": cur_lr,
                "grad_norm": float(grad_norm),
                "num_passes": num_passes,
                "n_grad": num_gradient_passes,
                "n_no_grad": no_gradient_passes,
                "loss": loss_value,
                "rank0_cuda_max_allocated_mib": rank0_cuda_max_allocated_mib,
                "rank0_cuda_max_reserved_mib": rank0_cuda_max_reserved_mib,
            }
        )

        if is_rank0 and (step == 0 or (step + 1) % args.log_every == 0):
            log(
                f"step {step + 1:>5d}/{recipe.steps}: "
                f"n={num_passes:>3d} (g={num_gradient_passes} ng={no_gradient_passes})  "
                f"lm={row['loss_rolling']:.4f} "
                f"lr={cur_lr:.2e} grad={float(grad_norm):.2f} "
                f"wall={row['wall_s']:.0f}s"
            )

        # Save after the update and metrics so resume starts at the next batch.
        save_this_step = bool(args.save_every and (step + 1) % args.save_every == 0)
        if save_this_step:
            _save_training_checkpoint(
                step + 1,
                write_latest_model=True,
            )
            last_full_checkpoint_step = step + 1

    if args.save_every and completed_steps == recipe.steps and last_full_checkpoint_step != completed_steps:
        # The periodic cadence need not divide the training endpoint. Preserve
        # optimizer/RNG state at the true endpoint so relaunching a completed
        # run never silently repeats the trailing updates.
        _save_training_checkpoint(
            completed_steps,
            write_latest_model=False,
        )

    metrics.close()
    if is_rank0:
        final_model_ckpt = results_path / "final"
        save_model_for_eval(final_model_ckpt, step=completed_steps)
        log(
            f"training done in {time.monotonic() - t0:.1f}s  →  "
            f"{final_model_ckpt} @ step {completed_steps}  "
            f"(non-finite-grad steps skipped: {n_skipped}/{recipe.steps})"
        )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()

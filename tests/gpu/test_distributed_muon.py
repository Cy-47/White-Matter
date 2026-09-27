"""Sharded Muon preserves actual training updates and consolidated resume state."""

import copy
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_training_optimizations import assert_training_close, make_model, make_optimizers, training_step

from training.compile import compile_feedback
from training.distributed import all_reduce_grads
from training.forward import TrainingForward
from training.optim import clip_grad_norm_if_needed_, step_optimizers
from training.precision import configure_precision

pytestmark = pytest.mark.gpu


def _worker(rank, rendezvous):
    from white_matter.models import register_models

    register_models()
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=f"file://{rendezvous}")
    try:
        configure_precision("cuda")
        torch.manual_seed(135)
        reference = make_model(residual_dtype="bf16")
        candidate = copy.deepcopy(reference)
        runners = []
        for model in (reference, candidate):
            compile_feedback(model, mode="default")
            runners.append(torch.compile(TrainingForward(model, external_ce=True), fullgraph=False, dynamic=False))
        reference_opt = make_optimizers(reference, compiled=False)
        candidate_opt = make_optimizers(candidate, distributed=True)
        assert all("compiled" not in group for group in candidate_opt.muon.param_groups)
        # All ranks use different packed documents, then the same clipping/reduction order as training.
        generator = torch.Generator(device="cuda").manual_seed(141 + rank)
        for step in range(3):
            ids = torch.randint(0, 256, (2, 128), device="cuda", generator=generator)
            ids[0, [5 + step + rank, 29, 82]] = 256
            ids[1, [3, 56 + step, 101]] = 256
            expected = training_step(reference, runners[0], ids, cce=True)
            actual = training_step(candidate, runners[1], ids, cce=True)
            assert_training_close(actual, expected)
            # CCE accumulates in a nondeterministic order. Compare optimizers on
            # identical gradients after independently checking both backwards.
            for a, b in zip(candidate.parameters(), reference.parameters(), strict=True):
                a.grad.copy_(b.grad)
            for model, optimizer in ((reference, reference_opt), (candidate, candidate_opt)):
                clip_grad_norm_if_needed_(optimizer.parameters, 1.0)
                all_reduce_grads(model.parameters(), 2, validate_presence=step == 0)
                clip_grad_norm_if_needed_(optimizer.parameters, 1.0)
                step_optimizers(optimizer)
            for (name, a), (_, b) in zip(candidate.named_parameters(), reference.named_parameters(), strict=True):
                torch.testing.assert_close(
                    a, b, rtol=1e-3, atol=3e-6, msg=lambda message, name=name: f"{name}: {message}"
                )
                peer = a.detach().clone()
                dist.broadcast(peer, src=0)
                torch.testing.assert_close(a, peer, rtol=0, atol=0, msg=name)
            if step == 0:
                # Exercise the all-rank consolidation and native load used by the engine.
                candidate_opt.muon.consolidate_state_dict(to=0)
                state = [copy.deepcopy(candidate_opt.muon.state_dict()) if rank == 0 else None]
                dist.broadcast_object_list(state, src=0)
                restarted = make_optimizers(candidate, distributed=True)
                restarted.muon.load_state_dict(state[0])
                restarted.adamw.load_state_dict(copy.deepcopy(candidate_opt.adamw.state_dict()))
                candidate_opt = restarted
        local_state = sum(
            t.numel()
            for state in candidate_opt.muon.optim.state.values()
            for t in state.values()
            if isinstance(t, torch.Tensor)
        )
        full_state = sum(
            t.numel()
            for state in reference_opt.muon.state.values()
            for t in state.values()
            if isinstance(t, torch.Tensor)
        )
        assert 0 < local_state < full_state
        print(f"rank={rank} Muon state elements={local_state}/{full_state}", flush=True)
    finally:
        torch.compiler.reset()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_distributed_muon_training_and_resume():
    pytest.importorskip("cut_cross_entropy")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_worker, args=(str(Path(directory) / "nccl"),), nprocs=2, join=True)

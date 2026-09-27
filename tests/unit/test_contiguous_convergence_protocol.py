"""Independent visibility and fixed-point checks for the study-local schedule."""

import pytest
import torch

from evals.execution import execution
from studies.prefill_convergence.contiguous import forward_contiguous
from tests.unit.test_experiment_protocols import tiny_model


@torch.inference_mode()
def reference(block, x, passes, chunks):
    # Keep layer-input state, and rebuild the entire pool before every chunk.
    # This intentionally does not share the candidate's in-place KV publication.
    qrope, krope = block._prepare_rope(x)
    states = x.unsqueeze(2).expand(-1, -1, len(block.layers), -1).clone()
    outputs = torch.empty_like(x)
    length = x.shape[1]
    for _ in range(passes):
        for c in range(chunks):
            start, end = c * length // chunks, (c + 1) * length // chunks
            key, value = block.kv_pool.project_sequence(states, krope, dummy_token=block.dummy_token)
            mask = (torch.arange(length + 1)[None, :] <= torch.arange(start, end)[:, None]).to(x.device)
            hidden = x[:, start:end]
            fresh = []
            for layer_index, layer in enumerate(block.layers):
                fresh.append(hidden)
                channel = layer_index % block.num_kv_channels
                hidden = layer(
                    hidden,
                    key[:, channel],
                    value[:, channel],
                    tuple(t[:, start:end] for t in qrope),
                    decode_key_mask=mask[None, None],
                )
            states[:, start:end] = torch.stack(fresh, dim=2)
            outputs[:, start:end] = hidden
    return outputs, block.kv_pool.project_sequence(states, krope, dummy_token=block.dummy_token)


@pytest.mark.parametrize("chunks", [1, 2, 4, 7])
@torch.inference_mode()
def test_contiguous_matches_independent_state_updates(chunks):
    torch.manual_seed(9)
    model = tiny_model()
    block = model.model.decoder.block
    x = torch.randn(2, 7, 16)
    expected, state = reference(block, x, 3, chunks)
    observations = []
    actual, actual_state = forward_contiguous(
        block,
        x,
        num_passes=3,
        chunks=chunks,
        on_pass=lambda p, h: observations.append(h.clone()),
        output_final_state=True,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state, state)
    for p, hidden in enumerate(observations, 1):
        torch.testing.assert_close(hidden, reference(block, x, p, chunks)[0])
    torch.testing.assert_close(forward_contiguous(block, x, num_passes=3, chunks=chunks), actual)


@torch.inference_mode()
def test_endpoints_and_future_independence():
    torch.manual_seed(10)
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    x = model.model.embed_tokens(ids)
    block = model.model.decoder.block
    with execution(model, mode="autoregressive", precision="fp32"):
        jacobi = block.forward_jacobi(x, num_passes=3, num_gradient_passes=0)
        torch.testing.assert_close(forward_contiguous(block, x, num_passes=3, chunks=1), jacobi)
        ar = model.model(ids, use_cache=False).last_hidden_state
        actual = model.model.norm(forward_contiguous(block, x, num_passes=1, chunks=7))
        torch.testing.assert_close(actual, ar)
        before = forward_contiguous(block, x, num_passes=3, chunks=2)
        changed = x.clone()
        changed[:, 3:] += 100
        after = forward_contiguous(block, changed, num_passes=3, chunks=2)
        torch.testing.assert_close(before[:, :3], after[:, :3])


def test_pool_rejects_different_contiguous_semantics():
    from studies.prefill_convergence.analyze import pool
    from tests.unit.test_experiment_protocols import quality_shard

    shards = [quality_shard(0, 96, 3.1), quality_shard(96, 96, 3.1)]
    shards[1]["schedules"]["contiguous16"]["update"] = "after_pass"
    with pytest.raises(ValueError, match="different schedules"):
        pool(shards)


def test_convergence_plot_includes_curves(tmp_path):
    import json

    from scripts.plot_experiments import plot

    pytest.importorskip("matplotlib")
    data = {
        "ar_perplexity": 10,
        "tolerance": 0.01,
        "modes": {"contiguous16": {"ce": [3, 2.31, 2.30]}},
        "timing": {"rows": {"contiguous16": {"seconds_per_sequence": 0.1}}},
    }
    source, output = tmp_path / "result.json", tmp_path / "figure.pdf"
    source.write_text(json.dumps(data))
    plot("convergence", source, output)
    assert output.read_bytes().startswith(b"%PDF-")


def test_timing_shards_merge_disjoint_cases_and_reject_mismatches():
    import copy

    from studies.prefill_convergence.analyze import merge_timings
    from studies.prefill_convergence.protocol import schedules

    base = {
        "protocol": "prefill_convergence",
        "checkpoint": {},
        "windows_sha256": "same",
        "sequence_length": 2048,
        "batch_size": 64,
        "precision": "bf16",
        "compiled": True,
        "scope": "decoder",
        "warmups": 5,
        "repetitions": 30,
        "ar_repetitions": 10,
        "contiguous_backend": "flash_attention_2",
        "source": {"sha256": "same"},
        "environment": {"gpu": "A6000", "torch": "same", "cuda": "same", "packages": {}},
    }
    shards = [
        {**copy.deepcopy(base), "rows": {mode: {"passes": 2}}, "schedules": schedules([mode])}
        for mode in ("cyclic64", "contiguous64")
    ]
    assert set(merge_timings(shards)["rows"]) == {"cyclic64", "contiguous64"}
    with pytest.raises(ValueError, match="overlapping"):
        merge_timings([shards[0], shards[0]])
    wrong = copy.deepcopy(shards)
    wrong[1]["environment"]["gpu"] = "different"
    with pytest.raises(ValueError, match="hardware"):
        merge_timings(wrong)


def test_full_sweep_plot(tmp_path):
    import json

    from scripts.plot_experiments import plot
    from studies.prefill_convergence.protocol import GROUPS, MODES

    pytest.importorskip("matplotlib")
    assert GROUPS == (2, 4, 8, 16, 32, 64)
    data = {
        "ar_perplexity": 10,
        "tolerance": 0.01,
        "modes": {mode: {"ce": [3, 2.31, 2.30], "passes": 3} for mode in MODES},
        "timing": {"rows": {mode: {"seconds_per_sequence": 0.1} for mode in ["ar", *MODES]}},
    }
    source, output = tmp_path / "sweep.json", tmp_path / "sweep.pdf"
    source.write_text(json.dumps(data))
    plot("convergence-sweep", source, output)
    assert output.read_bytes().startswith(b"%PDF-")


@pytest.mark.parametrize("batch_size", [32, 64, 0, 193, True])
def test_timing_join_validates_batch_size(batch_size):
    from studies.prefill_convergence.analyze import join_timings, pool
    from tests.unit.test_experiment_protocols import quality_shard

    quality = pool([quality_shard(0, 192, 3.1)])
    timing = {key: quality[key] for key in ("checkpoint", "windows_sha256", "sequence_length", "schedules")}
    timing.update(
        batch_size=batch_size,
        precision="bf16",
        rows={"ar": {"passes": None}, **{mode: {"passes": row["passes"]} for mode, row in quality["modes"].items()}},
    )
    if batch_size in (32, 64):
        assert join_timings(quality, timing)["timing"]["batch_size"] == batch_size
    else:
        with pytest.raises(ValueError, match="workload"):
            join_timings(quality, timing)


@pytest.mark.parametrize("unreached", [("contiguous2",), ("jacobi", "cyclic2", "contiguous2")])
def test_timing_join_preserves_unreached_modes(unreached):
    from studies.prefill_convergence.analyze import join_timings, pool
    from studies.prefill_convergence.protocol import schedules
    from tests.unit.test_experiment_protocols import quality_shard

    shard = quality_shard(0, 192, 3.1)
    modes = ("jacobi", "cyclic2", "contiguous2")
    shard["curves"] = {m: [3.1 * shard["targets"]] if m in unreached else shard["curves"][m] for m in modes}
    shard["schedules"] = schedules(modes)
    quality = pool([shard])
    reached = set(modes) - set(unreached)
    timing = {key: quality[key] for key in ("checkpoint", "windows_sha256", "sequence_length")}
    timing.update(
        batch_size=32,
        precision="bf16",
        schedules=schedules(reached),
        rows={"ar": {"passes": None}, **{m: {"passes": quality["modes"][m]["passes"]} for m in reached}},
    )
    joined = join_timings(quality, timing)
    assert joined["modes"] == quality["modes"]
    assert joined["schedules"] == quality["schedules"]
    assert all(joined["modes"][m]["status"] == "not_reached" for m in unreached)
    assert joined["timing"] == timing
    if reached:
        timing["schedules"][next(iter(reached))]["update"] = "invalid"
        with pytest.raises(ValueError, match="schedule mismatch"):
            join_timings(quality, timing)

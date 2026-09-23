"""Capacity search must account for the entire worker's memory and record failures."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from benchmarks import generation, _runner


def test_benchmark_selects_each_feedback_prefill_policy():
    from transformers import AutoConfig

    lckv = AutoConfig.for_model('lckv', num_hidden_layers=4, prefill_mode='autoregressive')
    white_matter = AutoConfig.for_model('white_matter', num_hidden_layers=4, num_kv_channels=2)
    generation.configure_prefill_mode(lckv)
    generation.configure_prefill_mode(white_matter)
    assert lckv.prefill_mode == 'jacobi'
    assert white_matter.prefill_mode == 'cyclic'


@pytest.mark.parametrize("phase", ["prefill", "decode", "end-to-end"])
def test_matched_memory_search_records_oom_and_checks_boundary(tmp_path, monkeypatch, phase):
    models = []
    for name in ('wm', 'vanilla'):
        model = tmp_path / name
        model.mkdir()
        (model / 'config.json').write_text('{}')
        (model / 'model.safetensors').write_bytes(b'test checkpoint identity')
        models.append(str(model))
    output = tmp_path / 'outputs'
    monkeypatch.setattr(sys, 'argv', ['generation.py', '--models', *models, '--output', str(output),
                                     '--batch-sizes', '1', '--prompt-lengths', '128', '--memory-budget-gib', '10',
                                     '--max-batch-size', '8', '--phase', phase, '--num-splits', '2', '--worker-repeats', '1'])
    calls = []

    def worker(command, **kwargs):
        path = Path(command[command.index('--worker') + 1])
        case = json.loads(path.read_text())
        payload = case['workload']
        batch, model = payload['batch_size'], payload['model']
        assert payload['phase'] == phase
        assert payload['num_splits'] == 2
        assert payload['prefill_batch_size'] == 1
        calls.append((model, batch))
        # Different phase footprints must produce independent capacity boundaries.
        base = (6 if phase == 'decode' else 8) if Path(model).name == 'wm' else 7
        memory = base + batch / 2
        row = dict(status='cuda_oom') if batch == 8 else dict(
            status='ok', memory={'budget_accounted_bytes': memory * 2**30}, total={'tokens_per_second': batch * 100}, decode=None if phase == 'prefill' else {'tokens_per_second': batch * 100},
            prefill=None if phase == 'decode' else {'tokens_per_second': batch * 200})
        case.update(status=row.pop('status'), result=row)
        path.write_text(json.dumps(case))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(_runner.subprocess, 'run', worker)
    generation.main()
    report = json.loads(next(output.glob('*-generation/summary.json')).read_text())
    assert len(calls) == len(set(calls)), 'matched-batch measurements should be reused by capacity search'
    maxima = [7, 5] if phase == 'decode' else [3, 5]
    capacity = {row['model']: row for row in report['capacity']}
    assert [capacity[model]['maximum_feasible_batch'] for model in models] == maxima
    assert [capacity[model]['best_measured_throughput_batch'] for model in models] == maxima
    for model, maximum in zip(models, maxima, strict=True):
        assert (model, maximum) in calls and (model, maximum + 1) in calls
    assert any(row['status'] == 'cuda_oom' for row in report['measurements'])
    assert not any(row['search_capped'] for row in report['capacity'])


def test_timing_summary_counts_tokens_and_single_output():
    assert generation.summarize([2.0, 4.0], 12)['tokens_per_second'] == 4
    assert generation.summarize([2.0, 4.0], 12)['iqr_seconds'] == 1
    assert generation.summarize([1.0], 0)['tokens_per_second'] is None
    assert generation.summarize([1.0], 0)['iqr_seconds'] == 0


def test_worker_failure_is_persisted_before_raising(tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'model.safetensors').write_bytes(b'test')
    output = tmp_path / 'outputs'
    monkeypatch.setattr(sys, 'argv', ['generation.py', '--models', str(tmp_path), '--output', str(output),
                                     '--batch-sizes', '1', '--prompt-lengths', '128'])
    monkeypatch.setattr(_runner.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=9))
    with pytest.raises(RuntimeError, match='worker failed'):
        generation.main()
    run = next(output.glob('*-generation'))
    result = json.loads(next((run / 'cases').glob('*.json')).read_text())
    assert result['status'] == 'failed' and result['returncode'] == 9
    assert next((run / 'cases').glob('*.log')).is_file()
    assert not json.loads((run / 'summary.json').read_text())['complete']


@pytest.mark.parametrize('family', ['white_matter', 'vanilla'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=[
    pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')])])
def test_decode_preparation_copies_real_prefix_into_independent_rows(family, device):
    import copy
    from transformers import AutoConfig, AutoModelForCausalLM
    from white_matter.models.generation import DecodeGraph

    torch.manual_seed(17)
    config = AutoConfig.for_model(
        family, vocab_size=127, hidden_size=192, intermediate_size=384,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=96,
        num_kv_channels=2, num_passes=3, cyclic_groups=4, prefill_mode='cyclic',
        eos_token_id=126, document_separator_token_id=None,
    )
    config._attn_implementation = 'flash_attention_2' if device == 'cuda' else 'sdpa'
    model = AutoModelForCausalLM.from_config(config).to(device).eval()
    prompt = torch.randint(1, 125, (1, 32), device=device)
    with torch.no_grad():
        reference = model.allocate_inference_cache(32)
        logits = model(prompt, past_key_values=reference, use_cache=True, logits_to_keep=1).logits
        cache, first = generation.prepare_decode(model, prompt, 3, 40)
        assert cache.get_seq_length() == 32 and cache.position.shape == (3, 1)
        torch.testing.assert_close(first, logits.argmax(-1).expand(3, -1), rtol=0, atol=0)
        for actual, expected, extra in zip(cache.layers, reference.layers, cache.prefix_slots, strict=True):
            assert actual.max_batch_size == 3 and actual.keys.stride(0) > 0
            for a, b in zip((actual.keys, actual.values), (expected.keys, expected.values), strict=True):
                torch.testing.assert_close(a[:, :, :32 + extra], b.expand(3, -1, -1, -1), rtol=0, atol=0)
        other = copy.deepcopy(cache)
        tokens = torch.tensor([[3], [4], [5]], device=device)
        expected = model(tokens, past_key_values=other, use_cache=True, logits_to_keep=1).logits
        actual = (DecodeGraph(model, cache)(tokens) if device == 'cuda' else
                  model(tokens, past_key_values=cache, use_cache=True, logits_to_keep=1).logits)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert cache.get_seq_length() == 33


@pytest.mark.parametrize("wrapped", [False, True])
def test_oom_cannot_establish_capacity_when_source_changes(tmp_path, monkeypatch, wrapped):
    path = tmp_path / 'cases' / 'case.json'
    path.parent.mkdir()
    path.write_text(json.dumps(dict(case_id='case', status='pending', workload={'model': 'model', 'profile': None})))
    (tmp_path / 'run.json').write_text(json.dumps(dict(source={}, checkpoints={'model': {}})))
    calls = 0

    def verify(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('source changed')

    def oom(_):
        error = torch.OutOfMemoryError('capacity')
        if wrapped:
            from torch._dynamo.exc import BackendCompilerFailed
            raise BackendCompilerFailed(lambda: None, error, None)
        raise error

    monkeypatch.setattr(_runner, 'verify_sources', verify)
    monkeypatch.setattr(_runner, 'input_files', lambda _: {})
    monkeypatch.setattr(generation, 'benchmark', oom)
    with pytest.raises(SystemExit):
        _runner.run_worker(path, generation.benchmark)
    case = json.loads(path.read_text())
    assert calls == 2 and case['status'] == 'failed' and 'source changed' in case['error']

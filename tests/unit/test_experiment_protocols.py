"""Regression checks for the main-body analysis and shared inference paths."""

import json

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM

from evals.execution import execution
from studies.prefill_convergence.evaluate import measure_curves
from studies.prefill_convergence.protocol import first_crossing
from studies.prefill_convergence.prepare_data import select_disjoint_pairs
from studies.schedules import analyze
from white_matter.models.white_matter import WhiteMatterConfig


def tiny_model(device='cpu'):
    config = WhiteMatterConfig(
        vocab_size=31, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, num_kv_channels=2,
        num_passes=2, cyclic_groups=2, router_layer_stride=1,
        eos_token_id=30, document_separator_token_id=None,
    )
    return AutoModelForCausalLM.from_config(config).to(device).eval()


def test_seed_curves_are_averaged_before_metric_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, 'NO_GRAD', (1,))
    monkeypatch.setattr(analyze, 'GRAD', (1,))
    monkeypatch.setattr(analyze, 'MODES', ('tp',))
    for seed, start in [(1337, [10, 20]), (1338, [20, 10])]:
        directory = tmp_path/f'seed{seed}'/'ng1_g1_tp'
        directory.mkdir(parents=True)
        for mode in ('native', 'cyclic16'):
            (directory/f'eval_{mode}.json').write_text(json.dumps(dict(
                protocol='figure7a_sequential_final', seed=seed, arm='ng1_g1_tp',
                evaluation_mode=mode, n_tok=analyze.PAPER_TEST_TARGETS,
                rows=[dict(n_passes=i, perplexity=v) for i, v in enumerate(start+[30]*30, 1)],
            )))
    seeds, averaged = analyze.collect(tmp_path)
    assert [s['native_best_perplexity'] for s in seeds] == [10, 10]
    assert averaged[0]['native_best_perplexity'] == 15
    assert averaged[0]['native_best_pass'] == 1
    assert averaged[0]['jacobi_passes_within_1pct'] == 1


def test_threshold_and_disjoint_selection():
    assert first_crossing([3.1, 3.005, 3.0], 3.0) == 2
    assert first_crossing([3.1, 3.02], 3.0) is None
    with pytest.raises(ValueError, match='nonfinite'):
        first_crossing([float('nan')], 3.0)
    assert select_disjoint_pairs(np.array([True, True, True, False, True, True]), 2) == [(0, 1), (4, 5)]
    with pytest.raises(ValueError, match='found'):
        select_disjoint_pairs(np.ones(3, dtype=bool), 2)


@torch.inference_mode()
def test_observed_passes_match_independent_forward_and_exact_ar():
    torch.manual_seed(51)
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    curves = measure_curves(model, ids, limits={'jacobi': 6, 'cyclic2': 3}, batch_size=1)
    from evals.execution import evaluate
    loader = [{'input_ids': ids}]
    for mode, groups, name in [('jacobi', None, 'jacobi'), ('cyclic', 2, 'cyclic2')]:
        for count, total in enumerate(curves['curves'][name], 1):
            with execution(model, mode=mode, groups=groups, precision='fp32'):
                expected, targets = evaluate(model, loader, num_passes=count)
            assert total == pytest.approx(expected, abs=1e-5)
            assert targets == curves['targets']
    assert curves['curves']['jacobi'][-1] == pytest.approx(curves['ar_ce_sum'], abs=1e-5)
    assert curves['curves']['cyclic2'][-1] == pytest.approx(curves['ar_ce_sum'], abs=1e-5)


@torch.inference_mode()
def test_execution_restores_settings_after_failure():
    model = tiny_model()
    original = model.config.to_dict()
    with pytest.raises(RuntimeError, match='test'):
        with execution(model, mode='jacobi', passes=4, precision='fp32'):
            raise RuntimeError('test')
    assert model.config.to_dict() == original
    for module in model.modules():
        assert not hasattr(module, '_force_jacobi_reference')


def quality_shard(offset, count, loss):
    from studies.prefill_convergence.protocol import MODES, schedules
    tokens = count*2047
    return dict(protocol='prefill_convergence', precision='fp32', checkpoint={'weights': 'same'},
                windows_sha256='same', sequence_length=2048, offset=offset, count=count,
                targets=tokens, ar_ce_sum=3*tokens,
                schedules=schedules(MODES), curves={mode: [loss*tokens, 3.0*tokens] for mode in MODES})


def test_shards_pool_losses_before_threshold_and_reject_overlap():
    from studies.prefill_convergence.analyze import pool
    shards = [quality_shard(0, 96, 3.04), quality_shard(96, 96, 2.99)]
    assert pool(shards)['modes']['jacobi']['passes'] == 2
    with pytest.raises(ValueError, match='overlapping'):
        pool([shards[0], shards[0]])
    with pytest.raises(ValueError, match='all 192'):
        pool(shards[:1])
    shards[1]['checkpoint'] = {'weights': 'different'}
    with pytest.raises(ValueError, match='different identities'):
        pool(shards)


def test_complete_downstream_selection_and_unrounded_average():
    from evals.paper import METRICS, downstream_metrics
    payload = dict(limit=None, num_fewshot=0, max_length=1024,
                   checkpoint={'weights': 'same'}, model_config={'num_passes': 3},
                   harness={'git_commit': 'pinned'}, task_versions={'squad_completion': 1},
                   n_samples={}, results={})
    for i, (task, metric, count) in enumerate(METRICS):
        payload['results'].setdefault(task, {})[f'{metric},none'] = i/20
        payload['n_samples'][task] = {'effective': count}
    scores, _ = downstream_metrics([payload])
    assert scores['average'] == pytest.approx(sum(i/20 for i in range(2, 13))/11)
    payload['limit'] = 2
    with pytest.raises(ValueError, match='complete'):
        downstream_metrics([payload])


def test_causal_pair_formula_matches_explicit_masks():
    from benchmarks._flop_counter import causal_pairs
    for length in (7, 16):
        for groups in (2, 4):
            counts = [causal_pairs(len(range(offset, length, groups)), length, stride=groups, offset=offset)
                      for offset in range(groups)]
            assert sum(counts) == length*(length+1)//2

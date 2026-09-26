"""Closed Figure 7a matrix and fixed-pass scoring checks."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from studies.protocol import evaluate_fixed_passes
from studies.schedules.evaluate import validate_checkpoint
from studies.schedules.matrix import GRAD, MODES, NO_GRAD, SEEDS, recipe_path, validate_recipe
from training.recipes import load_recipe
from white_matter.models import register_models
from white_matter.models.white_matter.configuration_white_matter import WhiteMatterConfig


def test_schedule_matrix_contains_exactly_48_valid_paper_recipes():
    register_models()
    paths = {path.resolve() for path in Path("studies/schedules/recipes").glob("seed*/*.yaml")}
    assert len(paths) == len(SEEDS) * len(NO_GRAD) * len(GRAD) * len(MODES) == 48
    for seed in SEEDS:
        for no_grad in NO_GRAD:
            for grad in GRAD:
                for mode in MODES:
                    arm = f"ng{no_grad}_g{grad}_{mode}"
                    path = recipe_path(seed, arm)
                    assert path in paths
                    recipe = load_recipe(path)
                    assert validate_recipe(recipe, path) == (seed, arm)
                    recipe.model.training_step = 20_000
                    recipe.model.training_sequence_length = 2048
                    recipe.model.recipe_name = recipe.name
                    assert validate_checkpoint(recipe.model) == (seed, arm)


def test_schedule_recipe_rejects_changed_model_size():
    path = recipe_path(1337, "ng1_g1_tp")
    recipe = load_recipe(path)
    recipe.model.hidden_size = 1792
    with pytest.raises(ValueError, match="hidden_size"):
        validate_recipe(recipe, path)


def test_fixed_pass_scorer_matches_direct_lm_ce_for_cyclic_and_jacobi():
    register_models()
    ids = torch.tensor([[1, 2, 100, 3, 4], [5, 100, 6, 7, 8]])
    loader = DataLoader([{"input_ids": row} for row in ids], batch_size=2)
    config = WhiteMatterConfig(
        vocab_size=101, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, rope_theta=10_000.0,
        eos_token_id=100, document_separator_token_id=100,
        num_kv_channels=2, num_passes=2, cyclic_groups=2,
        router_layer_stride=1, router_prior="shifted_identity:0.25",
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    for mode in ("cyclic", "jacobi"):
        model.config.execution_mode = mode
        for passes in (1, 2):
            loss_sum, targets = evaluate_fixed_passes(model, loader, num_passes=passes)
            assert targets == 8
            with torch.inference_mode():
                logits = model(ids, num_passes=passes).logits[:, :-1]
                expected = F.cross_entropy(logits.float().reshape(-1, 101), ids[:, 1:].reshape(-1))
            assert loss_sum / targets == pytest.approx(float(expected), abs=1e-5)


@pytest.mark.parametrize('arm,horizon,crossing', [
    ('ng4_g2_c4', 96, 36), ('ng4_g1_c8', 128, 105), ('ng4_g2_c16', 96, None),
])
def test_extended_schedule_aggregation(tmp_path, monkeypatch, arm, horizon, crossing):
    import json
    from studies.schedules import analyze
    from studies.schedules.matrix import arm_values, evaluation_horizon

    no_grad, grad, mode = arm_values(arm)
    monkeypatch.setattr(analyze, 'NO_GRAD', (no_grad,))
    monkeypatch.setattr(analyze, 'GRAD', (grad,))
    monkeypatch.setattr(analyze, 'MODES', (mode,))
    assert evaluation_horizon(arm, 'tp') == horizon
    assert evaluation_horizon(arm, 'native') == evaluation_horizon(arm, 'cyclic16') == 32
    for seed in SEEDS:
        directory = tmp_path / f'seed{seed}' / arm
        directory.mkdir(parents=True)
        for evaluation_mode in ('native', 'cyclic16', 'tp'):
            count = horizon if evaluation_mode == 'tp' else 32
            rows = [dict(n_passes=p, perplexity=(10 if evaluation_mode != 'tp' or
                    (crossing is not None and p >= crossing) else 12)) for p in range(1, count + 1)]
            (directory / f'eval_{evaluation_mode}.json').write_text(json.dumps(dict(
                protocol='figure7a_sequential_final', seed=seed, arm=arm,
                evaluation_mode=evaluation_mode, n_tok=analyze.PAPER_TEST_TARGETS, rows=rows,
            )))
    per_seed, averaged = analyze.collect(tmp_path)
    for row in per_seed + averaged:
        assert row['jacobi_passes_within_1pct'] == crossing
        assert row['jacobi_max_evaluated_pass'] == horizon
        assert row['jacobi_passes_within_1pct_censored'] == (crossing is None)
        assert row['native_best_perplexity'] == row['cyclic16_pass32_perplexity'] == 10
    path = tmp_path / 'seed1338' / arm / 'eval_tp.json'
    payload = json.loads(path.read_text())
    payload['rows'].append(dict(n_passes=horizon + 1, perplexity=10))
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='horizons differ'):
        analyze.collect(tmp_path)
    payload['rows'] = payload['rows'][:32]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='requires consecutive passes'):
        analyze.collect(tmp_path)


@pytest.mark.parametrize('mode,last_pass,accepted', [('tp', 105, True), ('tp', 128, True),
                                                   ('native', 105, False), ('cyclic16', 96, False)])
def test_schedule_cli_extended_pass_range(monkeypatch, tmp_path, mode, last_pass, accepted):
    import sys
    from studies.schedules import evaluate

    def stop_before_loading(_):
        raise RuntimeError('range accepted')

    monkeypatch.setattr(evaluate, 'validate_paper_cache', stop_before_loading)
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--model', 'unused', '--data-dir', str(tmp_path),
                                    '--output', str(tmp_path / 'result.json'), '--mode', mode,
                                    '--last-pass', str(last_pass)])
    if accepted:
        with pytest.raises(RuntimeError, match='range accepted'):
            evaluate.main()
    else:
        with pytest.raises(SystemExit) as error:
            evaluate.main()
        assert error.value.code == 2

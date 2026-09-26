"""The documented plotting script works without checkout imports on PYTHONPATH."""

import csv
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_schedule_plot_runs_as_standalone_script(tmp_path):
    pytest.importorskip('matplotlib')
    source, output = tmp_path / 'schedules.csv', tmp_path / 'schedules.pdf'
    with source.open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['no_gradient_passes', 'gradient_passes', 'schedule',
                         'native_best_perplexity', 'cyclic16_pass32_perplexity', 'jacobi_passes_within_1pct'])
        for no_grad in (1, 2, 4):
            for grad in (1, 2):
                for mode in ('tp', 'c4', 'c8', 'c16'):
                    writer.writerow([no_grad, grad, mode, 12, 11, '' if mode == 'tp' else 8])
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    env['MPLCONFIGDIR'] = str(tmp_path / 'matplotlib')
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'plot_experiments.py'
    subprocess.run([sys.executable, str(script), 'schedules', '--input', str(source), '--output', str(output)],
                   cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert output.read_bytes().startswith(b'%PDF-')

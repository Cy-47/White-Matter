"""Capture the timing driver and all schedule specializations on CPU."""

from types import SimpleNamespace

import pytest
import torch

from studies.prefill_convergence import benchmark
from studies.prefill_convergence.contiguous import _forward
from studies.prefill_convergence.protocol import MODES
from tests.unit.test_experiment_protocols import tiny_model
from white_matter.compilation import execution_policy


@pytest.fixture
def captured_graphs(monkeypatch):
    torch.compiler.reset()
    graphs = []
    compile = torch.compile

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def capture(fn, **kwargs):
        kwargs.pop("options")
        return compile(fn, backend=backend, **kwargs)

    monkeypatch.setattr(torch, "compile", capture)
    yield graphs
    torch.compiler.reset()


@torch.inference_mode()
def test_contiguous_trial_captures_without_changing_compiler_stance(monkeypatch, captured_graphs):
    model = tiny_model()
    x = torch.randn(1, 4, 16)
    positions = torch.arange(4)[None]

    def reference_attention(block, x, **kwargs):
        kwargs["backend"] = "reference"
        return _forward(block, x, **kwargs)

    monkeypatch.setattr(benchmark, "contiguous_forward", reference_attention)
    expected = _forward(
        model.model.decoder.block,
        x,
        num_passes=2,
        chunks=2,
        backend="reference",
        on_pass=None,
        output_final_state=False,
    )
    with execution_policy():
        trial = benchmark._compile_trial(model, x, positions)
        actual = trial("contiguous2", 2)
    torch.testing.assert_close(actual, expected)
    assert captured_graphs


@torch.inference_mode()
def test_all_timing_cases_fit_the_compilation_budget(monkeypatch, captured_graphs):
    # Small tensor kernels isolate the driver's dispatch and specialization budget.
    class Block:
        def forward_jacobi(self, x, *, num_passes, **kwargs):
            return x + num_passes

        def __call__(self, x, *, num_passes, cyclic_groups, **kwargs):
            return x + num_passes + cyclic_groups

    class Decoder:
        block = Block()

        def __call__(self, x, **kwargs):
            return x + 1

    def contiguous(block, x, *, num_passes, chunks, **kwargs):
        return x + num_passes + chunks

    model = SimpleNamespace(model=SimpleNamespace(decoder=Decoder()), allocate_inference_cache=lambda length: None)
    monkeypatch.setattr(benchmark, "contiguous_forward", contiguous)
    x = torch.zeros(1, 4, 16)
    cases = {"ar": None, **dict.fromkeys(MODES, 2)}
    with execution_policy():
        trial = benchmark._compile_trial(model, x, torch.arange(4)[None])
        for mode, passes in cases.items():
            expected = 1 if mode == "ar" else 2 + (MODES[mode] or 0)
            torch.testing.assert_close(trial(mode, passes), x + expected)
        count = len(captured_graphs)
        for mode, passes in cases.items():
            trial(mode, passes)
    assert count == len(cases)
    assert len(captured_graphs) == count

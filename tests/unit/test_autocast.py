"""Check parameter gradients across detached and live autocast phases."""

import copy

import pytest
import torch


@pytest.mark.parametrize("no_gradient_passes", [0, 2])
@pytest.mark.parametrize("gradient_passes", [1, 2])
def test_autocast_cache_preserves_gradients_across_phases_and_updates(no_gradient_passes, gradient_passes):
    torch.manual_seed(93)
    reference = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8))
    candidate = copy.deepcopy(reference)
    batches = [torch.randn(2, 8) for _ in range(2)]

    def run(model, cache_enabled):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        records = []
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            inputs = batch.clone().requires_grad_(True)
            with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=True):
                state = inputs
                # Disable caching only in the detached phase of the reference.
                # Both gradient phases reuse casts: disabling caching there
                # changes BF16 versus FP32 parameter-gradient accumulation.
                with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=cache_enabled):
                    for _ in range(no_gradient_passes):
                        state = model(state)
                for _ in range(gradient_passes):
                    state = model(state + inputs)
                loss = state.float().square().mean()
            loss.backward()
            gradients = {}
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                gradients[name] = parameter.grad.clone()
            assert inputs.grad is not None
            optimizer.step()
            records.append(
                (
                    state.detach(),
                    loss.detach(),
                    inputs.grad,
                    gradients,
                    {name: parameter.detach().clone() for name, parameter in model.named_parameters()},
                )
            )
        return records

    expected = run(reference, False)
    actual = run(candidate, True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

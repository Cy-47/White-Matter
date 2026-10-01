"""Top-layer feedback must reach future tokens and survive every execution schedule."""

import pytest
import torch

from tests.unit.test_model_serialization import config
from white_matter import WhiteMatterForCausalLM
from white_matter.modules.routing import _init_logits


@pytest.mark.parametrize("include_top_output", [False, True])
@pytest.mark.parametrize("mode", ["autoregressive", "jacobi", "cyclic"])
def test_top_output_is_a_differentiable_pool_source(mode, include_top_output):
    cfg = config()
    assert cfg.include_top_output is True
    cfg.include_top_output = include_top_output
    cfg.router_prior = "top:0.25"
    block = WhiteMatterForCausalLM(cfg).model.decoder.block.double()
    x = torch.randn(1, 4, cfg.hidden_size, dtype=torch.float64, requires_grad=True)
    with torch.compiler.set_stance("force_eager"):
        if mode == "autoregressive":
            _, state = block.forward_recurrent(x)
        elif mode == "jacobi":
            _, state = block.forward_jacobi(x, output_final_state=True)
        else:
            _, state = block(x, cyclic_groups=2, output_final_state=True)
        # With static initial top routing, the pool must depend on the final
        # MLP only when its output is included among the sources.
        gradient = torch.autograd.grad(
            state[1].square().sum(), block.layers[-1].mlp.down_proj.weight, allow_unused=True
        )[0]
    if include_top_output:
        assert gradient is not None
        assert gradient.abs().sum() > 0
    else:
        assert gradient is None


@pytest.mark.parametrize("prior", ["identity:0.25", "shifted_identity:0.25"])
def test_identity_priors_allow_extra_source(prior):
    logits = _init_logits(3, 4, prior)
    shift = int(prior.startswith("shifted"))
    torch.testing.assert_close(logits.argmax(-1), torch.arange(3) + shift)

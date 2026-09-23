import subprocess
import sys

import pytest
import torch

from white_matter.ops import cyclic_attention, prepare_cyclic_attention_metadata


def test_core_import_does_not_require_transformers_or_tilelang():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from importlib.abc import MetaPathFinder
class BlockOptional(MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'transformers', 'tilelang', 'flash_attn'}:
            raise AssertionError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from white_matter.ops import cyclic_attention
assert callable(cyclic_attention)
""",
        ],
        check=True,
    )


@pytest.mark.parametrize("document_masked", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
def test_cyclic_attention_outputs_and_gradients_match_explicit_attention(document_masked, offset):
    torch.manual_seed(42)
    q = torch.randn(2, 4, 5, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, 11, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    slots = offset + 2 * torch.arange(5)
    keep = (slots[:, None] >= torch.arange(11)[None, :]).expand(2, 5, 11)
    metadata = None
    if document_masked:
        segments = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3], [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]])
        q_seg = segments[:, slots]
        k_seg = torch.cat((torch.full((2, 1), -1), segments[:, :-1]), dim=1)
        metadata = prepare_cyclic_attention_metadata(q_seg, k_seg)
        keep = keep & ((q_seg[:, :, None] == k_seg[:, None, :]) | (k_seg[:, None, :] == -1))
    actual = cyclic_attention(q, k, v, query_stride=2, query_offset=offset, metadata=metadata)
    scores = q @ k.repeat_interleave(2, dim=1).transpose(-1, -2) / 8**0.5
    expected = scores.masked_fill(~keep[:, None], -torch.inf).softmax(-1) @ v.repeat_interleave(2, dim=1)
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), probe, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (q, k, v), probe)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=1e-12, atol=1e-12)


def test_invalid_backend_and_metadata_fail_explicitly():
    q = torch.randn(1, 2, 3, 8)
    with pytest.raises(ValueError, match="CUDA BF16"):
        cyclic_attention(q, q, q, backend="tilelang")
    with pytest.raises(ValueError, match="query_offset"):
        cyclic_attention(q, q, q, query_offset=-1)
    with pytest.raises(ValueError, match="dummy"):
        prepare_cyclic_attention_metadata(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long))

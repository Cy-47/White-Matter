"""Cache partitions and packing preserve the physical token order."""

import json

import pytest

from scripts._packing import Packer
from training.data import split_row_indices


@pytest.mark.parametrize("sizes", [(4, 2, 3), (0, 2, 0)])
def test_cache_splits_are_disjoint_contiguous_ranges(tmp_path, sizes):
    (tmp_path / "cache_meta.json").write_text(json.dumps({"n_total": 10}))
    offset = 0
    for split, size in zip(("train", "val", "test"), sizes, strict=True):
        indices = split_row_indices(tmp_path, split, n_train=sizes[0], n_val=sizes[1], n_test=sizes[2])
        assert isinstance(indices, range)
        assert list(indices) == list(range(offset, offset + size))
        offset += size
    with pytest.raises(ValueError, match="exceed"):
        split_row_indices(tmp_path, "train", n_train=11, n_val=0, n_test=0)


def test_packer_preserves_eos_and_partial_row_across_flushes():
    packer = Packer(4, 99)
    packer.add_doc([1, 2])
    assert packer.take_rows().shape == (0, 4)
    packer.add_doc([3, 4, 5, 6, 7, 8])
    assert packer.take_rows().tolist() == [[1, 2, 99, 3], [4, 5, 6, 7]]
    assert packer.take_rows().shape == (0, 4)
    assert packer.buf == [8, 99]

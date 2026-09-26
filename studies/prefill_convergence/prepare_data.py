"""Select the paper's 192 disjoint, single-document test windows."""

import argparse
from pathlib import Path

import numpy as np

from benchmarks._measurement import digest, write_json
from studies.protocol import validate_paper_cache
from training.data import split_row_indices


def select_disjoint_pairs(no_eos, count):
    if no_eos.ndim != 1 or count < 1:
        raise ValueError("expected a row mask and positive count")
    pairs, i = [], 0
    while i + 1 < len(no_eos) and len(pairs) < count:
        if no_eos[i] and no_eos[i + 1]:
            pairs.append((i, i + 1))
            i += 2
        else:
            i += 1
    if len(pairs) != count:
        raise ValueError(f"need {count} disjoint single-document windows, found {len(pairs)}")
    return pairs


def prepare(cache, output):
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    meta = validate_paper_cache(cache)
    indices = split_row_indices(cache, "test", **{k: meta["splits"][k] for k in ("n_train", "n_val", "n_test")})
    tokens = np.load(cache / "tokenized.npy", mmap_mode="r")
    rows = tokens[indices.start:indices.stop]
    pairs = select_disjoint_pairs(np.all(rows != 151643, axis=1), 192)
    # Preserve the original pair selection even though Figure 5 uses only its first half.
    windows = np.stack([np.concatenate((rows[a], rows[b])) for a, b in pairs])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, windows)
    write_json(output.with_suffix('.json'), dict(
        protocol="prefill_convergence", source=str(cache.resolve()),
        source_row_pairs=[[indices.start+a, indices.start+b] for a, b in pairs],
        selection="first disjoint adjacent test row pairs without EOS", eos_id=151643,
        shape=list(windows.shape), sha256=digest(output), evaluation_length=2048,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != '.npy':
        parser.error('--output must end in .npy')
    prepare(args.data_dir, args.output)


if __name__ == '__main__':
    main()

"""Pool disjoint quality shards before selecting passes; join validated timings."""

import argparse
import json
import math
from pathlib import Path

from benchmarks._measurement import write_json
from studies.prefill_convergence.protocol import MODES, first_crossing


def pool(shards):
    if not shards:
        raise ValueError('no quality shards')
    first = shards[0]
    identity = ('protocol', 'precision', 'checkpoint', 'windows_sha256', 'sequence_length')
    if first['protocol'] != 'prefill_convergence' or first['precision'] != 'fp32' or first['sequence_length'] != 2048:
        raise ValueError('wrong convergence quality protocol')
    seen, ar_sum, targets = set(), 0.0, 0
    curves = {mode: [0.0] * len(first['curves'][mode]) for mode in MODES}
    for shard in shards:
        if any(shard[key] != first[key] for key in identity):
            raise ValueError('quality shards have different identities')
        rows = set(range(shard['offset'], shard['offset']+shard['count']))
        if not rows or seen & rows or not rows <= set(range(192)):
            raise ValueError('overlapping or invalid quality shards')
        if shard['targets'] != len(rows)*2047 or set(shard['curves']) != set(MODES):
            raise ValueError('invalid shard targets or modes')
        seen |= rows
        ar_sum += shard['ar_ce_sum']
        targets += shard['targets']
        for mode, values in curves.items():
            incoming = shard['curves'][mode]
            if len(incoming) != len(values) or not values:
                raise ValueError('quality shards have different pass ranges')
            curves[mode] = [a+b for a, b in zip(values, incoming, strict=True)]
    if seen != set(range(192)):
        raise ValueError('quality shards must cover all 192 windows')
    reference = ar_sum/targets
    result = dict(protocol='prefill_convergence', checkpoint=first['checkpoint'],
                  windows_sha256=first['windows_sha256'], sequence_length=2048,
                  targets=targets, ar_ce=reference, ar_perplexity=math.exp(reference), tolerance=0.01, modes={})
    for mode, sums in curves.items():
        losses = [value/targets for value in sums]
        crossing = first_crossing(losses, reference)
        result['modes'][mode] = dict(ce=losses, passes=crossing, status='reached' if crossing else 'not_reached')
    return result


def join_timings(quality, timing):
    for key in ('checkpoint', 'windows_sha256', 'sequence_length'):
        if quality[key] != timing[key]:
            raise ValueError(f'timing identity mismatch: {key}')
    expected = {'ar', *[m for m, row in quality['modes'].items() if row['passes'] is not None]}
    if set(timing['rows']) != expected or timing['batch_size'] != 64 or timing['precision'] != 'bf16':
        raise ValueError('timing workload differs from paper protocol')
    for mode, row in timing['rows'].items():
        passes = None if mode == 'ar' else quality['modes'][mode]['passes']
        if row['passes'] != passes:
            raise ValueError('timing pass count differs from quality selection')
    return {**quality, 'timing': timing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality', type=Path, nargs='+', required=True)
    parser.add_argument('--timing', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = pool([json.loads(p.read_text()) for p in args.quality])
    if args.timing:
        result = join_timings(result, json.loads(args.timing.read_text()))
    write_json(args.output, result)


if __name__ == '__main__':
    main()

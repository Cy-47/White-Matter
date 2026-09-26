"""Pool disjoint quality shards before selecting passes; join validated timings."""

import argparse
import json
import math
from pathlib import Path

from benchmarks._measurement import write_json
from studies.prefill_convergence.protocol import MODES, first_crossing, schedules


def pool(shards):
    if not shards:
        raise ValueError('no quality shards')
    first = shards[0]
    identity = ('protocol', 'precision', 'checkpoint', 'windows_sha256', 'sequence_length')
    if first['protocol'] != 'prefill_convergence' or first['precision'] != 'fp32' or first['sequence_length'] != 2048:
        raise ValueError('wrong convergence quality protocol')
    seen, ar_sum, targets = set(), 0.0, 0
    modes = set(first['curves'])
    if not modes or modes - MODES.keys():
        raise ValueError('unknown or empty quality modes')
    if any(m.startswith('contiguous') for m in modes) and first.get('schedules') != schedules(modes):
        raise ValueError('missing or invalid schedule semantics')
    curves = {mode: [0.0] * len(values) for mode, values in first['curves'].items()}
    for shard in shards:
        if any(shard[key] != first[key] for key in identity):
            raise ValueError('quality shards have different identities')
        if shard.get('schedules') != first.get('schedules'):
            raise ValueError('quality shards have different schedules')
        rows = set(range(shard['offset'], shard['offset']+shard['count']))
        if not rows or seen & rows or not rows <= set(range(192)):
            raise ValueError('overlapping or invalid quality shards')
        if shard['targets'] != len(rows)*2047 or set(shard['curves']) != modes:
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
                  targets=targets, ar_ce=reference, ar_perplexity=math.exp(reference), tolerance=0.01,
                  schedules=first.get('schedules'), modes={})
    for mode, sums in curves.items():
        losses = [value/targets for value in sums]
        crossing = first_crossing(losses, reference)
        result['modes'][mode] = dict(ce=losses, passes=crossing, status='reached' if crossing else 'not_reached')
    return result


def join_timings(quality, timing):
    expected = {'ar', *[m for m, row in quality['modes'].items() if row['passes'] is not None]}
    quality_schedules = quality.get('schedules')
    # Unreached modes retain their quality curves but have no threshold timing.
    expected_schedules = (None if quality_schedules is None else
                          {m: spec for m, spec in quality_schedules.items() if m in expected})
    if expected_schedules != timing.get('schedules'):
        raise ValueError('timing schedule mismatch')
    for key in ('checkpoint', 'windows_sha256', 'sequence_length'):
        if quality[key] != timing[key]:
            raise ValueError(f'timing identity mismatch: {key}')
    batch_size = timing['batch_size']
    if (set(timing['rows']) != expected or type(batch_size) is not int or not 1 <= batch_size <= 192
            or timing['precision'] != 'bf16'):
        raise ValueError('invalid convergence timing workload')
    for mode, row in timing['rows'].items():
        passes = None if mode == 'ar' else quality['modes'][mode]['passes']
        if row['passes'] != passes:
            raise ValueError('timing pass count differs from quality selection')
    return {**quality, 'timing': timing}


def merge_timings(shards):
    """Combine disjoint timing cases without pooling samples from different GPUs."""
    if not shards:
        raise ValueError('no timing shards')
    first = shards[0]
    identity = ('protocol', 'checkpoint', 'windows_sha256', 'sequence_length', 'batch_size',
                'precision', 'compiled', 'scope', 'warmups', 'repetitions', 'ar_repetitions',
                'contiguous_backend')
    rows, specs = {}, {}
    for shard in shards:
        if any(shard[key] != first[key] for key in identity) or shard['source']['sha256'] != first['source']['sha256']:
            raise ValueError('timing shards have different protocols or source identities')
        if any(shard['environment'][key] != first['environment'][key] for key in ('gpu', 'torch', 'cuda', 'packages')):
            raise ValueError('timing shards have different hardware or software')
        if not shard['rows'] or rows.keys() & shard['rows'].keys():
            raise ValueError('empty or overlapping timing cases')
        if shard['schedules'] != schedules(set(shard['rows']) - {'ar'}):
            raise ValueError('timing schedule semantics mismatch')
        rows.update(shard['rows'])
        specs.update(shard['schedules'])
    return {**first, 'rows': rows, 'schedules': specs,
            'worker_environments': [s['environment'] for s in shards]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality', type=Path, nargs='+', required=True)
    parser.add_argument('--timing', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = pool([json.loads(p.read_text()) for p in args.quality])
    if args.timing:
        result = join_timings(result, merge_timings([json.loads(p.read_text()) for p in args.timing]))
    write_json(args.output, result)


if __name__ == '__main__':
    main()

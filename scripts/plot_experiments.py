"""Render main-body experiment artifacts without the manuscript checkout."""

import argparse
import csv
import json
from pathlib import Path


def plot(kind, source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    if kind == 'convergence-sweep':
        data = json.loads(source.read_text())
        batch_size = data.get('timing', {}).get('batch_size', 64)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        timing = data.get('timing', {}).get('rows', {})
        for column, family in enumerate(('cyclic', 'contiguous')):
            modes = sorted((m for m in data['modes'] if m.startswith(family)), key=lambda m: int(m[len(family):]))
            groups = [int(m[len(family):]) for m in modes]
            for mode, group in zip(modes, groups, strict=True):
                losses = data['modes'][mode]['ce']
                axes[0, column].plot(np.arange(1, len(losses)+1), np.exp(losses), label=f'{group} chunks')
            ax = axes[0, column]
            if 'jacobi' in data['modes']:
                losses = data['modes']['jacobi']['ce']
                ax.plot(np.arange(1, len(losses)+1), np.exp(losses), color='gray', linestyle=':', label='Jacobi')
            ax.axhline(data['ar_perplexity'], color='black', label='Exact AR')
            ax.axhline(data['ar_perplexity']*(1+data['tolerance']), color='gray', linestyle='--', label='1% threshold')
            ax.set(title=family.title(), xlabel='Passes', ylabel='Perplexity', yscale='log')
            ax.legend(fontsize=8, ncol=2)
            axes[1, 0].plot(groups, [data['modes'][m]['passes'] or np.nan for m in modes], 'o-', label=family)
            axes[1, 1].plot(groups, [1000*timing[m]['seconds_per_sequence'] if m in timing else np.nan for m in modes],
                            'o-', label=family)
        if 'jacobi' in data['modes'] and data['modes']['jacobi']['passes'] is not None:
            axes[1, 0].axhline(data['modes']['jacobi']['passes'], color='gray', linestyle='--', label='Jacobi')
        for mode, style in [('jacobi', '--'), ('ar', ':')]:
            if mode in timing:
                axes[1, 1].axhline(1000*timing[mode]['seconds_per_sequence'], color='gray', linestyle=style, label=mode)
        for ax, label in zip(axes[1], ('Passes to 1% threshold', f'ms / sequence at threshold (batch {batch_size})'), strict=True):
            ax.set(xlabel='Chunks', ylabel=label, xscale='log', xticks=groups, xticklabels=groups)
            ax.minorticks_off()
            ax.legend()
    elif kind == 'convergence':
        data = json.loads(source.read_text())
        batch_size = data.get('timing', {}).get('batch_size', 64)
        has_curves = 'modes' in data
        fig, axes = plt.subplots(1, 2 if has_curves else 1, figsize=(12 if has_curves else 7, 4), squeeze=False)
        ax = axes[0, 0]
        if has_curves:
            for mode, row in data['modes'].items():
                ax.plot(np.arange(1, len(row['ce'])+1), np.exp(row['ce']), label=mode)
            ax.axhline(data['ar_perplexity'], color='black', label='Exact AR')
            ax.axhline(data['ar_perplexity']*(1+data['tolerance']), color='gray', linestyle='--', label='1% threshold')
            ax.set(xlabel='Passes', ylabel='Perplexity', yscale='log')
            ax.legend(fontsize=8)
            ax = axes[0, 1]
        rows = data.get('timing', {}).get('rows', {})
        labels = list(rows)
        ax.bar(labels, [rows[k]['seconds_per_sequence'] for k in labels])
        ax.set_ylabel(f'Seconds / sequence (batch {batch_size})')
        ax.tick_params(axis='x', rotation=35)
    else:
        with source.open() as stream:
            rows = list(csv.DictReader(stream))
        if kind == 'quality':
            fig, ax = plt.subplots(figsize=(6, 4))
            for row in rows:
                if row['model'].endswith('1p3b'):
                    continue
                x, y = float(row['non_embedding_parameters'])/1e6, float(row['heldout_perplexity'])
                ax.scatter(x, y)
                ax.annotate(row['model'], (x, y), xytext=(3, 4), textcoords='offset points', fontsize=8)
            ax.set(xlabel='Non-embedding parameters (M)', ylabel='Held-out perplexity')
        elif kind == 'rank':
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar([r['arm'] for r in rows], [float(r['perplexity']) for r in rows])
            ax.tick_params(axis='x', rotation=40)
            ax.set_ylabel('Held-out perplexity')
        elif kind == 'schedules':
            fig, axes = plt.subplots(1, 3, figsize=(11, 4))
            metrics = ('native_best_perplexity', 'cyclic16_pass32_perplexity', 'jacobi_passes_within_1pct')
            index = {(int(r['no_gradient_passes']), int(r['gradient_passes']), r['schedule']): r for r in rows}
            pairs = sorted({(n, g) for n, g, _ in index})
            modes = list(dict.fromkeys(mode for _, _, mode in index))
            for ax, metric in zip(axes, metrics, strict=True):
                values = np.array([[float(index[n, g, m][metric] or 'nan') for m in modes] for n, g in pairs])
                image = ax.imshow(values, aspect='auto')
                for i, j in np.ndindex(values.shape):
                    text = '>32' if np.isnan(values[i, j]) else f'{values[i, j]:.2f}' if metric != metrics[-1] else f'{values[i, j]:.0f}'
                    ax.text(j, i, text, ha='center', va='center', fontsize=8)
                ax.set_xticks(range(len(modes)), modes)
                ax.set_yticks(range(len(pairs)), [f'ng={n}, grad={g}' for n, g in pairs])
                ax.set_title(metric.replace('_', ' '), fontsize=9)
                fig.colorbar(image, ax=ax, shrink=.7)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            families = list(dict.fromkeys(r['model'] for r in rows))
            for ax, metric in zip(axes, ('relative_throughput', 'peak_gib'), strict=True):
                for i, phase in enumerate(('prefill', 'decode')):
                    selected = {r['model']: r for r in rows if r['phase'] == phase}
                    ax.bar(np.arange(len(families)) + (i-.5)*.35,
                           [float(selected[f][metric]) for f in families], width=.35, label=phase)
                ax.set_xticks(range(len(families)), families, rotation=25)
                ax.set_ylabel(metric.replace('_', ' '))
                ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('quality', 'convergence', 'convergence-sweep', 'schedules', 'rank', 'runtime'))
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.kind, args.input, args.output)


if __name__ == '__main__':
    main()

"""Compiled BF16 decoder timing at the passes selected by FP32 evaluation."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from benchmarks._measurement import checkpoint_files, environment, summarize, write_json, snapshot_source, verify_sources
from evals.loading import load_complete_model
from studies.prefill_convergence.evaluate import load_windows
from studies.prefill_convergence.protocol import MODES, validate_checkpoint
from training.compile import compile_feedback
from white_matter.models import register_models
from white_matter.modules.precision import model_autocast_context


@torch.inference_mode()
def measure(model, ids, selections, *, warmups=5, repetitions=30, ar_repetitions=10):
    if not ids.is_cuda:
        raise ValueError('convergence timing requires CUDA')
    model.config.document_separator_token_id = None
    model.config.prefill_mode = 'autoregressive'
    block = model.model.decoder.block
    compile_feedback(model, mode='default')
    block._run_token_layers = torch.compile(block._run_token_layers, dynamic=True, fullgraph=False)
    rows = {}
    with model_autocast_context(ids.device):
        x = model.model.embed_tokens(ids)
        positions = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
        for mode, passes in {'ar': None, **selections}.items():
            if passes is None and mode != 'ar':
                continue

            def trial(mode=mode, passes=passes):
                if mode == 'ar':
                    cache = model.allocate_inference_cache(ids.shape[1])
                    return model.model.decoder(x, past_key_values=cache, position_ids=positions)
                if mode == 'jacobi':
                    return block.forward_jacobi(x, num_passes=passes, num_gradient_passes=0)
                return block(x, num_passes=passes, cyclic_groups=MODES[mode], num_gradient_passes=0)

            for _ in range(warmups):
                trial()
            torch.cuda.synchronize()
            samples = []
            for _ in range(ar_repetitions if mode == 'ar' else repetitions):
                start = perf_counter()
                trial()
                torch.cuda.synchronize()
                samples.append(perf_counter()-start)
            stats = summarize(samples)
            rows[mode] = dict(passes=passes, **stats, seconds_per_sequence=stats['median_seconds']/len(ids))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True, help='Pooled quality JSON from analyze')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    quality = json.loads(args.quality.read_text())
    windows, meta = load_windows(args.windows)
    identity = checkpoint_files(args.model)
    if quality['checkpoint'] != identity or quality['windows_sha256'] != meta['sha256']:
        raise ValueError('timing inputs differ from quality inputs')
    register_models()
    model = load_complete_model(args.model, dtype=torch.float32).cuda().eval()
    validate_checkpoint(model.config)
    for module in model.modules():
        if hasattr(module, 'attention_implementation'):
            module.attention_implementation = 'flash_attention_2'
    model.config._attn_implementation = 'flash_attention_2'
    ids = torch.from_numpy(windows[:64].astype(np.int64)).cuda()
    source = snapshot_source(args.output.parent/"sources")
    rows = measure(model, ids, {mode: row['passes'] for mode, row in quality['modes'].items()})
    if checkpoint_files(args.model) != identity:
        raise RuntimeError('checkpoint changed during timing')
    verify_sources(source)
    write_json(args.output, dict(
        source=source,
        protocol='prefill_convergence', checkpoint=identity, windows_sha256=meta['sha256'],
        sequence_length=2048, batch_size=64, precision='bf16', compiled=True,
        scope='decoder excluding embeddings, final norm, LM head and token selection',
        warmups=5, repetitions=30, ar_repetitions=10, rows=rows, environment=environment(0),
    ))


if __name__ == '__main__':
    main()

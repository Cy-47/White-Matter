"""FP32 exact-AR reference and pass trajectories, with optional disjoint shards."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks._measurement import checkpoint_files, digest, environment, write_json, snapshot_source, verify_sources
from evals.execution import execution, score_hidden
from evals.loading import load_complete_model
from studies.prefill_convergence.protocol import MODES, validate_checkpoint
from white_matter.models import register_models


def load_windows(path):
    meta = json.loads(path.with_suffix('.json').read_text())
    windows = np.load(path)
    if (meta.get('protocol') != 'prefill_convergence' or digest(path) != meta.get('sha256')
            or windows.shape != (192, 4096) or windows.dtype != np.int32 or np.any(windows == 151643)):
        raise ValueError('invalid convergence windows or fingerprint')
    return windows[:, :2048], meta


@torch.inference_mode()
def measure_curves(model, ids, *, limits, batch_size):
    """Run each trajectory once; observe outputs without restarting its passes."""
    if batch_size < 1 or not len(ids) or any(n < 1 for n in limits.values()):
        raise ValueError('positive batch, data, and pass limits required')
    config = model.config
    if config.model_type != 'white_matter' or config.num_pre_layers or config.num_post_layers:
        raise ValueError('convergence requires a pure WhiteMatter decoder')
    totals = {mode: [0.0] * n for mode, n in limits.items()}
    ar_sum = 0.0
    device = next(model.parameters()).device
    block = model.model.decoder.block
    separator = config.document_separator_token_id
    config.document_separator_token_id = None
    try:
        with execution(model, mode='autoregressive', precision='fp32'):
            for start in range(0, len(ids), batch_size):
                batch = ids[start:start+batch_size].to(device)
                ar = model.model(batch, use_cache=False).last_hidden_state
                ar_sum += score_hidden(ar, batch, model.lm_head.weight)
                del ar
                x = model.model.embed_tokens(batch)
                for mode, limit in limits.items():
                    def observe(passes, hidden, mode=mode, batch=batch):
                        normalized = model.model.norm(hidden)
                        totals[mode][passes-1] += score_hidden(normalized, batch, model.lm_head.weight)
                    if mode == 'jacobi':
                        block.forward_jacobi(x, num_passes=limit, num_gradient_passes=0, on_pass=observe)
                    else:
                        block(x, num_passes=limit, cyclic_groups=MODES[mode], num_gradient_passes=0, on_pass=observe)
    finally:
        config.document_separator_token_id = separator
    return dict(ar_ce_sum=ar_sum, targets=len(ids)*(ids.shape[1]-1), curves=totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--count', type=int, default=192)
    parser.add_argument('--jacobi-passes', type=int, default=80)
    parser.add_argument('--cyclic-passes', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    windows, meta = load_windows(args.windows)
    if args.offset < 0 or args.count < 1 or args.offset+args.count > len(windows):
        parser.error('shard lies outside the 192 windows')
    register_models()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_complete_model(args.model, dtype=torch.float32).to(device).eval()
    validate_checkpoint(model.config)
    identity = checkpoint_files(args.model)
    ids = torch.from_numpy(windows[args.offset:args.offset+args.count].astype(np.int64))
    limits = {mode: args.jacobi_passes if mode == 'jacobi' else args.cyclic_passes for mode in MODES}
    source = snapshot_source(args.output.parent/"sources")
    result = measure_curves(model, ids, limits=limits, batch_size=args.batch_size)
    if identity != checkpoint_files(args.model):
        raise RuntimeError('checkpoint changed during evaluation')
    verify_sources(source)
    write_json(args.output, dict(
        source=source,
        protocol='prefill_convergence', precision='fp32', model=args.model, checkpoint=identity,
        windows_sha256=meta['sha256'], offset=args.offset, count=args.count,
        sequence_length=2048, environment=environment(0 if device == 'cuda' else None), **result,
    ))


if __name__ == '__main__':
    main()

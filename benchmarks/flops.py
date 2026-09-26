"""Measure main-body decoder FLOPs, excluding the LM head and optimizer updates."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from benchmarks._flop_counter import make_counter, single_pass_formula
from benchmarks._measurement import environment, write_json, digest, snapshot_source, verify_sources
from training.recipes import load_recipe
from white_matter.models import register_models
from white_matter.models.cache import DecoderCache
from white_matter.modules.precision import model_autocast_context

PAPER_ARMS = ('vanilla_16l', 'fusedkv', 'lckv_w4', 'lckv_w7', 'white_matter_k8', 'white_matter_k16')


def fused_generation(decoder, x, cache, position):
    """Count the paper's shortened prefill using the existing FusedKV layers.

    The public FusedKV generator recomputes prefixes. This measurement adapter
    implements the paper's cached workload without duplicating layer math.
    """
    positions = torch.arange(position, position+x.shape[1], device=x.device)[None]
    rope = decoder.rotary_emb(x, positions)
    sources = []
    for index, layer in enumerate(decoder.source_layers):
        x, key, value = layer(x, position_embeddings=rope, attention_mask=None,
                              past_key_values=cache, cached_attention=bool(position))
        if index in (0, len(decoder.source_layers)-1):
            sources.append((key, value))
    x = x[:, -1:]
    rope = tuple(t[:, -1:] for t in rope)
    for layer, fusion in zip(decoder.reconstruction_layers, decoder.fusions, strict=True):
        key, value = fusion(sources[0][0], sources[-1][0], sources[0][1], sources[-1][1])
        x = layer(x, key_states=key, value_states=value, position_embeddings=rope, attention_mask=None)
    return x


def seed_cache(decoder, config, context, device):
    prefixes = getattr(decoder, 'cache_prefix_slots', (0,)*(config.num_hidden_layers//2))
    cache = DecoderCache(prefixes)
    for index, extra in enumerate(prefixes):
        channels = (getattr(config, 'num_kv_channels', 1)
                    if config.model_type == 'white_matter' and index == config.num_pre_layers else 1)
        tensor = torch.zeros(1, channels*config.num_key_value_heads, context+extra, config.head_dim,
                             device=device, dtype=torch.bfloat16)
        cache.update(tensor, tensor.clone(), index)
    cache.seen_tokens = context
    cache.position = torch.full((1, 1), context, device=device, dtype=torch.long)
    return cache


def measure(recipe, *, length=2048, context=2048):
    if not torch.cuda.is_available():
        raise RuntimeError('FLOP measurement requires CUDA and the production attention backends')
    config = recipe.model
    config._attn_implementation = 'flash_attention_2'
    config.document_separator_token_id = None
    if config.model_type in {'white_matter', 'lckv'}:
        config.prefill_mode = config.execution_mode
    torch.manual_seed(1337)
    model = AutoModelForCausalLM.from_config(config).cuda()
    decoder = model.model.decoder
    result = {}
    # Count dispatcher-visible algorithmic work, not compiler fusion or replay.
    with torch.compiler.set_stance('force_eager'):
        for regime in ('train', 'prefill', 'decode'):
            decoder.train(regime == 'train')
            model.zero_grad(set_to_none=True)
            x = torch.randn(1, 1 if regime == 'decode' else length, config.hidden_size,
                            device='cuda', requires_grad=regime == 'train')
            cache = seed_cache(decoder, config, context, x.device) if regime == 'decode' else None
            counter = make_counter()
            with counter, model_autocast_context('cuda'), torch.set_grad_enabled(regime == 'train'):
                if regime == 'train':
                    out = decoder(x, num_passes=getattr(config, 'num_passes', 1),
                                  num_gradient_passes=recipe.gradient_passes or 1)
                    out.sum().backward()
                elif config.model_type == 'fusedkv':
                    fused_generation(decoder, x, cache or DecoderCache((0,)*len(decoder.source_layers)),
                                     context if regime == 'decode' else 0)
                elif regime == 'prefill':
                    decoder(x, num_passes=getattr(config, 'num_passes', 1), num_gradient_passes=0)
                else:
                    decoder(x, past_key_values=cache, position_ids=cache.position)
            counts = {str(op): int(n) for op, n in counter.get_flop_counts()['Global'].items()}
            if config.model_type == 'fusedkv':
                fusion = 6*(config.num_hidden_layers//2)*config.num_key_value_heads*config.head_dim
                counts['fusedkv.cache_fusion'] = fusion*(context+1 if regime == 'decode' else length)*(3 if regime == 'train' else 1)
            total = sum(counts.values())
            tokens = 1 if regime == 'decode' else length
            row = dict(flops=total, tokens=tokens, flops_per_token=total/tokens, per_op=counts)
            if config.model_type in {'vanilla', 'fusedkv'}:
                expected = single_pass_formula(config, regime=regime, length=length, context=context)
                row['independent_formula'] = expected
                if total != expected:
                    raise RuntimeError(f'{recipe.name}/{regime}: measured {total}, expected {expected}')
            result[regime] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipes', type=Path, nargs='+', default=[Path('recipes/paper')/f'{a}.yaml' for a in PAPER_ARMS])
    parser.add_argument('--sequence-length', type=int, default=2048)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.sequence_length, args.context) < 1:
        parser.error('length and context must be positive')
    register_models()
    source = snapshot_source(args.output.parent/"sources")
    rows = {}
    for path in args.recipes:
        recipe = load_recipe(path)
        if recipe.name in rows:
            raise ValueError('duplicate recipe name')
        rows[recipe.name] = dict(recipe_sha256=digest(path), regimes=measure(
            recipe, length=args.sequence_length, context=args.context,
        ))
    baseline = rows.get('vanilla_16l')
    if baseline:
        for row in rows.values():
            for regime, values in row['regimes'].items():
                values['relative_to_vanilla'] = values['flops_per_token']/baseline['regimes'][regime]['flops_per_token']
    verify_sources(source)
    write_json(args.output, dict(
        source=source,
        protocol='paper_flops', sequence_length=args.sequence_length, context=args.context,
        conventions=dict(mac=2, attention_forward=4, attention_backward=10, lm_head=False,
                         document_mask=False, compiled=False, optimizer=False),
        environment=environment(0), models=rows,
    ))


if __name__ == '__main__':
    main()

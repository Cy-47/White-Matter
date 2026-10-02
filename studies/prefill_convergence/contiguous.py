"""Inference-only contiguous feedback iteration; deliberately study-local."""

import torch

from white_matter.blocks.decoder_layer import run_feedback_layers
from white_matter.modules.precision import model_autocast_context


def _chunk(block, x, keys, values, query_rope, key_rope, query_start, return_output, backend):
    # Batch is fixed within a benchmark; keep it distinct from symbolic prefix lengths.
    for tensor in (x, keys, values):
        torch._dynamo.mark_static(tensor, 0)
    mask = None
    if backend == "reference":
        qpos = query_start + torch.arange(x.shape[1], device=x.device)
        kpos = torch.arange(keys.shape[-2], device=x.device) - int(block.use_dummy_token)
        mask = (kpos[None, :] < qpos[:, None])[None, None]
    hidden, states = run_feedback_layers(
        block.layers,
        x,
        keys.unbind(1),
        values.unbind(1),
        query_rope,
        jacobi=mask is None,
        decode_key_mask=mask,
        prefix_length=query_start,
        return_output=return_output,
        include_top_output=block.include_top_output,
    )
    key, value = block.kv_pool.project_sequence(torch.stack(states, dim=2), key_rope)
    return hidden, key, value


_compiled_chunk = torch.compile(
    _chunk, dynamic=True, fullgraph=False, recompile_limit=64, options={"emulate_precision_casts": True}
)


@torch.inference_mode()
def forward_contiguous(
    block, x, *, num_passes, chunks=16, backend="reference", compiled=False, on_pass=None, output_final_state=False
):
    """Publish each chunk after its layer sweep, preserving earlier-token visibility.

    Shared strict-causal attention handles both optional-dummy layouts.
    """
    if x.ndim != 3 or not 1 <= chunks <= x.shape[1] or num_passes < 1:
        raise ValueError("require nonempty inputs, 1 <= chunks <= length, and positive passes")
    if backend not in {"reference", "flash_attention_2"}:
        raise ValueError("unsupported contiguous attention backend")
    if backend == "flash_attention_2" and (
        not x.is_cuda or any(layer.self_attn.attention_implementation != backend for layer in block.layers)
    ):
        raise ValueError("FlashAttention requires CUDA and flash_attention_2 layers")
    if block.training:
        raise ValueError("contiguous iteration is inference-only")
    offset = int(block.use_dummy_token)
    length = x.shape[1]
    bounds = [i * length // chunks for i in range(chunks + 1)]
    query_rope, key_rope = block._prepare_rope(x)
    run_chunk = _compiled_chunk if compiled else _chunk
    with model_autocast_context(x.device):
        keys, values = block.kv_pool.project_sequence(
            x.unsqueeze(2).expand(-1, -1, block.kv_pool.num_layers, -1),
            key_rope,
            dummy_token=block.dummy_token,
        )
        for iteration in range(num_passes):
            observe = on_pass is not None or iteration == num_passes - 1
            outputs = []
            for start, end in zip(bounds[:-1], bounds[1:], strict=True):
                hidden, key, value = run_chunk(
                    block,
                    x[:, start:end],
                    keys[..., : end + offset, :],
                    values[..., : end + offset, :],
                    tuple(t[:, start:end] for t in query_rope),
                    tuple(t[:, start + offset : end + offset] for t in key_rope),
                    start,
                    observe,
                    backend,
                )
                keys[..., start + offset : end + offset, :].copy_(key)
                values[..., start + offset : end + offset, :].copy_(value)
                if observe:
                    outputs.append(hidden)
            if observe:
                hidden = torch.cat(outputs, dim=1)
                if on_pass is not None:
                    on_pass(iteration + 1, hidden)
    return (hidden, (keys, values)) if output_final_state else hidden

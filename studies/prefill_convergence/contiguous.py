"""Inference-only contiguous feedback iteration; deliberately study-local."""

import torch

from white_matter.blocks.decoder_layer import run_feedback_layers
from white_matter.modules.precision import model_autocast_context


def _chunk(block, x, keys, values, query_rope, key_rope, mask, return_output):
    # Batch is fixed within a benchmark; keep it distinct from symbolic prefix lengths.
    for tensor in (x, keys, values):
        torch._dynamo.mark_static(tensor, 0)
    hidden, states = run_feedback_layers(
        block.layers,
        x,
        keys.unbind(1),
        values.unbind(1),
        query_rope,
        decode_key_mask=mask,
        return_output=return_output,
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

    Slot zero is the dummy; token t writes slot t+1 and reads slots <=t.
    FlashAttention's bottom-right causal mask on Q[start:end], KV[:end]
    implements exactly this bound, including unequal query/key lengths.
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
    length = x.shape[1]
    bounds = [i * length // chunks for i in range(chunks + 1)]
    query_rope, key_rope = block._prepare_rope(x)
    run_chunk = _compiled_chunk if compiled else _chunk
    with model_autocast_context(x.device):
        keys, values = block.kv_pool.project_sequence(
            x.unsqueeze(2).expand(-1, -1, len(block.layers), -1),
            key_rope,
            dummy_token=block.dummy_token,
        )
        masks = [None] * chunks
        if backend == "reference":
            masks = [
                (torch.arange(end, device=x.device)[None, :] <= torch.arange(start, end, device=x.device)[:, None])[
                    None, None
                ]
                for start, end in zip(bounds[:-1], bounds[1:], strict=True)
            ]
        for iteration in range(num_passes):
            observe = on_pass is not None or iteration == num_passes - 1
            outputs = []
            for index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
                hidden, key, value = run_chunk(
                    block,
                    x[:, start:end],
                    keys[..., :end, :],
                    values[..., :end, :],
                    tuple(t[:, start:end] for t in query_rope),
                    tuple(t[:, start + 1 : end + 1] for t in key_rope),
                    masks[index],
                    observe,
                )
                keys[..., start + 1 : end + 1, :].copy_(key)
                values[..., start + 1 : end + 1, :].copy_(value)
                if observe:
                    outputs.append(hidden)
            if observe:
                hidden = torch.cat(outputs, dim=1)
                if on_pass is not None:
                    on_pass(iteration + 1, hidden)
    return (hidden, (keys, values)) if output_final_state else hidden

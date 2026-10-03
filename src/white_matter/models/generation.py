"""Bounded prefill and fixed-batch CUDA graph replay of the ordinary HF forward."""

import torch

from .cache import DecoderCache


@torch.no_grad()
def prefill(model, input_ids: torch.Tensor, cache: DecoderCache, *, batch_size: int = 1) -> torch.Tensor:
    """Return last-token logits, batching before embeddings/head to bound workspace.

    Requires an empty static cache. Batch views share its KV storage; decoding
    can subsequently use the full batch, including through DecodeGraph.
    """
    if model.training or cache.capacity is None or cache.seen_tokens:
        raise ValueError("prefill requires an eval model and an empty static cache")
    if model.config.document_separator_token_id is not None or cache.document_ids is not None:
        raise ValueError("prefill requires single-document inference")
    if batch_size < 1 or input_ids.ndim != 2 or min(input_ids.shape) < 1:
        raise ValueError("prefill requires nonempty (B,T) input_ids and a positive batch_size")
    batch = input_ids.shape[0]
    if cache.position is not None and cache.position.shape[0] != batch:
        raise ValueError("cache batch differs from inputs")
    if batch_size >= batch:
        return model(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
    output = None
    for start in range(0, batch, batch_size):
        stop = min(start + batch_size, batch)
        with cache._prefill_batch(start, stop, batch_size=batch) as chunk:
            logits = model(input_ids[start:stop], past_key_values=chunk, use_cache=True, logits_to_keep=1).logits
            if output is None:
                output = logits.new_empty((batch, *logits.shape[1:]))
            output[start:stop].copy_(logits)
            del logits
    return output


class DecodeGraph:
    """Capture one token; replay returns borrowed logits overwritten by the next call.

    Reset/refill can reuse storage. Moving/replacing model parameters or cache
    buffers requires recapture. The caller owns sampling and conversation state.
    """

    @torch.inference_mode()
    def __init__(self, model, cache: DecoderCache):
        if model.training or cache.capacity is None or cache.position is None or not cache.position.is_cuda:
            raise ValueError("decode capture requires an eval model and a prefilled CUDA static cache")
        if not 0 < cache.get_seq_length() < cache.capacity:
            raise ValueError("prefill must leave room for a decoding token")
        if model.config.document_separator_token_id is not None or cache.document_ids is not None:
            raise ValueError("decode capture requires single-document inference")
        self.model, self.cache = model, cache
        self.input_ids = torch.zeros_like(cache.position)
        self._buffers = tuple(
            t for layer in cache.layers for t in (layer.keys, layer.values, layer.cumulative_length)
        ) + (cache.position,)
        position = cache._snapshot_position()

        def forward():
            return model(self.input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits

        stream = torch.cuda.Stream(device=cache.position.device)
        stream.wait_stream(torch.cuda.current_stream())
        try:
            try:
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        cache._restore_position(position)
                        forward()
                    cache._restore_position(position)
            finally:
                # Order restoration/capture after queued warmup writes, even on failure.
                torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.logits = forward()
        finally:
            cache._restore_position(position)

    @torch.inference_mode()
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        if (
            input_ids.shape != self.input_ids.shape
            or input_ids.device != self.input_ids.device
            or input_ids.dtype != torch.long
        ):
            raise ValueError("decode input must match captured (B,1) int64 shape/device")
        if not 0 < self.cache.seen_tokens < self.cache.capacity:
            raise ValueError("prefill must leave room within the captured cache capacity")
        # Reorder/reset must not silently leave replay pointing at obsolete storage.
        buffers = tuple(
            t for layer in self.cache.layers for t in (layer.keys, layer.values, layer.cumulative_length)
        ) + (self.cache.position,)
        if any(a is not b for a, b in zip(self._buffers, buffers, strict=True)):
            raise ValueError("cache storage changed; recapture decoding")
        self.input_ids.copy_(input_ids)
        self.graph.replay()
        self.cache.seen_tokens += 1
        return self.logits

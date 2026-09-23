"""Generate from a local HF checkpoint with bounded prefill and CUDA graph decoding.

python examples/generation.py checkpoint --tokenizer tokenizer --prompt "Hello"
Requires CUDA and FlashAttention. Sampling here is greedy and fixed-length;
use model.generate for Hugging Face's EOS, sampling, and beam-search policies.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from white_matter.models import register_models
from white_matter.models.generation import DecodeGraph, prefill


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", help="Defaults to the checkpoint directory")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    register_models()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).cuda().eval()
    model.config.document_separator_token_id = None
    if hasattr(model.config, "prefill_mode"):
        model.config.prefill_mode = "cyclic"
    inputs = tokenizer(args.prompt, return_tensors="pt").input_ids.cuda()
    cache = model.allocate_inference_cache(inputs.shape[1] + args.tokens)
    logits = prefill(model, inputs, cache)
    tokens = [logits[:, -1].argmax(-1, keepdim=True)]
    del logits
    if args.tokens > 1:
        # Preserve explicit BF16 rounding and cache read/write order.
        model.compile(options={"emulate_precision_casts": True, "reorder_for_locality": False})
        graph = DecodeGraph(model, cache)
        for _ in range(1, args.tokens):
            # Graph logits are borrowed; consume them before the next replay.
            tokens.append(graph(tokens[-1])[:, -1].argmax(-1, keepdim=True))
    print(tokenizer.decode(torch.cat((inputs, *tokens), dim=1)[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

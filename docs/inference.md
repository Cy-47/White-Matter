# Inference and benchmarking

WhiteMatter, Vanilla, and LCKV support Hugging Face `forward` and `generate` with dynamic or static caches. FusedKV currently supports uncached execution only. For CUDA inference, install the `gpu` extra and load the model with BF16 weights and FlashAttention:

```python
import torch
from transformers import AutoModelForCausalLM
from white_matter.models import register_models

register_models()
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/checkpoint",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).cuda().eval()
model.config.document_separator_token_id = None
model.config.prefill_mode = "cyclic"  # WhiteMatter only

with torch.no_grad():
    tokens = model.generate(input_ids.cuda(), max_new_tokens=32, use_cache=True)
```

WhiteMatter's `prefill_mode="cyclic"` uses the configured finite number of cyclic passes for the prompt; `"autoregressive"` processes it sequentially. Both modes continue autoregressively during cached decoding. Set the mode explicitly when a WhiteMatter checkpoint does not specify one. LCKV uses Jacobi prefill by default and also accepts `"autoregressive"` prefill.

Dynamic caches support padding, document boundaries, and beam reordering. Static caches are for unpadded, single-document inference and require a capacity. A supplied cache persists across calls until reset; a generation call does not reset it. `document_separator_token_id=None` treats the prompt as one document even if it contains EOS. Explicit document IDs can instead define packed-document boundaries.

For fixed-batch CUDA decoding with bounded prefill memory, see [examples/generation.py](../examples/generation.py). It shows `allocate_inference_cache`, `prefill`, and `DecodeGraph`. The example uses greedy decoding; `model.generate` provides Hugging Face sampling and stopping policies. Multi-turn cyclic suffix prefill, cache rollback, and paging are not supported.

## Measure throughput and memory

Install the benchmark dependencies and run the generation benchmark on local Hugging Face exports:

```bash
pip install -e '.[benchmarks]'
python -m benchmarks.generation --models /path/to/wm /path/to/vanilla \
  --prompt-lengths 2048 --batch-sizes 1 2 4 8 16 32 64 \
  --tokens 129 --memory-budget-gib 40 --output outputs/benchmarks
```

Use `--phase prefill` or `--phase decode` to measure each stage separately; the default `end-to-end` phase includes both. `--tokens 129` means one token selected during prefill and 128 timed decode steps. `--prefill-batch-size` bounds prefill workspace independently of the resident decode batch; `0` uses the full batch.

The benchmark uses seeded synthetic prompts and random weights when given recipe files. Checkpoint inputs measure local model exports. Runs include embedding, output projection, and token selection, but exclude loading, compilation, cache allocation, and warmup from steady-state timing. Decode-only runs copy one prompt into independent cache rows before timing. Results record the workload, raw samples, model identity, software and GPU versions, and memory measurements. Capacity searches apply the same memory budget and headroom to each model; use an otherwise idle GPU for comparable results.

Each run writes `run.json`, case records, and `summary.json` below the requested output directory. Export a summary as CSV with:

```bash
python -m benchmarks.report outputs/benchmarks/<run-id> --csv outputs/table.csv
```

The [benchmark guide](../benchmarks/README.md) documents training measurements, capacity search, profiling, and result files.

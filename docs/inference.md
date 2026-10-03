# Inference and benchmarking

WhiteMatter, Vanilla, LCKV, and Feedback Transformer support Hugging Face `forward` and `generate` with dynamic or static caches. FusedKV supports `generate(use_cache=False)`, recomputing the full prefix at each step. The evaluation harness selects this automatically and uses the same Hugging Face generation controls for every family. For CUDA inference, install the model and GPU dependencies, then load the model with BF16 weights and PyTorch SDPA:

```bash
pip install -e '.[tilelang]'
```

See the [installation guide](installation.md) for CUDA environment requirements.

```python
import torch
from transformers import AutoModelForCausalLM
from white_matter.models import register_models

register_models()
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/checkpoint",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).cuda().eval()
model.config.document_separator_token_id = None
model.config.prefill_mode = "cyclic"  # WhiteMatter only

with torch.no_grad():
    tokens = model.generate(input_ids.cuda(), max_new_tokens=32, use_cache=True)
```

WhiteMatter's `prefill_mode="cyclic"` uses cyclic passes for the prompt, while `"jacobi"` uses token-parallel Jacobi passes and `"autoregressive"` processes it sequentially. All three modes continue autoregressively during cached decoding. Set the mode explicitly when a WhiteMatter checkpoint does not specify one. LCKV uses Jacobi prefill by default and also accepts `"autoregressive"` prefill.

Strict-causal readers share the [backend policy](installation.md#backend-selection).
See [compilation policy](installation.md#compilation-policy) for model compilation
and autoregressive loop behavior.

Dynamic caches support padding, document boundaries, and beam reordering. Static caches are for unpadded, single-document inference and require a capacity. A supplied cache persists across calls until reset; a generation call does not reset it. `document_separator_token_id=None` treats the prompt as one document even if it contains EOS. Explicit document IDs can instead define packed-document boundaries.

Feedback Transformer currently requires unpadded inputs and uses exact sequential
prefill. Its benchmark recipe measures an architecture control with random weights.
Its uncached forward supports differentiation, but the shared training CLI's
autoregressive checkpointing is not implemented for this family.

The `DecodeGraph` helper accepts SDPA and external FlashAttention. Capture
requires a prefilled CUDA static cache and single-document inference.

For fixed-batch CUDA decoding with bounded prefill memory, see [examples/generation.py](../examples/generation.py). It shows `allocate_inference_cache`, `prefill`, and `DecodeGraph`. The example uses greedy decoding; `model.generate` provides Hugging Face sampling and stopping policies. Multi-turn cyclic suffix prefill, cache rollback, and paging are not supported.

## Measure throughput and memory

Install the benchmark dependencies and run the generation benchmark on local Hugging Face exports:

```bash
pip install -e '.[research]'
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

### Strict-past attention and cached Jacobi prefill

`use_dummy_token=False` is the WhiteMatter default. The KV cache then contains
only real tokens; a query with no earlier tokens receives zero attention output.
Set `use_dummy_token=True` for the learned leading dummy used by older checkpoints.

With `prefill_mode="jacobi"`, a multi-token call with an existing cache runs Jacobi
on the new tokens against the frozen cached prefix. A single-token continuation
uses exact autoregressive decode. Each new query at offset `i` can read the `P`
cached real tokens and the first `i` new tokens from the previous iteration.
Only new K/V is appended after prefill; the existing prefix is preserved.
Document boundaries still isolate independent prompts. Reset the cache when
starting an unrelated prompt unless explicit document IDs establish that boundary.

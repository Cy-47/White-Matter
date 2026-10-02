# WhiteMatter

Implementation of [WhiteMatter: All-to-All Cross-Layer Connections via KV Source Mixing](https://arxiv.org/abs/2608.18486). WhiteMatter lets each Transformer layer attend to a learned mixture of past-token states from every depth. Multiple layers can share a KV channel, reducing cache size. Cyclic iteration makes the feedback connections parallelizable during training and prefill.

This repository provides the WhiteMatter model, its cyclic attention operator, training and evaluation commands, and the configurations used for the paper. Vanilla, LCKV, and FusedKV implementations are included as comparison models.

## Installation

Python 3.11 or newer is required. From this repository:

```bash
pip install -e '.[models,training,data]'
```

For CUDA kernels and benchmarks, install the `gpu` and `benchmarks` extras as needed. The tested versions of PyTorch, Transformers, FlashAttention, and TileLang are specified in [pyproject.toml](pyproject.toml). The base package requires only PyTorch; model classes require the `models` extra.

## Use a model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from white_matter.models import register_models

register_models()
model = AutoModelForCausalLM.from_pretrained("/path/to/checkpoint")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
inputs = tokenizer("Hello", return_tensors="pt")
outputs = model(**inputs)
```

The model families are `white_matter`, `vanilla`, `lckv`, `fusedkv`, and `feedback_transformer`. WhiteMatter supports cyclic, Jacobi, or exact autoregressive execution. For generation with a WhiteMatter checkpoint, choose `prefill_mode="cyclic"` or `"jacobi"` for finite-pass prefill, or `"autoregressive"` for exact sequential prefill. See [inference](docs/inference.md) and the [generation example](examples/generation.py) for cache use and supported modes.

WhiteMatter defaults to `use_dummy_token=False`: real tokens attend only to
strictly earlier tokens, and the first token of each document receives zero
attention output. Set `use_dummy_token=True` to retain the learned dummy slot.
Older dummy-trained checkpoints without this configuration field must be loaded
with `use_dummy_token=True`; paper recipes and the checkpoint importer specify it.

The shared `white_matter.ops.strict_causal_attention` operator supports Jacobi,
prefill with a cached prefix, and decode. It accepts already-projected Q/K/V in
`(B,H,T,D)` layout and returns `(B,T,H,D)`. `query_start` and `kv_lengths` count
real tokens, excluding the optional leading dummy slot. Cache writes remain the
caller's responsibility. For packed documents, prepare and reuse
`prepare_strict_causal_metadata(...)` outside the layer/iteration loop.

The cyclic attention operator also works independently of the model classes:

```python
from white_matter.ops import cyclic_attention

output = cyclic_attention(query, key, value, query_stride=4)
```

Q/K/V have shape `(batch, heads, sequence, head_dim)`. The portable PyTorch backend is the default; the optional TileLang backend requires a compatible CUDA installation. See the [operator guide](docs/cyclic_attention.md) for the shape and masking contract, backend behavior, and contribution boundaries, and the [operator example](examples/cyclic_attention.py) for packed-document metadata and gradients.

## Reproduce the experiments

Prepare the FineWeb-Edu token cache, then train from a paper configuration:

```bash
python scripts/prepare_fineweb_edu.py --reproduce-paper-order
torchrun --standalone --nproc-per-node=8 -m training.train \
  --recipe recipes/paper/white_matter_k8.yaml \
  --data-dir data/cache_fineweb_edu_20b_len2048 \
  --output-dir outputs/white_matter_k8
```

The configurations in [recipes/paper](recipes/paper) cover the reported WhiteMatter models and comparison models. [The convergence study](studies/prefill_convergence/README.md) contains the exact autoregressive control and its quality/timing workflow. Training writes a resume checkpoint and a Hugging Face export for evaluation. The [Slurm examples](slurm/README.md) show cluster launches.

The [studies index](studies/README.md) contains the Figure 7a schedule matrix,
Figure 7b rank and connectivity ablations, and the shared-mixture ablation.
Each study keeps its recipes, evaluation protocol, and analysis alongside any
study-specific model code. See [reproduction tools](docs/reproduction.md) for
repository layout, cache requirements, and checkpoint conversion.

Evaluate held-out perplexity or run the paper's zero-shot task suite:

```bash
pip install 'lm-eval @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@ddd67220430a2470529f25fd5c05a576ca1057a0'
```

```bash
python -m evals.heldout --model outputs/white_matter_k8/final \
  --data-dir /path/to/data --output outputs/white_matter_k8/heldout.json
python -m evals.lm_eval --model outputs/white_matter_k8/final \
  --output outputs/white_matter_k8/lm_eval.json
```

The pinned [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
release is v0.4.13. Evaluation outputs record the installed harness revision,
task versions, and task configurations. Compare checkpoints using the same
evaluation settings. SQuAD-Completion v1 normalizes whitespace in prompts and
answers; its scores are not comparable to v0. See [reproduction tools](docs/reproduction.md)
for additional evaluation tasks.

## Benchmarks and validation

The [benchmark guide](benchmarks/README.md) covers training throughput, prefill, cached decoding, memory capacity, and cyclic attention. Benchmark inputs use synthetic data; quality measurements use the evaluation commands above.

```bash
python -m pytest -m 'not gpu'
ruff check src training evals examples benchmarks tests scripts studies
ruff format --check src training evals examples benchmarks tests scripts studies
```

GPU tests exercise the optional kernels and compiled execution. Run them on compatible hardware with the `gpu` extra installed.

## License

MIT. See [NOTICE](NOTICE) for attribution and licenses of derived code.

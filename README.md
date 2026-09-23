# WhiteMatter

Implementation of [WhiteMatter: All-to-All Cross-Layer Connections via KV Source Mixing](https://arxiv.org/abs/2608.18486). WhiteMatter lets each Transformer layer attend to a learned mixture of past-token states from every depth. Multiple layers can share a KV channel, reducing cache size. Cyclic iteration makes the feedback connections parallelizable during training and prefill.

This repository provides the WhiteMatter model, its cyclic attention operator, training and evaluation commands, and the configurations used for the paper. Vanilla, LCKV, and FusedKV implementations are included as comparison models. The paper reports lower held-out perplexity with half the KV cache at both evaluated scales; iterative prefill costs more than standard Transformer prefill.

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

The model families are `white_matter`, `vanilla`, `lckv`, and `fusedkv`. WhiteMatter supports cyclic or exact autoregressive execution. For generation with a WhiteMatter checkpoint, choose `prefill_mode="cyclic"` for finite-pass prefill or `"autoregressive"` for exact sequential prefill. See [inference](docs/inference.md) and the [generation example](examples/generation.py) for cache use and supported modes.

The cyclic attention operator also works independently of the model classes:

```python
from white_matter.ops import cyclic_attention

output = cyclic_attention(query, key, value, query_stride=4)
```

Q/K/V have shape `(batch, heads, sequence, head_dim)`. The portable PyTorch backend is the default; the optional TileLang backend requires a compatible CUDA installation. See the [operator example](examples/cyclic_attention.py) for packed-document metadata and gradients.

## Reproduce the experiments

Prepare the FineWeb-Edu token cache, then train from a paper configuration:

```bash
python scripts/prepare_fineweb_edu.py --output /path/to/data --workers 8
torchrun --standalone --nproc-per-node=8 -m training.train \
  --recipe recipes/paper/white_matter_k8.yaml \
  --data-dir /path/to/data --output-dir outputs/white_matter_k8
```

The configurations in [recipes/paper](recipes/paper) cover the reported WhiteMatter models and comparison models. [recipes/analysis](recipes/analysis) contains the exact autoregressive convergence control. Training writes a resume checkpoint and a Hugging Face export for evaluation. The [Slurm examples](slurm/README.md) show cluster launches.

Evaluate held-out perplexity or run the paper's zero-shot task suite:

```bash
pip install 'lm-eval @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@9b2b9280330a3a5b20953346c8b51e23c4c8c4e2'
```

```bash
python -m evals.heldout --model outputs/white_matter_k8/final \
  --data-dir /path/to/data --output outputs/white_matter_k8/heldout.json
python -m evals.lm_eval --model outputs/white_matter_k8/final \
  --output outputs/white_matter_k8/lm_eval.json
```

The pinned [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) revision matches the paper's evaluation setup.

## Benchmarks and validation

The [benchmark guide](benchmarks/README.md) covers training throughput, prefill, cached decoding, memory capacity, and cyclic attention. Benchmark inputs use synthetic data; quality measurements use the evaluation commands above.

```bash
python -m pytest -m 'not gpu'
ruff check src training evals examples benchmarks tests scripts
```

GPU tests exercise the optional kernels and compiled execution. Run them on compatible hardware with the `gpu` extra installed.

## License

MIT. See [NOTICE](NOTICE) for attribution and licenses of derived code.

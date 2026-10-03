# Installation

Use Python 3.11 or newer. Run these commands from the repository root. The wheel
installs the reusable `white_matter` package; data preparation, training, and
benchmark commands run from the checkout.

## Install options

| Setup | Command | Includes |
| --- | --- | --- |
| Base | `pip install -e .` | PyTorch, Transformers, Safetensors, HF Hub, operators, model classes, and our kernel source |
| TileLang | `pip install -e '.[tilelang]'` | Base plus TileLang and compatible TVM FFI |
| Research | `pip install -e '.[research]'` | TileLang and development tools plus NumPy, PyYAML, Datasets, Gigatoken, Matplotlib, and NVIDIA monitoring |
| Development | `pip install -e '.[dev]'` | Base plus Pytest, Ruff, mypy, and build |
| External FlashAttention (optional) | `pip install -e '.[flash-attn]'` | Base plus external FlashAttention for explicitly selected external kernels |

The authoritative versions are in [pyproject.toml](../pyproject.toml).
Combine extras as needed: paper recipes and external-backend benchmarks use
`pip install -e '.[research,flash-attn]'`. PyTorch's built-in FlashAttention
does not require the external package.

## CUDA setup

Base code remains usable through PyTorch implementations without TileLang or
external FlashAttention. Our accelerated kernel source ships with the package;
TileLang execution requires `tilelang`. The `research` extra includes CUDA
extensions and is intended for a compatible CUDA research environment.

Start with CUDA-enabled PyTorch satisfying the version range in `pyproject.toml`,
then install `.[tilelang]` or `.[research]`. Installing extras does not select the
correct PyTorch wheel index or install a GPU driver for you.

TileLang compiles kernels on first use. Make the CUDA toolkit available in the
job environment, with `nvcc` on `PATH` or the toolkit root set as `CUDA_HOME`.

If you need external FlashAttention, its source build requires a compatible CUDA
toolkit and build tools. See the [upstream installation instructions](https://github.com/Dao-AILab/flash-attention#installation-and-features).
When building against the installed PyTorch environment:

```bash
pip install packaging psutil ninja
pip install 'flash-attn==2.8.3' --no-build-isolation
pip install -e '.[research,flash-attn]'
```

## Backend selection

Cache preparation defaults to `--tokenizer-backend auto`, selecting Gigatoken
when installed. Research installs include it. If it is missing, the builder
warns before using slower Hugging Face tokenization. Explicit `gigatoken`
selection raises if unavailable; explicit `hf` selection does not warn.

On CUDA, automatic strict-causal attention uses efficient SDPA for differentiable
single-query reads with masks or per-row lengths. This keeps lengths on the GPU during CUDA graph
capture and avoids rebuilding document schedules in each layer. Other automatic
CUDA reads select standalone FlashAttention when installed, or PyTorch's native
FlashAttention otherwise.
Unsupported CUDA inputs raise; CPU automatic dispatch uses the reference.
`backend="torch_flash"` requires native FlashAttention, `backend="flash_attention_2"`
requires the standalone package, and `backend="reference"` selects the portable
implementation explicitly. FP32 evaluation selects reference attention throughout.
Packed documents use variable-length FlashAttention. Single-token inference uses
GPU-resident lengths and packs visible slots when document masks require it.

Cyclic attention currently selects its backend explicitly and defaults to the
PyTorch reference. Installing `tilelang` supplies TileLang but does not change the
standalone operator's default. See [inference](inference.md) for model dispatch,
packed-sequence performance, and cached-decoding limitations.

## Compilation policy

Runnable training, evaluation, examples, and runtime benchmarks compile tensor
work by default on their supported devices. Training compiles the loss workload,
including embeddings. Evaluation compiles the complete model and the backbone
entry point used by chunked likelihood scoring. Autoregressive token loops stay
in Python; their child tensor calls compile under the enclosing workload. Use
the default `fullgraph=False` for these models. WhiteMatter Jacobi inference and
cyclic training passes, plus LCKV inference passes, use nested regions to reuse
a pass graph. Jacobi training traces into the enclosing graph to preserve BF16
gradients on PyTorch 2.12.
Metadata preparation, observation callbacks, and I/O remain outside tensor graphs.
The schedule study keeps its outer pass loop in Python and compiles each reusable
pass. Changing the pass count reuses the same tensor graphs.

Evaluation and examples accept `--no-compile`; model benchmarks accept
`--no-compiled`. These options warn and force eager execution for the run,
including internal compiled helpers. Compiled workflows raise on compiler errors
and recompile-limit exhaustion instead of silently falling back.

FLOP accounting uses eager dispatch to count unfused operations and warns about
this requirement. Packed autoregressive training retains its established
`aot_eager` boundary for gradient consistency; this captures autograd graphs
while retaining ATen kernels.

Python API callers control whole-model compilation with `model.compile()` or
`torch.compile`. Nested regions are captured by the enclosing compile call;
kernel JIT compilation remains backend-specific. For an explicit eager diagnostic scope, use
`white_matter.compilation.execution_policy(False)`; it warns and restores the
previous compiler settings on exit.

## Reproducing experiments

Follow [reproduction tools](reproduction.md) for the additional pinned Cut
Cross-Entropy and evaluation-harness revisions. Those packages are not included
in `research`. Package compatibility ranges are not an environment lock;
preserve resolved versions, source revisions, and the GPU software environment
alongside experimental results.

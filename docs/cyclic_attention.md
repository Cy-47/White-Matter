# Cyclic attention

`white_matter.ops.cyclic_attention` is a standalone differentiable operator.
It does not require a model, training loop, or WhiteMatter KV cache. Install the
base package for the PyTorch reference, or the `gpu` extra for TileLang.

```python
from white_matter.ops import cyclic_attention

output = cyclic_attention(
    query, key, value,
    query_stride=4,
    query_offset=1,
    backend="reference",  # Use "tilelang" for the CUDA BF16 implementation.
)
```

## Tensor and schedule contract

| Input | Shape / requirement |
| --- | --- |
| Query | `(B, HQ, Q, D)` |
| Key and value | Both `(B, HKV, K, D)` |
| Heads | `HQ` must be a positive multiple of `HKV` (grouped-query attention) |
| Device and dtype | All three inputs share device and floating-point dtype supported by the selected backend |
| Dimensions | All dimensions are positive |
| `query_stride` | Positive Python integer |
| `query_offset` | Python integer in `[0, query_stride)` |

Query row `i` attends to key slots `j <= query_offset + query_stride * i`.
The last query slot must be within the key sequence:
`query_offset + query_stride * (Q - 1) < K`.
For example, stride four and offset one give query slots `1, 5, 9, ...`.
Keys remain densely indexed; the operator does not subsample them. Forward and
query-gradient kernels bound causal work by the last real query, avoiding extra
key tiles implied only by padded query rows. Key/value-gradient kernels skip
query loops for key tiles beyond every query's causal reach and return zero
gradients for those tiles.

The scale is `1 / sqrt(D)`, with no dropout. The output has the query's shape,
dtype, and device. Its storage strides are backend-specific; use `reshape` or
an explicit contiguous conversion where downstream code requires a layout.
Inputs are not mutated. Both backends provide first-order gradients for Q/K/V;
metadata and schedule arguments are not differentiable. Higher-order gradients
are not part of the TileLang contract.

## Backends and layouts

`backend="reference"` is the default. It uses PyTorch's math SDPA backend and
materializes a dense attention mask. It is intended for checking semantics,
small experiments, and environments without the optional GPU dependency. Its
memory cost grows with `Q * K`.

`backend="tilelang"` requires CUDA, BF16 Q/K/V, and head dimension 64, 96, or
128. It is selected explicitly: unsupported inputs raise an error rather than
silently switching backends. Architecture-specific tuning currently includes
A6000 and A100 configurations; other GPU targets use the fallback tile table
and should be benchmarked before drawing performance conclusions.

Strided tensor views are accepted. Plain forward attention reads native K/V
views without copying when each pointer is 16-byte aligned, its outer strides
are multiples of eight BF16 elements, and its innermost stride is one. This
includes common cache views with capacity padding. Other K/V layouts are copied
to aligned contiguous storage before dispatch. Query storage is normalized
internally; document forward and backward also normalize contiguous storage and
alignment where needed. Passing views is therefore semantically supported but
does not imply a copy-free execution path.

## Packed documents

Without metadata, only the cyclic causal mask applies. With metadata, real keys
must also belong to the same document as the query. Key slot zero is a shared
initial source and remains visible to every query.

Prepare metadata with `prepare_cyclic_attention_metadata(query_ids, key_ids)`:

- Both ID tensors are integer matrices on the Q/K/V device, with shapes `(B, Q)`
  and `(B, K)` respectively, with dtype `int32` or `int64`.
- IDs are nondecreasing within each row. Query IDs are nonnegative. All IDs
  must fit int32 (maximum `2147483647`), including when inputs use int64.
- Each key row starts with exactly one dummy ID `-1`; all remaining IDs are
  nonnegative.
- Document numbering and the mapping from token positions to key slots belong
  to the caller. The helper does not create or initialize dummy K/V vectors.

The returned object contains IDs and derived bounds. Sorted document bounds let
the forward kernel skip tiles belonging entirely to earlier documents while
retaining the shared dummy key; backward kernels also use bounds to restrict
their work. Build metadata outside compiled loops, rebuild it when document
assignments change, and do not mutate it between forward and backward. Pass it as `metadata=metadata` to either backend.
See the [standalone example](../examples/cyclic_attention.py) for dummy-shifted
keys, grouped-query heads, and backward execution:

```bash
python examples/cyclic_attention.py
python examples/cyclic_attention.py --backend tilelang
```

## Compilation and contribution boundaries

The six TileLang attention kernels (plain/document forward and both gradient
families) use the following specialization policy:

| Parameter | Treatment |
| --- | --- |
| Batch size, query length, KV length | Runtime dimensions |
| Q/K/V values and document IDs/bounds | Runtime tensor inputs |
| Query offset | Runtime device scalar |
| Plain-forward K/V batch, head, sequence strides | Runtime strides, including cache capacity padding |
| Query and KV head counts, head dimension | Compile-time constants |
| Cyclic query stride | Compile-time constant |
| Plain versus document mode; forward, query gradient, KV gradient | Separate kernel families |
| Resolved tile sizes and pipeline stages | Compile-time constants |
| GPU compute capability | Explicit compilation target and cache-key component |

Plain forward uses a single layout contract with unit innermost stride and
aligned storage to permit vectorized copies. Input normalization handles layouts
outside that contract without adding kernel variants. Exact sequence lengths
and outer stride values remain runtime inputs.

BF16 is fixed for these kernels; dtype is not a selectable specialization axis.
Changing only runtime dimensions or metadata reuses the same cached TileLang
kernel while its structural parameters remain fixed. Tile configurations are
resolved before cache lookup, so requests that select identical tiles share an
entry. The in-process cache holds up to 768 entries and can evict older kernels.

TileLang compilation is lazy. Measure cold compilation separately from warmed
execution; a Python cache miss may still reuse TileLang's persistent disk cache.
This policy describes kernel reuse, not a measured performance guarantee.
Kernel reuse and PyTorch graph reuse are separate: changing Python schedule
arguments or graph guards can cause `torch.compile` to retrace independently of
the underlying attention kernels. Warm the intended operator configurations
before CUDA graph capture. Captured CUDA graphs remain tied to capture-time
shapes and buffers; dynamic TileLang kernel reuse does not make graph replay
shape-dynamic.

The implementation is organized into four layers:

| Location | Responsibility |
| --- | --- |
| `src/white_matter/ops/cyclic_attention/functional.py` | Public validation and backend selection |
| `reference.py`, `metadata.py` in the same directory | Reference semantics and document bounds |
| `_tilelang/registration.py` | Autograd, layout normalization, dispatch, and caching |
| `_tilelang/*forward.py`, `_tilelang/*backward*.py` | GPU programs |

When extending semantics, update the reference and public contract first, then
compare optimized outputs and gradients against the reference. Validate partial
tiles, grouped-query heads, offsets, and document boundaries. A new tuning entry
should include warm forward/backward measurements and compilation costs on its
target GPU. Avoid specializing on values that only change the amount of work
unless measurements justify the additional variants.

The [benchmark guide](../benchmarks/README.md) describes the existing attention
benchmark. GPU correctness tests live in `tests/gpu/test_cyclic_attention.py`.

### Strict-past keys without a dummy

Set `strict_past=True` to read only real keys whose position is strictly less
than `query_offset + query_stride * query_index`. Empty attention rows return
zero. For packed inputs, call `prepare_cyclic_attention_metadata` with
`use_dummy_token=False` and real-token document labels without a dummy sentinel.
The default `strict_past=False` preserves the inclusive slot bound used by the
dummy-shifted layout. Both reference and TileLang backends support these modes.

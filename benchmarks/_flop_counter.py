"""Algorithmic attention FLOPs: causal pairs, 4 forward and 10 backward ops/dim.

Custom formulas bind dispatcher schemas by name, rejecting missing inputs.
Measurements require unsegmented sequences; document kernels are rejected.
"""

import torch
from torch.utils.flop_counter import FlopCounterMode


def causal_pairs(queries, keys, *, stride=1, offset=None):
    offset = keys - queries if offset is None else offset
    return sum(max(0, min(keys, offset + i * stride + 1)) for i in range(queries))


def attention_formula(op, *, layout, backward=False):
    schema = op.default._schema
    upper_left = schema.name in {
        "aten::_scaled_dot_product_efficient_attention",
        "aten::_scaled_dot_product_efficient_attention_backward",
    }
    names = {a.name for a in schema.arguments}
    qname, kname = (
        ("Q", "K") if layout == "cyclic" else ("query", "key") if layout in {"sdpa", "native_varlen"} else ("q", "k")
    )
    if not {qname, kname} <= names:
        raise RuntimeError(f"unsupported attention schema: {schema}")

    def formula(*args, out_val=None, **kwargs):
        values = {a.name: a.default_value for a in schema.arguments}
        values.update(zip((a.name for a in schema.arguments), args, strict=False))
        values.update(kwargs)
        q, k = values[qname], values[kname]
        multiplier = 10 if backward else 4
        if layout == "cyclic":
            batch, heads, queries, dim = q.shape
            pairs = causal_pairs(queries, k.shape[2], stride=values["K_stride"], offset=values["residue"])
        elif layout == "sdpa":
            batch, heads, queries, dim = q.shape
            keys = k.shape[2]
            bias = values.get("attn_bias")
            if bias is not None:
                keep = bias if bias.dtype == torch.bool else bias.isfinite()
                if values["is_causal"]:
                    keep = keep & torch.ones(queries, keys, device=q.device, dtype=torch.bool).tril()
                pairs = keep.expand(batch, heads, queries, keys).sum().item()
                return multiplier * dim * pairs
            pairs = (
                causal_pairs(queries, keys, offset=0 if upper_left else None)
                if values.get("is_causal", values.get("causal", False))
                else queries * keys
            )
        elif layout == "native_varlen":
            cu_q = values.get("cu_seq_q", values.get("cum_seq_q"))
            cu_k = values.get("cu_seq_k", values.get("cum_seq_k"))
            if cu_q is None:
                batch, queries, heads, dim = q.shape
                keys = k.shape[1]
                pairs = causal_pairs(queries, keys) if values["is_causal"] else queries * keys
                return multiplier * batch * heads * dim * pairs
            lengths_q = cu_q.diff().tolist()
            lengths_k = values.get("seqused_k")
            lengths_k = (cu_k.diff() if lengths_k is None else lengths_k).tolist()
            pairs = sum(
                causal_pairs(nq, nk) if values["is_causal"] else nq * nk
                for nq, nk in zip(lengths_q, lengths_k, strict=True)
            )
            return multiplier * q.shape[1] * q.shape[2] * pairs
        elif layout == "varlen":
            lengths_q = values["cu_seqlens_q"].diff().tolist()
            lengths_k = values["cu_seqlens_k"].diff().tolist()
            if len(set(lengths_q)) != 1 or len(set(lengths_k)) != 1:
                raise ValueError("FLOP protocol requires equal-length unsegmented rows")
            batch, heads, dim = len(lengths_q), q.shape[1], q.shape[2]
            nq, nk = lengths_q[0], lengths_k[0]
            pairs = causal_pairs(nq, nk) if values["causal"] else nq * nk
        else:
            batch, queries, heads, dim = q.shape
            keys = k.shape[1]
            pairs = causal_pairs(queries, keys) if values["causal"] else queries * keys
        return multiplier * batch * heads * dim * pairs

    formula._get_raw = True
    return formula


def make_counter(attention_backend="sdpa"):
    from white_matter.ops.cyclic_attention._tilelang import registration  # noqa: F401
    from white_matter.ops.flash_attention import flash_attention_decode  # noqa: F401
    from white_matter.ops.strict_causal_attention import _standalone_flash_available

    mapping = {}
    for suffix, backward in [("fwd", False), ("bwd", True)]:
        op = getattr(torch.ops.white_matter, f"cyclic_attn_{suffix}")
        mapping[op] = attention_formula(op, layout="cyclic", backward=backward)
    if attention_backend == "flash_attention_2" or _standalone_flash_available():
        from flash_attn import flash_attn_interface  # noqa: F401

        for prefix, layout in [("", "dense"), ("varlen_", "varlen")]:
            for suffix, backward in [("forward", False), ("backward", True)]:
                op = getattr(torch.ops.flash_attn, f"_flash_attn_{prefix}{suffix}")
                mapping[op] = attention_formula(op, layout=layout, backward=backward)
    for kernel in ("flash", "efficient"):
        for suffix, backward in [("", False), ("_backward", True)]:
            op = getattr(torch.ops.aten, f"_scaled_dot_product_{kernel}_attention{suffix}")
            mapping[op] = attention_formula(op, layout="sdpa", backward=backward)

    for op, backward in (
        (torch.ops.aten._flash_attention_forward, False),
        (torch.ops.aten._flash_attention_backward, True),
    ):
        mapping[op] = attention_formula(op, layout="native_varlen", backward=backward)

    op = torch.ops.white_matter.torch_flash_dense_forward
    mapping[op] = attention_formula(op, layout="sdpa")

    def decode(query, key, value, lengths, scale, causal, num_splits=0, *, out_val=None):
        del value, scale, num_splits, out_val
        pairs = sum(causal_pairs(query.shape[1], n) if causal else query.shape[1] * n for n in lengths.tolist())
        return 4 * query.shape[2] * query.shape[3] * pairs

    decode._get_raw = True
    mapping[torch.ops.white_matter.flash_attention_decode] = decode
    return FlopCounterMode(display=False, custom_mapping=mapping)


def single_pass_formula(config, *, regime, length, context):
    """Independent exact matrix/attention/fusion count for vanilla and FusedKV."""
    d, ff, layers = config.hidden_size, config.intermediate_size, config.num_hidden_layers
    q, kv = config.num_attention_heads * config.head_dim, config.num_key_value_heads * config.head_dim
    common, projection = 4 * d * q + 6 * d * ff, 4 * d * kv
    sources = layers if config.model_type == "vanilla" else layers // 2
    upper = layers - sources
    fusion = 6 * upper * kv
    # RoPE computes its frequency outer product once, without gradients.
    rotary = config.head_dim * (1 if regime == "decode" else length)
    if regime == "train":
        return (
            rotary
            + length * (3 * (layers * common + sources * projection + fusion))
            + 14 * q * layers * causal_pairs(length, length)
        )
    if regime == "prefill":
        return (
            rotary
            + length * sources * (common + projection)
            + upper * common
            + 4 * q * (sources * causal_pairs(length, length) + upper * length)
            + length * fusion
        )
    return rotary + layers * common + sources * projection + 4 * q * layers * (context + 1) + fusion * (context + 1)

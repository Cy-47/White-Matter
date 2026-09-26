"""Import the paper's final research checkpoints into the HF evaluation layout.

Every persistent target tensor must come from a source tensor with the same
shape. Obsolete research-only buffers are explicitly allowlisted below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, GenerationConfig

from white_matter.models import register_models
from white_matter.models.fusedkv import FusedKVConfig
from white_matter.models.lckv import LCKVConfig
from white_matter.models.vanilla import VanillaConfig
from white_matter.models.white_matter import WhiteMatterConfig


SMALL = dict(hidden_size=512, intermediate_size=1536, num_hidden_layers=16,
             num_attention_heads=6, num_key_value_heads=3, head_dim=96,
             max_position_embeddings=4096, rope_theta=1_000_000.0,
             document_separator_token_id=151643, residual_dtype="fp32")
LARGE = dict(hidden_size=1792, intermediate_size=5376, num_hidden_layers=28,
             num_attention_heads=14, num_key_value_heads=7, head_dim=128,
             max_position_embeddings=32768, rope_theta=1_000_000.0,
             document_separator_token_id=151643, residual_dtype="bf16")
SPECS = {
    "fusedkv_16l": (FusedKVConfig, SMALL, {}),
    "lckv_w4": (LCKVConfig, SMALL, dict(num_passes=9, num_pre_layers=2, num_post_layers=2)),
    "lckv_w7": (LCKVConfig, SMALL, dict(num_passes=9, num_pre_layers=3, num_post_layers=4)),
    "vanilla_24l": (VanillaConfig, {**SMALL, "num_hidden_layers": 24}, {}),
    "vanilla_28l": (VanillaConfig, LARGE, {}),
    "white_matter_k14": (WhiteMatterConfig, LARGE,
                         dict(num_kv_channels=14, num_passes=3, prefill_mode="cyclic")),
}

def rename(name: str) -> str | None:
    name = name.replace("model.student.", "model.decoder.")
    if name.startswith("model.decoder."):
        name = name.replace("pre_mix_K_weight", "pre_mix_k_weight")
        name = name.replace("pre_mix_V_weight", "pre_mix_v_weight")
        name = name.replace("residual_post_mix.post_mix_K_gain", "post_mix.k_gain")
        name = name.replace("residual_post_mix.post_mix_V_gain", "post_mix.v_gain")
        name = name.replace("post_mix.post_mix_K_gain", "post_mix.k_gain")
        name = name.replace("post_mix.post_mix_V_gain", "post_mix.v_gain")
        name = name.replace("mixer.router_K.head.linear", "mixer.k_router.linear")
        name = name.replace("mixer.router_V.head.linear", "mixer.v_router.linear")
        name = name.replace("block.sequence_seed", "block.dummy_token")
        for old, new in (("fusedkv_k_old.weight", "k_bottom"),
                         ("fusedkv_k_new.weight", "k_middle"),
                         ("fusedkv_v_old.weight", "v_bottom"),
                         ("fusedkv_v_new.weight", "v_middle")):
            name = name.replace(old, new)
    if name.endswith(("kv_pool.beta_temperature", "kv_pool.beta_basis_idx")):
        return None
    return name


def source_tensors(raw: dict) -> dict[str, torch.Tensor]:
    if "model" in raw and "io_model" in raw and isinstance(raw["model"], dict):
        result = {"model.decoder." + k: v for k, v in raw["model"].items()}
        result["model.embed_tokens.weight"] = raw["io_model"]["embed_tokens.weight"]
        result["model.norm.weight"] = raw["io_model"]["final_norm.weight"]
        return result
    return dict(raw)


def convert(label: str, source_path: Path, dest: Path, *, config=None) -> None:
    if config is None:
        cls, dimensions, extra = SPECS[label]
        config = cls(**dimensions, **extra)
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite {dest}")
    raw = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    source = source_tensors(raw)
    register_models()
    with torch.device("meta"):
        template = AutoModelForCausalLM.from_config(config)
    expected = {k: tuple(v.shape) for k, v in template.state_dict().items()
                if k != "lm_head.weight"}
    mapped: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    for old_name, value in source.items():
        if old_name == "lm_head.weight":
            if not torch.equal(value, source["model.embed_tokens.weight"]):
                raise ValueError(f"{label}: LM head is not tied to input embeddings")
            ignored.append(old_name)
            continue
        new_name = rename(old_name)
        if new_name is None:
            ignored.append(old_name)
            continue
        if new_name.startswith("model.decoder.block.kv_pool.mixer.") and label.startswith("lckv_"):
            # The published LCKV uses a fixed last-layer source; these old
            # trainable router tensors were frozen and never consulted.
            ignored.append(old_name)
            continue
        if new_name == "model.decoder.block.dummy_token" and label.startswith("lckv_"):
            ignored.append(old_name)
            continue
        if new_name in mapped:
            raise ValueError(f"duplicate target key {new_name}")
        mapped[new_name] = value
    missing = sorted(set(expected) - set(mapped))
    unexpected = sorted(set(mapped) - set(expected))
    wrong = [(k, tuple(mapped[k].shape), expected[k]) for k in expected.keys() & mapped.keys()
             if tuple(mapped[k].shape) != expected[k]]
    if missing or unexpected or wrong:
        raise ValueError(f"{label}: missing={missing}, unexpected={unexpected}, wrong_shapes={wrong}")
    dest.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in mapped.items()}, str(dest / "model.safetensors"))
    config.architectures = [type(template).__name__]
    config.save_pretrained(dest)
    GenerationConfig.from_model_config(config).save_pretrained(dest)
    print(f"{label}: wrote {len(mapped)} tensors; ignored {ignored}; {dest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=sorted(SPECS))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted research checkpoint (.pt).")
    parser.add_argument("--output", type=Path, required=True, help="New Hugging Face model directory.")
    args = parser.parse_args()
    convert(args.architecture, args.checkpoint, args.output)

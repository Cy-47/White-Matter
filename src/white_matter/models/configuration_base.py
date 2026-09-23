"""Shared dimensions and numerical policy for the paper model families."""

import math

from transformers import PretrainedConfig


class DecoderConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        *,
        residual_dtype: str = "fp32",
        vocab_size: int = 151_936,
        hidden_size: int = 512,
        intermediate_size: int = 1_536,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 6,
        num_key_value_heads: int = 3,
        head_dim: int = 96,
        max_position_embeddings: int = 4_096,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1.0e-6,
        eos_token_id: int = 151_643,
        pad_token_id: int | None = None,
        document_separator_token_id: int | None = None,
        tie_word_embeddings: bool = True,
        **kwargs,
    ) -> None:
        if residual_dtype not in {"fp32", "bf16"}:
            raise ValueError("residual_dtype must be 'fp32' or 'bf16'")
        integer_fields = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "max_position_embeddings": max_position_embeddings,
        }
        for name, value in integer_fields.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if num_attention_heads % num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if type(rope_theta) is not float or not math.isfinite(rope_theta) or rope_theta <= 0:
            raise ValueError("rope_theta must be a positive float")
        if type(rms_norm_eps) is not float or not math.isfinite(rms_norm_eps) or rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be a positive float")
        if type(eos_token_id) is not int or not 0 <= eos_token_id < vocab_size:
            raise ValueError("eos_token_id must be an integer inside the vocabulary")
        if pad_token_id is not None and (type(pad_token_id) is not int or not 0 <= pad_token_id < vocab_size):
            raise ValueError("pad_token_id must be null or an integer inside the vocabulary")
        if tie_word_embeddings is not True:
            raise ValueError("the paper models require tied input and output embeddings")

        if document_separator_token_id is not None and (
            type(document_separator_token_id) is not int or not 0 <= document_separator_token_id < vocab_size
        ):
            raise ValueError("document_separator_token_id must be null or an integer inside the vocabulary")
        self.document_separator_token_id = document_separator_token_id
        self.residual_dtype = residual_dtype
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = 0.02
        self.hidden_act = "silu"
        self.attention_bias = False
        self.attention_dropout = 0.0
        self.use_cache = False

        super().__init__(
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

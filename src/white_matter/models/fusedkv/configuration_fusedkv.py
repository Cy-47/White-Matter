"""Hugging Face configuration for FusedKV."""

from ..configuration_base import DecoderConfig


class FusedKVConfig(DecoderConfig):
    model_type = "fusedkv"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.num_hidden_layers < 2 or self.num_hidden_layers % 2:
            raise ValueError("FusedKV requires an even num_hidden_layers >= 2")
        self.execution_mode = "single_pass"

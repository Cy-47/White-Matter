"""Hugging Face configuration for Vanilla."""

from ..configuration_base import DecoderConfig


class VanillaConfig(DecoderConfig):
    model_type = "vanilla"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_mode = "single_pass"

"""Hugging Face configuration for LCKV."""

from ..configuration_base import DecoderConfig


class LCKVConfig(DecoderConfig):
    model_type = "lckv"

    def __init__(self, *, num_passes=9, num_pre_layers=0, num_post_layers=0, prefill_mode="jacobi", **kwargs):
        super().__init__(**kwargs)
        if type(num_passes) is not int or num_passes <= 0:
            raise ValueError("num_passes must be a positive integer")
        self.num_passes = num_passes
        self.execution_mode = "jacobi"
        if prefill_mode not in {"jacobi", "autoregressive"}:
            raise ValueError("LCKV prefill_mode must be jacobi or autoregressive")
        self.prefill_mode = prefill_mode
        if any(type(n) is not int or n < 0 for n in (num_pre_layers, num_post_layers)):
            raise ValueError("feedforward layer counts must be nonnegative integers")
        if num_pre_layers + num_post_layers >= self.num_hidden_layers:
            raise ValueError("at least one feedback layer is required")
        self.num_pre_layers = num_pre_layers
        self.num_post_layers = num_post_layers

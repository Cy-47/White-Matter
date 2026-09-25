"""Configuration for a Qwen3-shaped implementation of Fan et al. feedback."""

from ..configuration_base import DecoderConfig


class FeedbackTransformerConfig(DecoderConfig):
    model_type = "feedback_transformer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_mode = "autoregressive"
        self.prefill_mode = "autoregressive"

"""Feedback Transformer Hugging Face registration."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_feedback_transformer import FeedbackTransformerConfig
from .modeling_feedback_transformer import (
    FeedbackTransformerForCausalLM,
    FeedbackTransformerModel,
    FeedbackTransformerPreTrainedModel,
)

AutoConfig.register(FeedbackTransformerConfig.model_type, FeedbackTransformerConfig)
AutoModel.register(FeedbackTransformerConfig, FeedbackTransformerModel)
AutoModelForCausalLM.register(FeedbackTransformerConfig, FeedbackTransformerForCausalLM)

__all__ = [
    "FeedbackTransformerConfig",
    "FeedbackTransformerPreTrainedModel",
    "FeedbackTransformerModel",
    "FeedbackTransformerForCausalLM",
]

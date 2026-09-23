"""WhiteMatter model family and Hugging Face auto-class registration."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_white_matter import WhiteMatterConfig
from .modeling_white_matter import WhiteMatterForCausalLM, WhiteMatterModel, WhiteMatterPreTrainedModel

AutoConfig.register(WhiteMatterConfig.model_type, WhiteMatterConfig)
AutoModel.register(WhiteMatterConfig, WhiteMatterModel)
AutoModelForCausalLM.register(WhiteMatterConfig, WhiteMatterForCausalLM)
__all__ = ["WhiteMatterConfig", "WhiteMatterPreTrainedModel", "WhiteMatterModel", "WhiteMatterForCausalLM"]

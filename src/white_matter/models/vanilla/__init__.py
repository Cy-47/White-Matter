"""Vanilla model family and Hugging Face auto-class registration."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_vanilla import VanillaConfig
from .modeling_vanilla import VanillaForCausalLM, VanillaModel, VanillaPreTrainedModel

AutoConfig.register(VanillaConfig.model_type, VanillaConfig)
AutoModel.register(VanillaConfig, VanillaModel)
AutoModelForCausalLM.register(VanillaConfig, VanillaForCausalLM)
__all__ = ["VanillaConfig", "VanillaPreTrainedModel", "VanillaModel", "VanillaForCausalLM"]

"""LCKV model family and Hugging Face auto-class registration."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_lckv import LCKVConfig
from .modeling_lckv import LCKVForCausalLM, LCKVModel, LCKVPreTrainedModel

AutoConfig.register(LCKVConfig.model_type, LCKVConfig)
AutoModel.register(LCKVConfig, LCKVModel)
AutoModelForCausalLM.register(LCKVConfig, LCKVForCausalLM)
__all__ = ["LCKVConfig", "LCKVPreTrainedModel", "LCKVModel", "LCKVForCausalLM"]

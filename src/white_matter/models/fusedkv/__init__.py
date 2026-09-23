"""FusedKV model family and Hugging Face auto-class registration."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_fusedkv import FusedKVConfig
from .modeling_fusedkv import FusedKVForCausalLM, FusedKVModel, FusedKVPreTrainedModel

AutoConfig.register(FusedKVConfig.model_type, FusedKVConfig)
AutoModel.register(FusedKVConfig, FusedKVModel)
AutoModelForCausalLM.register(FusedKVConfig, FusedKVForCausalLM)
__all__ = ["FusedKVConfig", "FusedKVPreTrainedModel", "FusedKVModel", "FusedKVForCausalLM"]

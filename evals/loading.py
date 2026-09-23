"""Load evaluation checkpoints only when every model weight is present."""

from transformers import AutoConfig, AutoModelForCausalLM


def load_complete_model(path: str, *, dtype):
    config = AutoConfig.from_pretrained(path)
    if getattr(config, "white_matter_checkpoint_format", None) is not None:
        raise ValueError(
            "This checkpoint format is incompatible with the current model layout. "
            "Load a checkpoint saved by this package."
        )
    model, info = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, output_loading_info=True)
    problems = {key: info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys") if info.get(key)}
    if problems:
        raise ValueError(f"checkpoint is incompatible with this model layout: {problems}")
    return model

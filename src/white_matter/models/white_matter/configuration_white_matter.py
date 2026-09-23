"""Hugging Face configuration for WhiteMatter."""

from ..configuration_base import DecoderConfig


class WhiteMatterConfig(DecoderConfig):
    model_type = "white_matter"

    def __init__(
        self,
        *,
        num_kv_channels=8,
        num_passes=3,
        cyclic_groups=8,
        router_layer_stride=2,
        router_prior="cyclic:0.25",
        num_pre_layers=0,
        num_post_layers=0,
        execution_mode="cyclic",
        prefill_mode=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if execution_mode not in {"cyclic", "autoregressive"}:
            raise ValueError("WhiteMatter execution_mode must be cyclic or autoregressive")
        if prefill_mode not in {None, "cyclic", "autoregressive"}:
            raise ValueError("prefill_mode must be cyclic, autoregressive, or None")
        for name, value in {
            "num_passes": num_passes,
            "cyclic_groups": cyclic_groups,
            "router_layer_stride": router_layer_stride,
        }.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if router_prior not in {"cyclic:0.25", "shifted_identity:0.25"}:
            raise ValueError(f"unsupported router prior: {router_prior!r}")
        self.num_kv_channels = num_kv_channels
        self.num_passes = num_passes
        self.cyclic_groups = cyclic_groups
        self.router_layer_stride = router_layer_stride
        self.router_prior = router_prior
        self.execution_mode = execution_mode
        # An explicit cyclic choice acknowledges frozen-prefix AR continuation.
        self.prefill_mode = prefill_mode
        if any(type(n) is not int or n < 0 for n in (num_pre_layers, num_post_layers)):
            raise ValueError("feedforward layer counts must be nonnegative integers")
        if num_pre_layers + num_post_layers >= self.num_hidden_layers:
            raise ValueError("at least one feedback layer is required")
        self.num_pre_layers = num_pre_layers
        self.num_post_layers = num_post_layers
        if (
            type(num_kv_channels) is not int
            or not 1 <= num_kv_channels <= self.num_hidden_layers - num_pre_layers - num_post_layers
        ):
            raise ValueError("num_kv_channels must be between one and the feedback depth")

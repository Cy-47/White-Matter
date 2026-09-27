"""Signed cross-layer routing and mixing."""

import torch
from torch import nn

_PEAK_LOGIT = 2.0


def _init_logits(rows: int, cols: int, mode: str) -> torch.Tensor:
    """Return one of the routing priors used in the paper code."""

    base, separator, peak_text = mode.partition(":")
    peak = float(peak_text) if separator else _PEAK_LOGIT
    logits = torch.zeros(rows, cols)

    if base == "cyclic":
        bias = peak if max(rows, cols) > 1 else 0.0
        if rows >= cols:
            for row in range(rows):
                logits[row, row % cols] = bias
        else:
            for column in range(cols):
                logits[column % rows, column] = bias
        return logits

    if base == "shifted_identity":
        if rows != cols:
            raise ValueError(f"shifted_identity init requires square matrix, got ({rows}, {cols})")
        for row in range(rows):
            logits[row, min(row + 1, cols - 1)] = peak
        return logits

    if base == "identity":
        if rows != cols:
            raise ValueError(f"identity init requires square matrix, got ({rows}, {cols})")
        logits.diagonal().fill_(peak)
        return logits

    if base == "top":
        logits[:, -1] = peak
        return logits

    raise ValueError(f"unknown router prior: {mode}")


class Router(nn.Module):
    """Content-dependent dense router with an optional strided source view."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_kv_channels: int,
        *,
        router_prior: str,
        layer_stride: int,
        dynamic: bool = True,
    ) -> None:
        super().__init__()
        if layer_stride < 1:
            raise ValueError("router_layer_stride must be positive")
        if type(dynamic) is not bool:
            raise ValueError("router_dynamic must be boolean")
        self.num_layers = num_layers
        self.layer_stride = layer_stride
        self.source_start = (num_layers - 1) % layer_stride
        self.hidden_size = hidden_size
        self.num_kv_channels = num_kv_channels
        self.router_prior = router_prior
        self.dynamic = dynamic

        self.num_sources = len(range(self.source_start, num_layers, layer_stride))
        self.linear = nn.Linear(self.num_sources * hidden_size, num_kv_channels * num_layers, bias=True)
        self.reset_parameters()
        self.linear.weight.requires_grad_(dynamic)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.linear.bias.copy_(_init_logits(self.num_kv_channels, self.num_layers, self.router_prior).reshape(-1))
        self.linear.weight.zero_()

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, num_layers, hidden_size = stacked.shape
        if (num_layers, hidden_size) != (self.num_layers, self.hidden_size):
            raise ValueError(f"router expected (*,*,{self.num_layers},{self.hidden_size}), got {tuple(stacked.shape)}")
        if not self.dynamic:
            # The learned bias is the complete static mixture. Match the dtype
            # a skipped linear projection would produce under autocast.
            dtype = (
                torch.get_autocast_dtype(stacked.device.type)
                if torch.is_autocast_enabled(stacked.device.type)
                else stacked.dtype
            )
            logits = self.linear.bias.to(dtype).view(1, 1, self.num_kv_channels, self.num_layers)
            return logits.expand(batch, sequence_length, -1, -1).to(stacked.dtype)
        selected = stacked[:, :, self.source_start :: self.layer_stride]
        # Striding limits router context; the predicted mixture still spans all layers.
        context = selected.reshape(batch, sequence_length, self.num_sources * hidden_size)
        logits = self.linear(context).view(batch, sequence_length, self.num_kv_channels, self.num_layers)
        # Use signed logits directly; the pool normalizes the mixed hidden states.
        return logits.to(stacked.dtype)


class _RoutedMixer(nn.Module):
    """Separate K- and V-side routers followed by their layer mixtures."""

    def __init__(self, k_router: Router, v_router: Router) -> None:
        super().__init__()
        self.k_router = k_router
        self.v_router = v_router

    def reset_parameters(self) -> None:
        self.k_router.reset_parameters()
        self.v_router.reset_parameters()

    def forward(self, stacked_K: torch.Tensor, stacked_V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_K = self.k_router(stacked_K)
        alpha_V = self.v_router(stacked_V)
        return (
            torch.einsum("btld,btkl->btkd", stacked_K, alpha_K),
            torch.einsum("btld,btkl->btkd", stacked_V, alpha_V),
        )


class FixedSourceMixer(nn.Module):
    """The paper LCKV pool reads the last source with a fixed weight of two."""

    logits: torch.Tensor

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.register_buffer("logits", torch.zeros(1, num_layers), persistent=False)
        self.reset_runtime_buffers()

    def reset_runtime_buffers(self) -> None:
        self.logits = torch.zeros(1, self.num_layers, device=self.logits.device)
        self.logits[:, -1] = 2.0

    def forward(self, stacked_K: torch.Tensor, stacked_V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.einsum("btld,kl->btkd", stacked_K, self.logits.to(stacked_K.dtype)),
            torch.einsum("btld,kl->btkd", stacked_V, self.logits.to(stacked_V.dtype)),
        )

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .cnn_backbones import HybridCNNEncoder
from .optical.scatter_neural_network import ScatterNeuralNetwork


class CnnScatterNetwork(nn.Module):
    """Hybrid electronic/optical model: CNN amplitude encoder followed by scatter propagation."""

    def __init__(
        self,
        input_hw: tuple[int, int],
        output_hw: tuple[int, int],
        *,
        cnn_encoder_depth: int = 1,
        cnn_encoder_width: int = 8,
        in_channels: int = 1,
        cnn_encoder_norm: str = "batchnorm",
        cnn_encoder_activation: str = "relu",
        cnn_encoder_out_activation: str = "softplus",
        normalize_input: bool = True,
        sqrt_amplitude: bool = True,
        phase_init: str = "uniform",
        tmatrix_scale: float = 1.0,
        tmatrix_compute_dtype: Optional[torch.dtype] = None,
        num_layers: int = 1,
        seed: Optional[int] = None,
        tp_enabled: bool = False,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        activation: str = "abs",
        activation_params: Optional[dict] = None,
        phase_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = HybridCNNEncoder(
            input_hw=input_hw,
            in_channels=int(in_channels),
            depth=int(cnn_encoder_depth),
            width=int(cnn_encoder_width),
            norm=str(cnn_encoder_norm),
            activation=str(cnn_encoder_activation),
            out_activation=str(cnn_encoder_out_activation),
        )
        self.scatter = ScatterNeuralNetwork(
            input_hw=input_hw,
            output_hw=output_hw,
            num_layers=int(num_layers),
            seed=seed,
            tmatrix_compute_dtype=tmatrix_compute_dtype,
            tp_enabled=bool(tp_enabled),
            tp_rank=int(tp_rank),
            tp_world_size=int(tp_world_size),
            activation=str(activation),
            activation_params=activation_params,
            phase_dropout=float(phase_dropout),
            normalize_input=bool(normalize_input),
            sqrt_amplitude=bool(sqrt_amplitude),
            phase_init=str(phase_init),
            tmatrix_scale=float(tmatrix_scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amplitude = self.encoder(x)
        return self.scatter(amplitude)

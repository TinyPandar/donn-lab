from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from donn_lab.tm.io import ShapeLike, load_tmatrix_torch


class FixedMeasuredTMLayer(nn.Module):
    def __init__(self, tm_tensor: torch.Tensor):
        super().__init__()
        self.register_buffer("transmission_matrix", tm_tensor.contiguous(), persistent=False)

    def forward(self, field_vec: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.transmission_matrix, field_vec.t()).t()


def load_measured_tmatrix(
    path: str | Path,
    *,
    shape: ShapeLike = None,
    dtype: str = "complex64",
    layout: str = "out_in",
    mmap: bool = True,
) -> torch.Tensor:
    """Load a measured transmission matrix as complex64 [N_out, N_in]."""
    return load_tmatrix_torch(path, shape=shape, dtype=dtype, layout=layout, mmap=mmap)


def normalize_tmatrix(tm: torch.Tensor, mode: str, *, input_dim: int) -> torch.Tensor:
    mode_key = str(mode or "none").lower()
    eps = 1e-12
    if mode_key in {"none", "off", "false"}:
        return tm
    if mode_key in {"match_random", "input_dim", "random_scale"}:
        target_rms = 1.0 / math.sqrt(max(int(input_dim), 1))
        rms = torch.sqrt(torch.mean(torch.abs(tm).square())).clamp_min(eps)
        return tm * (target_rms / rms)
    if mode_key in {"unit_rms", "global_unit"}:
        rms = torch.sqrt(torch.mean(torch.abs(tm).square())).clamp_min(eps)
        return tm / rms
    if mode_key in {"row_unit", "rows"}:
        denom = torch.linalg.vector_norm(tm, dim=1, keepdim=True).clamp_min(eps)
        return tm / denom
    if mode_key in {"col_unit", "cols", "column_unit"}:
        denom = torch.linalg.vector_norm(tm, dim=0, keepdim=True).clamp_min(eps)
        return tm / denom
    raise ValueError(
        "tmatrix_normalization must be one of "
        "{'none','match_random','unit_rms','row_unit','col_unit'}"
    )


class MeasuredTMScatterNetwork(nn.Module):
    """DONN forward model with a fixed, measured transmission matrix.

    The trainable part remains the phase masks. The measured matrix is stored as
    a non-trainable buffer and is expected to map flattened input modes to the
    flattened camera ROI: [N_out, N_in] @ [B, N_in]^T.
    """

    def __init__(
        self,
        input_hw: Tuple[int, int],
        output_hw: Tuple[int, int],
        *,
        tmatrix_path: str | Path,
        tmatrix_shape: str | Sequence[int] | None = None,
        tmatrix_dtype: str = "complex64",
        tmatrix_layout: str = "out_in",
        tmatrix_normalization: str = "none",
        input_modes_hw: Optional[Tuple[int, int]] = None,
        normalize_input: bool = True,
        sqrt_amplitude: bool = True,
        phase_init: str = "uniform",
        num_layers: int = 1,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        activation: str = "abs",
        activation_params: Optional[dict] = None,
        phase_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(input_hw, Sequence) or len(input_hw) != 2:
            raise ValueError("input_hw must be a (H_in, W_in) tuple")
        if not isinstance(output_hw, Sequence) or len(output_hw) != 2:
            raise ValueError("output_hw must be a (H_out, W_out) tuple")
        if not tmatrix_path:
            raise ValueError("tmatrix_path is required for MeasuredTMScatterNetwork")

        self.height, self.width = int(input_hw[0]), int(input_hw[1])
        self.output_height, self.output_width = int(output_hw[0]), int(output_hw[1])
        self.output_dim = self.output_height * self.output_width
        self.normalize_input = bool(normalize_input)
        self.sqrt_amplitude = bool(sqrt_amplitude)
        self.eps = 1e-8

        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        self.num_layers = int(num_layers)

        valid_activations = {"abs", "relu", "leaky_relu", "tanh", "elu", "softplus", "none"}
        if activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, but got {activation!r}")
        self.activation = activation
        self.activation_params = activation_params or {}

        if phase_dropout < 0.0 or phase_dropout >= 1.0:
            raise ValueError(f"phase_dropout must be in [0.0, 1.0), got {phase_dropout}")
        self.phase_dropout = float(phase_dropout)

        tm = load_measured_tmatrix(
            tmatrix_path,
            shape=tmatrix_shape,
            dtype=tmatrix_dtype,
            layout=tmatrix_layout,
            mmap=True,
        )
        if tm.shape[0] != self.output_dim:
            raise ValueError(
                f"Measured TM output dim {tm.shape[0]} does not match output_hw "
                f"{output_hw} -> {self.output_dim}"
            )

        if input_modes_hw is None:
            if tm.shape[1] == self.height * self.width:
                input_modes_hw = (self.height, self.width)
            else:
                side = int(math.isqrt(int(tm.shape[1])))
                if side * side == int(tm.shape[1]):
                    input_modes_hw = (side, side)
                else:
                    raise ValueError(
                        f"Cannot infer input mode grid from N_in={tm.shape[1]}; "
                        "set tmatrix_input_h and tmatrix_input_w in the config"
                    )
        self.mode_height, self.mode_width = int(input_modes_hw[0]), int(input_modes_hw[1])
        self.input_dim = self.mode_height * self.mode_width
        if self.input_dim != tm.shape[1]:
            raise ValueError(
                f"input_modes_hw {input_modes_hw} has {self.input_dim} modes, "
                f"but measured TM has N_in={tm.shape[1]}"
            )

        tm = normalize_tmatrix(tm, tmatrix_normalization, input_dim=self.input_dim)
        self.transmission_matrix = FixedMeasuredTMLayer(tm_tensor=tm)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))

        phases = []
        for _ in range(self.num_layers):
            phase_l = torch.empty(1, 1, self.mode_height, self.mode_width, device=device, dtype=torch.float32)
            if phase_init == "zeros":
                nn.init.zeros_(phase_l)
            elif phase_init == "uniform":
                if gen is None:
                    nn.init.uniform_(phase_l, a=0.0, b=2.0 * math.pi)
                else:
                    phase_l.uniform_(0.0, 2.0 * math.pi, generator=gen)
            else:
                raise ValueError("phase_init must be one of {'zeros', 'uniform'}")
            phases.append(nn.Parameter(phase_l))
        self.phases = nn.ParameterList(phases)

    def _normalize_spatial_per_sample(self, amplitude: torch.Tensor) -> torch.Tensor:
        amin = amplitude.amin(dim=(2, 3), keepdim=True)
        amax = amplitude.amax(dim=(2, 3), keepdim=True)
        return (amplitude - amin) / (amax - amin + self.eps)

    def _image_to_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Input must be a 4D tensor [B,C,H,W]")
        if x.shape[-2] != self.height or x.shape[-1] != self.width:
            raise ValueError(
                f"Input spatial size must be ({self.height}, {self.width}) "
                f"but got ({x.shape[-2]}, {x.shape[-1]})"
            )

        x = x.float()
        if x.shape[1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            amplitude = (x * weights).sum(dim=1, keepdim=True)
        elif x.shape[1] == 1:
            amplitude = x
        else:
            amplitude = x.mean(dim=1, keepdim=True)

        if self.normalize_input:
            amplitude = self._normalize_spatial_per_sample(amplitude)
        amplitude = torch.clamp(amplitude, min=0.0)
        if self.sqrt_amplitude:
            amplitude = torch.sqrt(amplitude + self.eps)
        if (self.mode_height, self.mode_width) != (self.height, self.width):
            amplitude = F.interpolate(amplitude, size=(self.mode_height, self.mode_width), mode="area")
        return amplitude

    def _capture_apply_activation(self, y: torch.Tensor) -> torch.Tensor:
        intensity = torch.abs(y).square()
        if self.activation == "abs":
            return intensity
        if self.activation == "relu":
            return F.relu(intensity)
        if self.activation == "leaky_relu":
            negative_slope = self.activation_params.get("negative_slope", 0.01)
            return F.leaky_relu(intensity, negative_slope=negative_slope)
        if self.activation == "tanh":
            return torch.tanh(intensity)
        if self.activation == "elu":
            alpha = self.activation_params.get("alpha", 1.0)
            return F.elu(intensity, alpha=alpha)
        if self.activation == "softplus":
            beta = self.activation_params.get("beta", 1.0)
            threshold = self.activation_params.get("threshold", 20.0)
            return F.softplus(intensity, beta=beta, threshold=threshold)
        return intensity

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_complex: bool = False,
        return_intermediate_amplitude: bool = False,
    ):
        batch_size = x.shape[0]
        propagation_amplitude = self._image_to_amplitude(x)

        detector_field = None
        detector_intensity = None
        for layer_idx in range(self.num_layers):
            propagation_amplitude = self._normalize_spatial_per_sample(propagation_amplitude)

            phase_l = self.phases[layer_idx]
            if self.phase_dropout > 0.0 and self.training:
                dropout_mask = torch.bernoulli(torch.ones_like(phase_l) * (1.0 - self.phase_dropout))
                dropout_mask = dropout_mask / (1.0 - self.phase_dropout)
                phase_l = phase_l * dropout_mask

            field = torch.polar(propagation_amplitude.float(), phase_l)
            field_vec = field.reshape(batch_size, -1)
            detector_vec = self.transmission_matrix(field_vec)
            detector_field = detector_vec.reshape(batch_size, 1, self.output_height, self.output_width)
            detector_intensity = self._capture_apply_activation(detector_field)

            if layer_idx < self.num_layers - 1:
                if (self.output_height, self.output_width) != (self.mode_height, self.mode_width):
                    propagation_amplitude = F.interpolate(
                        detector_intensity,
                        size=(self.mode_height, self.mode_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    propagation_amplitude = detector_intensity

        result = detector_intensity.view(batch_size, self.output_height, self.output_width)
        if return_complex and return_intermediate_amplitude:
            return result, detector_field, propagation_amplitude
        if return_complex:
            return result, detector_field
        if return_intermediate_amplitude:
            return result, propagation_amplitude
        return result


__all__ = ["MeasuredTMScatterNetwork", "load_measured_tmatrix", "normalize_tmatrix"]

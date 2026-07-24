from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class FixedTMLayer(nn.Module):
    def __init__(self, tm_tensor: torch.Tensor):
        super().__init__()
        self.register_buffer("transmission_matrix", tm_tensor.contiguous())

    def forward(self, field_vec: torch.Tensor) -> torch.Tensor:
        if not field_vec.requires_grad:
            return torch.matmul(self.transmission_matrix, field_vec.t()).t()
        return FixedTM_Mult.apply(field_vec, self.transmission_matrix)


class FixedTM_Mult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(w)
        return torch.matmul(w, x.t()).t()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (w,) = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.matmul(grad_output.conj(), w).conj()
        return grad_x, None


class _TPAllGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y_local: torch.Tensor,
        rows_per_rank: int,
        output_dim: int,
        world_size: int,
    ) -> torch.Tensor:
        ctx.rows_per_rank = int(rows_per_rank)
        ctx.output_dim = int(output_dim)
        ctx.world_size = int(world_size)

        batch_size = y_local.shape[0]
        local_cols = y_local.shape[1]
        ctx.local_cols = int(local_cols)

        pad_cols = rows_per_rank - local_cols
        if pad_cols > 0:
            pad = torch.zeros(batch_size, pad_cols, dtype=y_local.dtype, device=y_local.device)
            y_pad = torch.cat([y_local, pad], dim=1)
        else:
            y_pad = y_local

        y_real = y_pad.real.contiguous()
        y_imag = y_pad.imag.contiguous()
        gather_real = [torch.empty_like(y_real) for _ in range(world_size)]
        gather_imag = [torch.empty_like(y_imag) for _ in range(world_size)]
        dist.all_gather(gather_real, y_real)
        dist.all_gather(gather_imag, y_imag)

        y_full_real = torch.cat(gather_real, dim=1)[:, :output_dim]
        y_full_imag = torch.cat(gather_imag, dim=1)[:, :output_dim]
        return torch.complex(y_full_real, y_full_imag)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not dist.is_available() or not dist.is_initialized() or ctx.world_size == 1:
            return grad_output, None, None, None

        rank = dist.get_rank()
        start = rank * ctx.rows_per_rank
        end = min(start + ctx.rows_per_rank, ctx.output_dim)
        grad_local = grad_output[:, start:end]
        return grad_local, None, None, None


class ScatterNeuralNetwork(nn.Module):
    """基于稠密固定传输矩阵的光学数字孪生前向模型。"""

    def __init__(
        self,
        input_hw: Tuple[int, int],
        output_hw: Tuple[int, int],
        *,
        normalize_input: bool = True,
        sqrt_amplitude: bool = True,
        phase_init: str = "uniform",
        tmatrix_scale: float = 1.0,
        tmatrix_compute_dtype: Optional[torch.dtype] = None,
        num_layers: int = 1,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tp_enabled: bool = False,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        activation: str = "none",
        activation_params: Optional[dict] = None,
        phase_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(input_hw, Sequence) or len(input_hw) != 2:
            raise ValueError("input_hw must be a (H_in, W_in) tuple")
        if not isinstance(output_hw, Sequence) or len(output_hw) != 2:
            raise ValueError("output_hw must be a (H_out, W_out) tuple")

        self.height, self.width = int(input_hw[0]), int(input_hw[1])
        self.output_height, self.output_width = int(output_hw[0]), int(output_hw[1])
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output_hw must be positive")

        self.output_dim = self.output_height * self.output_width
        self.normalize_input = bool(normalize_input)
        self.sqrt_amplitude = bool(sqrt_amplitude)
        self.eps = 1e-8

        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        self.num_layers = int(num_layers)

        valid_activations = {"abs", "relu", "leaky_relu", "tanh", "elu", "softplus", "none"}
        if activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, but got '{activation}'")
        self.activation = activation
        self.activation_params = activation_params or {}

        if phase_dropout < 0.0 or phase_dropout >= 1.0:
            raise ValueError(f"phase_dropout must be in [0.0, 1.0), but got {phase_dropout}")
        self.phase_dropout = float(phase_dropout)

        in_dim = self.height * self.width

        if tmatrix_compute_dtype is not None and tmatrix_compute_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("tmatrix_compute_dtype must be one of {None, torch.float16, torch.bfloat16}")
        self.tmatrix_compute_dtype = tmatrix_compute_dtype

        self.tp_enabled = bool(tp_enabled and (tp_world_size or 1) > 1)
        self.tp_rank = int(tp_rank)
        self.tp_world_size = int(tp_world_size) if int(tp_world_size) > 0 else 1
        self.rows_per_rank = (self.output_dim + self.tp_world_size - 1) // max(self.tp_world_size, 1)
        if self.tp_enabled:
            start = self.tp_rank * self.rows_per_rank
            end = min(start + self.rows_per_rank, self.output_dim)
        else:
            start, end = 0, self.output_dim
        self._row_start = start
        self._row_end = end

        phases = []
        for _ in range(self.num_layers):
            phase_l = torch.empty(1, 1, self.height, self.width, device=device, dtype=torch.float32)
            if phase_init == "zeros":
                nn.init.zeros_(phase_l)
            elif phase_init == "uniform":
                nn.init.uniform_(phase_l, a=0.0, b=2.0 * math.pi)
            else:
                raise ValueError("phase_init must be one of {'zeros', 'uniform'}")
            phases.append(nn.Parameter(phase_l))
        self.phases = nn.ParameterList(phases)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed) + int(self.tp_rank))

        local_rows = self._row_end - self._row_start
        scale = float(tmatrix_scale) / math.sqrt(in_dim)
        real = torch.randn(local_rows, in_dim, generator=gen, device=device, dtype=torch.float32)
        imag = torch.randn(local_rows, in_dim, generator=gen, device=device, dtype=torch.float32)
        transmission_matrix = torch.complex(real * scale, imag * scale).to(torch.complex64)
        self.transmission_matrix = FixedTMLayer(tm_tensor=transmission_matrix)

        if dtype is not None:
            self.to(dtype=dtype)

    def _normalize_spatial_per_sample(self, amplitude: torch.Tensor) -> torch.Tensor:
        """对每个样本仅在空间维度上做归一化，避免 batch 内样本相互影响。"""
        amin = amplitude.amin(dim=(2, 3), keepdim=True)
        amax = amplitude.amax(dim=(2, 3), keepdim=True)
        return (amplitude - amin) / (amax - amin + self.eps)

    def _image_to_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Input must be a 4D tensor [B, C, H, W]")
        if x.shape[-2] != self.height or x.shape[-1] != self.width:
            raise ValueError(
                f"Input spatial size must be ({self.height}, {self.width}) "
                f"but got ({x.shape[-2]}, {x.shape[-1]})"
            )

        x = x.float()
        channels = x.shape[1]
        if channels != 1:
            raise ValueError(
                "ScatterNeuralNetwork expects a single-channel input tensor [B,1,H,W]. "
                "Set data.input_mode to gray/mean/r/g/b in the preprocessing config."
            )
        amplitude = x

        if self.normalize_input:
            amplitude = self._normalize_spatial_per_sample(amplitude)

        amplitude = torch.clamp(amplitude, min=0.0)
        if self.sqrt_amplitude:
            amplitude = torch.sqrt(amplitude + self.eps)
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
        """返回最终探测面强度；可选同时返回复数场和内部传播幅度。"""
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
            y_local = self.transmission_matrix(field_vec)

            if self.tp_enabled and dist.is_available() and dist.is_initialized() and self.tp_world_size > 1:
                detector_vec = _TPAllGather.apply(
                    y_local,
                    self.rows_per_rank,
                    self.output_dim,
                    self.tp_world_size,
                )
            else:
                detector_vec = y_local

            detector_field = detector_vec.reshape(batch_size, 1, self.output_height, self.output_width)
            detector_intensity = self._capture_apply_activation(detector_field)

            if layer_idx < self.num_layers - 1:
                if (self.output_height, self.output_width) != (self.height, self.width):
                    propagation_amplitude = F.interpolate(
                        detector_intensity,
                        size=(self.height, self.width),
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


__all__ = ["ScatterNeuralNetwork"]

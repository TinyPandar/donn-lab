from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


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


class FixedTMLayer(nn.Module):
    def __init__(self, tm_tensor: torch.Tensor):
        super().__init__()
        self.register_buffer("transmission_matrix", tm_tensor.contiguous())

    def forward(self, field_vec: torch.Tensor) -> torch.Tensor:
        if not field_vec.requires_grad:
            return torch.matmul(self.transmission_matrix, field_vec.t()).t()
        return FixedTM_Mult.apply(field_vec, self.transmission_matrix)


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


class ComplexFieldGenerator(nn.Module):
    def __init__(
        self,
        field_hw: Tuple[int, int],
        *,
        normalize_input: bool = True,
        sqrt_amplitude: bool = True,
        learnable_amplitude_bias: bool = False,
        phase_init: str = "uniform",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.height, self.width = int(field_hw[0]), int(field_hw[1])
        self.normalize_input = bool(normalize_input)
        self.sqrt_amplitude = bool(sqrt_amplitude)
        self.eps = 1e-8

        phase = torch.empty(1, 1, self.height, self.width, device=device, dtype=torch.float32)
        if phase_init == "zeros":
            nn.init.zeros_(phase)
        elif phase_init == "uniform":
            nn.init.uniform_(phase, a=0.0, b=2.0 * math.pi)
        else:
            raise ValueError("phase_init must be one of {'zeros', 'uniform'}")
        self.phase = nn.Parameter(phase)

        if learnable_amplitude_bias:
            self.amplitude_bias = nn.Parameter(
                torch.zeros(1, 1, self.height, self.width, device=device, dtype=torch.float32)
            )
        else:
            self.register_parameter("amplitude_bias", None)

    def _normalize_spatial_per_sample(self, amplitude: torch.Tensor) -> torch.Tensor:
        amin = amplitude.amin(dim=(2, 3), keepdim=True)
        amax = amplitude.amax(dim=(2, 3), keepdim=True)
        return (amplitude - amin) / (amax - amin + self.eps)

    def _image_to_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Input must be a 4D tensor [B, C, H, W]")
        if x.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"Input spatial size must be ({self.height}, {self.width}), got {tuple(x.shape[-2:])}"
            )

        x = x.float()
        channels = x.shape[1]
        if channels != 1:
            raise ValueError(
                "ScatterTileNeuralNetwork expects a single-channel input tensor [B,1,H,W]. "
                "Set data.input_mode to gray/mean/r/g/b in the preprocessing config."
            )
        amplitude = x

        if self.normalize_input:
            amplitude = self._normalize_spatial_per_sample(amplitude)

        amplitude = torch.clamp(amplitude, min=0.0)
        if self.amplitude_bias is not None:
            amplitude = torch.clamp(amplitude + self.amplitude_bias, min=0.0)
        if self.sqrt_amplitude:
            amplitude = torch.sqrt(amplitude + self.eps)
        return amplitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amplitude = self._image_to_amplitude(x)
        return torch.polar(amplitude, self.phase)


class MicromirrorLayoutLayer(nn.Module):
    def __init__(
        self,
        field_hw: Tuple[int, int],
        *,
        layout_mode: str = "block_average",
        tile_hw: Tuple[int, int] = (8, 8),
        learnable_tile_gains: bool = False,
        energy_normalize: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.height, self.width = int(field_hw[0]), int(field_hw[1])
        self.tile_h, self.tile_w = int(tile_hw[0]), int(tile_hw[1])
        self.energy_normalize = bool(energy_normalize)

        valid_modes = {"identity", "block_average", "block_replication"}
        if layout_mode not in valid_modes:
            raise ValueError(f"layout_mode must be one of {valid_modes}, got '{layout_mode}'")
        self.layout_mode = layout_mode

        self.grid_h = math.ceil(self.height / self.tile_h)
        self.grid_w = math.ceil(self.width / self.tile_w)

        if learnable_tile_gains:
            self.tile_gains = nn.Parameter(
                torch.ones(1, 1, self.grid_h, self.grid_w, device=device, dtype=torch.float32)
            )
        else:
            self.register_parameter("tile_gains", None)

    def _apply_tile_gains(self, field: torch.Tensor) -> torch.Tensor:
        if self.tile_gains is None:
            return field
        gains = F.interpolate(self.tile_gains, size=(self.height, self.width), mode="nearest")
        return field * gains

    def _normalize_energy(self, input_field: torch.Tensor, output_field: torch.Tensor) -> torch.Tensor:
        if not self.energy_normalize:
            return output_field
        in_energy = torch.mean(torch.abs(input_field).square(), dim=(2, 3), keepdim=True)
        out_energy = torch.mean(torch.abs(output_field).square(), dim=(2, 3), keepdim=True)
        scale = torch.sqrt(in_energy / (out_energy + 1e-8))
        return output_field * scale

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.shape[-2:] != (self.height, self.width):
            raise ValueError(f"field spatial size must be ({self.height}, {self.width})")

        if self.layout_mode == "identity":
            out = field
        else:
            pooled_real = F.avg_pool2d(
                field.real,
                kernel_size=(self.tile_h, self.tile_w),
                stride=(self.tile_h, self.tile_w),
                ceil_mode=True,
            )
            pooled_imag = F.avg_pool2d(
                field.imag,
                kernel_size=(self.tile_h, self.tile_w),
                stride=(self.tile_h, self.tile_w),
                ceil_mode=True,
            )
            pooled = torch.complex(pooled_real, pooled_imag)

            if self.layout_mode == "block_average":
                out_real = F.interpolate(pooled.real, size=(self.height, self.width), mode="nearest")
                out_imag = F.interpolate(pooled.imag, size=(self.height, self.width), mode="nearest")
                out = torch.complex(out_real, out_imag)
            else:
                rep_real = F.interpolate(pooled.real, size=(self.height, self.width), mode="nearest")
                rep_imag = F.interpolate(pooled.imag, size=(self.height, self.width), mode="nearest")
                replicated = torch.complex(rep_real, rep_imag)
                mask = torch.ones_like(field.real)
                mask[..., :: self.tile_h, :: self.tile_w] = 0.0
                out = replicated * (1.0 - mask) + field * mask

        out = self._apply_tile_gains(out)
        return self._normalize_energy(field, out)


class SharedScatteringLayer(nn.Module):
    def __init__(
        self,
        input_hw: Tuple[int, int],
        output_hw: Tuple[int, int],
        *,
        tmatrix_scale: float = 1.0,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        tp_enabled: bool = False,
        tp_rank: int = 0,
        tp_world_size: int = 1,
    ) -> None:
        super().__init__()
        self.input_hw = (int(input_hw[0]), int(input_hw[1]))
        self.output_hw = (int(output_hw[0]), int(output_hw[1]))
        self.input_dim = self.input_hw[0] * self.input_hw[1]
        self.output_dim = self.output_hw[0] * self.output_hw[1]

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

        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed) + int(self.tp_rank))

        local_rows = self._row_end - self._row_start
        scale = float(tmatrix_scale) / math.sqrt(self.input_dim)
        real = torch.randn(local_rows, self.input_dim, generator=gen, device=device, dtype=torch.float32)
        imag = torch.randn(local_rows, self.input_dim, generator=gen, device=device, dtype=torch.float32)
        tm = torch.complex(real * scale, imag * scale).to(torch.complex64)
        self.tm_layer = FixedTMLayer(tm)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        batch_size = field.shape[0]
        field_vec = field.reshape(batch_size, -1)
        y_local = self.tm_layer(field_vec)

        if self.tp_enabled and dist.is_available() and dist.is_initialized() and self.tp_world_size > 1:
            y = _TPAllGather.apply(
                y_local,
                self.rows_per_rank,
                self.output_dim,
                self.tp_world_size,
            )
        else:
            y = y_local

        return y.reshape(batch_size, 1, self.output_hw[0], self.output_hw[1])


class ScatterTileNeuralNetwork(nn.Module):
    """兼容 ScatterNeuralNetwork 风格接口的 tile 化级联散射网络。"""

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
        tile_hw: Tuple[int, int] = (8, 8),
        layout_mode: str = "block_average",
        learnable_tile_gains: bool = False,
        energy_normalize: bool = True,
        learnable_amplitude_bias: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(input_hw, Sequence) or len(input_hw) != 2:
            raise ValueError("input_hw must be a (H_in, W_in) tuple")
        if not isinstance(output_hw, Sequence) or len(output_hw) != 2:
            raise ValueError("output_hw must be a (H_out, W_out) tuple")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if phase_dropout != 0.0:
            raise ValueError("ScatterTileNeuralNetwork does not currently support phase_dropout")
        if tmatrix_compute_dtype is not None:
            raise ValueError("ScatterTileNeuralNetwork does not currently use tmatrix_compute_dtype")
        if activation_params:
            raise ValueError("ScatterTileNeuralNetwork does not currently use activation_params")

        self.height, self.width = int(input_hw[0]), int(input_hw[1])
        self.output_height, self.output_width = int(output_hw[0]), int(output_hw[1])
        self.output_dim = self.output_height * self.output_width
        self.num_layers = int(num_layers)
        self.input_hw = (self.height, self.width)
        self.output_hw = (self.output_height, self.output_width)

        activation_map = {
            "none": "intensity",
            "abs": "intensity",
        }
        if activation not in activation_map:
            raise ValueError("ScatterTileNeuralNetwork currently supports activation in {'none', 'abs'}")
        self.output_mode = activation_map[activation]

        self.field_generators = nn.ModuleList(
            [
                ComplexFieldGenerator(
                    self.input_hw,
                    normalize_input=normalize_input,
                    sqrt_amplitude=sqrt_amplitude,
                    learnable_amplitude_bias=learnable_amplitude_bias,
                    phase_init=phase_init,
                    device=device,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.layout_layer = MicromirrorLayoutLayer(
            self.input_hw,
            layout_mode=layout_mode,
            tile_hw=tile_hw,
            learnable_tile_gains=learnable_tile_gains,
            energy_normalize=energy_normalize,
            device=device,
        )
        self.scattering_layer = SharedScatteringLayer(
            self.input_hw,
            self.output_hw,
            tmatrix_scale=tmatrix_scale,
            seed=seed,
            device=device,
            tp_enabled=tp_enabled,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )

        if dtype is not None:
            self.to(dtype=dtype)

    @property
    def phases(self):
        return nn.ParameterList([generator.phase for generator in self.field_generators])

    def _project_output(self, detector_field: torch.Tensor) -> torch.Tensor:
        if self.output_mode == "complex":
            return detector_field
        if self.output_mode == "amplitude":
            return torch.abs(detector_field)
        return torch.abs(detector_field).square()

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_complex: bool = False,
        return_intermediate_amplitude: bool = False,
        return_details: bool = False,
    ):
        propagation_input = x
        target_fields = []
        laid_out_fields = []
        detector_fields = []
        propagation_amplitudes = []

        for layer_idx, field_generator in enumerate(self.field_generators):
            target_field = field_generator(propagation_input)
            laid_out_field = self.layout_layer(target_field)
            detector_field = self.scattering_layer(laid_out_field)

            target_fields.append(target_field)
            laid_out_fields.append(laid_out_field)
            detector_fields.append(detector_field)

            if layer_idx < self.num_layers - 1:
                propagation_amplitude = torch.abs(detector_field)
                if self.output_hw != self.input_hw:
                    propagation_amplitude = F.interpolate(
                        propagation_amplitude,
                        size=self.input_hw,
                        mode="bilinear",
                        align_corners=False,
                    )
                propagation_amplitudes.append(propagation_amplitude)
                propagation_input = propagation_amplitude

        detector_field = detector_fields[-1]
        output = self._project_output(detector_field).view(x.shape[0], self.output_height, self.output_width)

        if return_details:
            return {
                "output": output,
                "target_field": target_fields[-1],
                "target_fields": target_fields,
                "laid_out_field": laid_out_fields[-1],
                "laid_out_fields": laid_out_fields,
                "detector_field": detector_field,
                "detector_fields": detector_fields,
                "propagation_amplitudes": propagation_amplitudes,
            }
        if return_complex and return_intermediate_amplitude:
            return output, detector_field, torch.abs(laid_out_fields[-1])
        if return_complex:
            return output, detector_field
        if return_intermediate_amplitude:
            return output, torch.abs(laid_out_fields[-1])
        return output


__all__ = ["ScatterTileNeuralNetwork"]

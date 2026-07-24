from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _fftshift2d(x: torch.Tensor) -> torch.Tensor:
    for dim in (-2, -1):
        n = x.size(dim)
        x = torch.roll(x, shifts=n // 2, dims=dim)
    return x


def _ifftshift2d(x: torch.Tensor) -> torch.Tensor:
    for dim in (-2, -1):
        n = x.size(dim)
        x = torch.roll(x, shifts=-(n // 2), dims=dim)
    return x


class _TPAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_local: torch.Tensor, rows_per_rank: int, output_dim: int, world_size: int) -> torch.Tensor:
        ctx.rows_per_rank = int(rows_per_rank)
        ctx.output_dim = int(output_dim)
        ctx.world_size = int(world_size)
        # Pad local columns to rows_per_rank for even concat
        B = y_local.shape[0]
        local_cols = y_local.shape[1]
        ctx.local_cols = int(local_cols)
        pad_cols = rows_per_rank - local_cols
        if pad_cols > 0:
            pad = torch.zeros(B, pad_cols, dtype=y_local.dtype, device=y_local.device)
            y_pad = torch.cat([y_local, pad], dim=1)
        else:
            y_pad = y_local
        # Gather real/imag separately for safety
        y_r = y_pad.real.contiguous()
        y_i = y_pad.imag.contiguous()
        gather_r = [torch.empty_like(y_r) for _ in range(world_size)]
        gather_i = [torch.empty_like(y_i) for _ in range(world_size)]
        dist.all_gather(gather_r, y_r)
        dist.all_gather(gather_i, y_i)
        y_full_r = torch.cat(gather_r, dim=1)[:, :output_dim]
        y_full_i = torch.cat(gather_i, dim=1)[:, :output_dim]
        return torch.complex(y_full_r, y_full_i)

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
    """具有可学习相位和固定复数传输矩阵的散射神经网络。

    参数
    ----------
    input_hw:
        期望输入空间尺寸的二元组 (H_in, W_in)。
    output_hw:
        期望输出空间尺寸的二元组 (H_out, W_out)，前向输出为 [B, H_out, W_out]。
    normalize_input:
        若为 True，按样本在空间维度上做 min-max，将幅度归一化到 [0, 1]。
    sqrt_amplitude:
        若为 True，将输入视为强度并通过开方转换为幅度；
        若为 False，直接使用（经夹紧）的数值作为幅度。
    return_intensity:
        若为 True，返回 |y|^2（实数）；若为 False，返回复数 y。
    phase_init:
        相位的初始化策略，取值为 {"zeros", "uniform"}。当为 "uniform" 时，
        在区间 [0, 2π) 上均匀初始化。
    tmatrix_scale:
        在按 sqrt(in_dim) 归一化之前，用于实部/虚部正态初始化的标准差缩放因子。
    tmatrix_compute_dtype:
        传输计算时使用的实数精度（用于分解为实/虚两路 matmul 的路径）。
        可为 None/torch.float16/torch.bfloat16；为 None 时使用复数 matmul（complex64）。
    tmatrix_sparsity:
        传输矩阵的空置率（稀疏度），范围 [0.0, 1.0)。例如 0.98 表示 98% 的元素为空。
        默认值为 0.98。
    seed:
        可选的随机数种子，用于确定性地初始化传输矩阵。
    device, dtype:
        可选的 torch 设备与 dtype，应用于参数/缓冲区。
    activation:
        激活函数类型，可选值：{"abs", "relu", "leaky_relu", "sigmoid", "tanh", "elu", "softplus"}。
        默认为 "abs"（绝对值函数）。
    activation_params:
        激活函数的参数字典，例如 {"negative_slope": 0.01} 用于 LeakyReLU。
    normalize_negative:
        若为 True，对激活函数输出的负数部分进行正则化处理。
    phase_dropout:
        相位参数的 dropout 概率，范围 [0.0, 1.0)。在训练时，会随机将某些位置的相位设置为0。
        设置为 0.0 时禁用 dropout。仅在训练模式下生效。
    """

    def __init__(
        self,
        input_hw: Tuple[int, int],
        output_hw: Tuple[int, int],
        *,
        normalize_input: bool = True,
        sqrt_amplitude: bool = True,
        return_intensity: bool = True,
        phase_init: str = "uniform",
        tmatrix_scale: float = 1.0,
        tmatrix_compute_dtype: Optional[torch.dtype] = None,
        tmatrix_sparsity: float = 0.98,
        num_layers: int = 1,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        # Tensor-parallel options
        tp_enabled: bool = False,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        # 激活函数选项
        activation: str = "none",
        activation_params: Optional[dict] = None,
        normalize_negative: bool = False,
        # Dropout 选项
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
        self.return_intensity = bool(return_intensity)
        self.eps: float = 1e-8
        # 前向传播重复层数（相位调制+传输+幅度更新的次数）
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        self.num_layers: int = int(num_layers)

        # 激活函数配置
        valid_activations = {"abs", "relu", "leaky_relu", "sigmoid", "tanh", "elu", "softplus", "none"}
        if activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, but got '{activation}'")
        self.activation: str = activation
        self.activation_params: dict = activation_params or {}
        self.normalize_negative: bool = bool(normalize_negative)
        
        # 相位参数 dropout 配置
        if phase_dropout < 0.0 or phase_dropout >= 1.0:
            raise ValueError(f"phase_dropout must be in [0.0, 1.0), but got {phase_dropout}")
        self.phase_dropout: float = float(phase_dropout)

        in_dim = self.height * self.width

        # 设置传输矩阵的计算精度（仅影响前向中的 matmul 计算，不改变存储精度）
        if tmatrix_compute_dtype is not None and tmatrix_compute_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("tmatrix_compute_dtype must be one of {None, torch.float16, torch.bfloat16}")
        self.tmatrix_compute_dtype: Optional[torch.dtype] = tmatrix_compute_dtype
        
        # 传输矩阵空置率配置
        if tmatrix_sparsity < 0.0 or tmatrix_sparsity >= 1.0:
            raise ValueError(f"tmatrix_sparsity must be in [0.0, 1.0), but got {tmatrix_sparsity}")
        self.tmatrix_sparsity: float = float(tmatrix_sparsity)

        # Tensor parallel config
        self.tp_enabled: bool = bool(tp_enabled and (tp_world_size or 1) > 1)
        self.tp_rank: int = int(tp_rank)
        self.tp_world_size: int = int(tp_world_size) if int(tp_world_size) > 0 else 1
        # 每 rank 均分行数（最后一块可能更短）；用于 all_gather 时对齐 padding
        self.rows_per_rank: int = (self.output_dim + self.tp_world_size - 1) // max(self.tp_world_size, 1)
        if self.tp_enabled:
            start = self.tp_rank * self.rows_per_rank
            end = min(start + self.rows_per_rank, self.output_dim)
        else:
            start, end = 0, self.output_dim
        self._row_start: int = start
        self._row_end: int = end

        # 可学习的逐像素相位（在 batch 维度上广播），单位：弧度
        # 为每一层分配独立的相位参数；使用 float32 存储以避免半精度构造复数的不支持
        phases: list = []
        for _ in range(self.num_layers):
            # 采用二维相位形状，便于与 [B, 1, H, W] 的幅度直接广播
            phase_l = torch.empty(1, 1, self.height, self.width, device=device, dtype=torch.float32)
            if phase_init == "zeros":
                nn.init.zeros_(phase_l)
            elif phase_init == "uniform":
                nn.init.uniform_(phase_l, a=0.0, b=2.0 * math.pi)
            else:
                raise ValueError("phase_init must be one of {'zeros', 'uniform'}")
            phases.append(nn.Parameter(phase_l))
        self.phases = nn.ParameterList(phases)  # 每层一个形状为 [1, 1, H, W] 的相位参数

        # 作为 buffer 注册的固定随机复数传输矩阵（或其本地 shard）
        if seed is not None:
            gen = torch.Generator(device=device)
            # 不同 rank 使用不同 seed，确保 shard 可复现
            rank_seed = int(seed) + int(self.tp_rank)
            gen.manual_seed(rank_seed)
        else:
            gen = None

        local_rows = self._row_end - self._row_start
        # 生成本地 shard 的实部/虚部（float32），随后与 in_dim 缩放以保持方差尺度，再组成 complex64 存储
        scale = float(tmatrix_scale) / math.sqrt(in_dim)
        
        if self.tmatrix_sparsity > 0:
            nnz = int(local_rows * in_dim * (1 - self.tmatrix_sparsity))
            # 随机生成 nnz 的坐标
            row_idx = torch.randint(local_rows, (nnz,), device=device)
            col_idx = torch.randint(in_dim,     (nnz,), device=device)
            indices = torch.stack([row_idx, col_idx], dim=0)   # [2, nnz]
            
            values_real = torch.randn(nnz, generator=gen, device=device)
            values_imag = torch.randn(nnz, generator=gen, device=device)
            values = torch.complex(values_real * scale, values_imag * scale).to(torch.complex64)
            
            transmission_matrix = torch.sparse_coo_tensor(
                indices,
                values,
                size=(local_rows, in_dim),
                dtype=torch.complex64,
                device=device
            ).coalesce()
        else:
            # 稠密矩阵初始化
            real = torch.randn(local_rows, in_dim, generator=gen, device=device, dtype=torch.float32)
            imag = torch.randn(local_rows, in_dim, generator=gen, device=device, dtype=torch.float32)
            transmission_matrix = torch.complex(real * scale, imag * scale).to(torch.complex64)

        self.register_buffer("transmission_matrix", transmission_matrix, persistent=True)

    @torch.no_grad()
    def reset_transmission_matrix(self, *, seed: Optional[int] = None, tmatrix_scale: float = 1.0) -> None:
        """使用新的随机种子重新初始化固定的传输矩阵。

        这不会影响梯度，因为该矩阵是不可训练的 buffer。
        """
        in_dim = self.height * self.width
        device = self.transmission_matrix.device
        if seed is not None:
            gen = torch.Generator(device=device)
            rank_seed = int(seed) + int(self.tp_rank)
            gen.manual_seed(rank_seed)
        else:
            gen = None

        local_rows = self._row_end - self._row_start
        scale = float(tmatrix_scale) / math.sqrt(in_dim)

        if self.transmission_matrix.is_sparse:
            nnz = int(local_rows * in_dim * (1 - self.tmatrix_sparsity))
            row_idx = torch.randint(local_rows, (nnz,), device=device)
            col_idx = torch.randint(in_dim,     (nnz,), device=device)
            indices = torch.stack([row_idx, col_idx], dim=0)
            
            values_real = torch.randn(nnz, generator=gen, device=device)
            values_imag = torch.randn(nnz, generator=gen, device=device)
            values = torch.complex(values_real * scale, values_imag * scale).to(torch.complex64)
            
            new_tm = torch.sparse_coo_tensor(
                indices,
                values,
                size=(local_rows, in_dim),
                dtype=torch.complex64,
                device=device
            ).coalesce()
        else:
            real = torch.randn(local_rows, in_dim, generator=gen, device=device, dtype=torch.float32)
            imag = torch.randn(local_rows, in_dim, generator=gen, device=device, dtype=torch.float32)
            new_tm = torch.complex(real * scale, imag * scale).to(torch.complex64)

        # Update the buffer
        self.transmission_matrix.data = new_tm

    @torch.no_grad()
    def set_transmission_matrix(self, tm: torch.Tensor) -> None:
        """从外部张量设置固定的复数传输矩阵。

        需要一个形状为 [H_out*W_out, H_in*W_in] 的复数张量。提供的张量将被移动到
        本模块的设备/数据类型，并更新缓冲区 `transmission_matrix`。
        """
        if tm.ndim != 2:
            raise ValueError("Transmission matrix must be 2D [output_dim, in_dim]")

        in_dim = self.height * self.width
        expected_shape = (self.output_dim, in_dim)
        if tuple(tm.shape) != expected_shape:
            raise ValueError(
                f"Expected transmission matrix shape {expected_shape} but got {tuple(tm.shape)}"
            )

        if not torch.is_complex(tm):
            raise TypeError("Transmission matrix must be a complex tensor (e.g., complex64)")

        # Extract local shard if TP is enabled
        if self.tp_enabled:
            tm_local = tm[self._row_start:self._row_end, :]
        else:
            tm_local = tm

        tm_local = tm_local.to(device=self.transmission_matrix.device, dtype=self.transmission_matrix.dtype)
        
        # If original was sparse, we might want to convert the input to sparse or vice versa.
        # But usually set_transmission_matrix is used to set a specific (often dense) matrix.
        # We will follow the sparsity of the provided matrix if possible, or just replace it.
        self.transmission_matrix.data = tm_local

    def _image_to_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """将输入图像转换为大致位于 [0, 1] 的幅度张量。
        - 若为 3 通道，使用标准 RGB 权重转换为亮度；
        - 若为其他通道数，在通道维上取平均；
        - 可选：按样本进行 min-max 归一化；
        - 可选：对强度开方以转换为幅度。
        """
        if x.ndim != 4:
            raise ValueError("Input must be a 4D tensor [B, C, H, W]")
        if x.shape[-2] != self.height or x.shape[-1] != self.width:
            raise ValueError(f"Input spatial size must be ({self.height}, {self.width}) but got ({x.shape[-2]}, {x.shape[-1]})")

        x = x.float()
        b, c, _, _ = x.shape

        if c == 3:
            # 亮度权重
            weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            amp = (x * weights).sum(dim=1, keepdim=True)
        elif c == 1:
            amp = x
        else:
            amp = x.mean(dim=1, keepdim=True)

        if self.normalize_input:
            min_val = amp.amin(dim=(2, 3), keepdim=True)
            max_val = amp.amax(dim=(2, 3), keepdim=True)
            amp = (amp - min_val) / (max_val - min_val + self.eps)

        amp = torch.clamp(amp, min=0.0)
        if self.sqrt_amplitude:
            amp = torch.sqrt(amp + self.eps)

        return amp  # 形状 [B, 1, H, W]，实数

    def _apply_activation(self, y: torch.Tensor) -> torch.Tensor:
        """应用选择的激活函数到复数输出 y。
        
        参数
        ----------
        y:
            复数张量，形状为 [B, 1, H, W]
            
        返回
        -------
        torch.Tensor
            实数张量，形状为 [B, 1, H, W]
        """
        # 获取幅度作为基础值
        amplitude = torch.abs(y)**2
        
        # 根据选择的激活函数进行处理
        if self.activation == "abs":
            return amplitude
            
        elif self.activation == "relu":
            return F.relu(amplitude)
            
        elif self.activation == "leaky_relu":
            negative_slope = self.activation_params.get("negative_slope", 0.01)
            return F.leaky_relu(amplitude, negative_slope=negative_slope)
            
        elif self.activation == "sigmoid":
            return torch.sigmoid(amplitude)
            
        elif self.activation == "tanh":
            return torch.tanh(amplitude)
            
        elif self.activation == "elu":
            alpha = self.activation_params.get("alpha", 1.0)
            return F.elu(amplitude, alpha=alpha)
            
        elif self.activation == "softplus":
            beta = self.activation_params.get("beta", 1.0)
            threshold = self.activation_params.get("threshold", 20.0)
            return F.softplus(amplitude, beta=beta, threshold=threshold)
        
        else:
            # 默认使用绝对值
            return amplitude

    def _normalize_negative_values(self, amplitude: torch.Tensor) -> torch.Tensor:
        """对激活函数输出的负数部分进行正则化处理。
        
        参数
        ----------
        amplitude:
            实数张量，可能包含负数
            
        返回
        -------
        torch.Tensor
            正则化后的实数张量，所有值都非负
        """
        # 找到负数部分
        negative_mask = amplitude < 0
        
        if not negative_mask.any():
            # 如果没有负数，直接返回
            return amplitude
            
        # 对负数部分进行正则化处理
        # 方法1：将负数映射到正数范围（例如使用绝对值）
        # 方法2：将负数缩放到 [0, 1] 范围
        # 这里使用方法2：将整个张量缩放到 [0, 1] 范围
        
        # 获取最小值和最大值
        min_val = amplitude.min()
        max_val = amplitude.max()
        
        # 如果所有值都是负数，特殊处理
        if max_val <= 0:
            # 将所有负数映射到 [0, 1] 范围
            normalized = (amplitude - min_val) / (max_val - min_val + self.eps)
        else:
            # 正常情况：将整个范围缩放到 [0, 1]
            normalized = (amplitude - min_val) / (max_val - min_val + self.eps)
        
        return normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数
        ----------
        x:
            形状为 [B, C, H, W] 的输入图像张量。H 与 W 必须与 `input_hw` 一致。

        返回
        -------
        torch.Tensor
            - 当 `return_intensity` 为 False：形状为 [B, H_out, W_out] 的复数张量；
            - 当 `return_intensity` 为 True：形状为 [B, H_out, W_out] 的实数张量（|y|^2）。
        """
        B, C, H, W = x.shape
        amplitude = self._image_to_amplitude(x)  # [B, 1, H, W]

        for l in range(self.num_layers):
            # 构建复数场：amplitude * exp(i * phase_l)
            # torch.polar 对 half CUDA 未实现，这里在极坐标构造处用 float32，然后再按需降精度
            amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + self.eps)
            
            # 对相位参数应用 dropout（仅在训练时）
            phase_l = self.phases[l]
            if self.phase_dropout > 0.0 and self.training:
                # 创建 dropout mask，随机将某些位置的相位设置为0
                # 使用伯努利分布生成 mask，保持被保留元素的期望值不变
                dropout_mask = torch.bernoulli(torch.ones_like(phase_l) * (1.0 - self.phase_dropout))
                # 在训练时，需要缩放以保持期望值：保留的元素除以 (1 - p)
                dropout_mask = dropout_mask / (1.0 - self.phase_dropout)
                phase_l = phase_l * dropout_mask
            
            field = torch.polar(amplitude.float(), phase_l)  # [B, 1, H, W] complex64
            # 若传输矩阵为 complex32，则将场降到 complex32 以节省显存
            if self.transmission_matrix.dtype == getattr(torch, "complex32", torch.complex64):
                field = field.to(getattr(torch, "complex32", torch.complex64))

            # 展平到向量后进行传输矩阵乘法
            field_vec = field.reshape(B, -1)  # [B, H*W]

            if self.transmission_matrix.is_sparse:
                y_local = torch.sparse.mm(self.transmission_matrix, field_vec.T).T
            else:
                y_local = torch.matmul(field_vec, self.transmission_matrix.T)

            # 张量并行：使用自定义 autograd-safe all_gather 组装完整输出
            if self.tp_enabled and dist.is_available() and dist.is_initialized() and self.tp_world_size > 1:
                y = _TPAllGather.apply(
                    y_local,
                    self.rows_per_rank,
                    self.output_dim,
                    self.tp_world_size,
                )
            else:
                y = y_local

            # 输出面自由衍射
            y = y.reshape(B, 1, self.output_height, self.output_width)

            # 幅度作为下一层的输入，应用选择的激活函数
            amplitude = self._apply_activation(y)
            
            # 如果启用了负数正则化，对负数部分进行处理
            if self.normalize_negative:
                amplitude = self._normalize_negative_values(amplitude)
            
            if (self.output_height, self.output_width) != (self.height, self.width):
                amplitude = F.interpolate(
                    amplitude,
                    size=(self.height, self.width),
                    mode="bilinear",
                    align_corners=False,
                )

        if self.return_intensity:
            # 强度：|y|^2 = real^2 + imag^2，然后重塑为二维输出
            y_real = y.real.pow(2) + y.imag.pow(2)
            return y_real.view(B, self.output_height, self.output_width)
        return y.view(B, self.output_height, self.output_width)


class SpectralMixer(nn.Module):
    def __init__(
        self,
        h: int,
        w: int,
        freq_range: float = 0.5,
        mode: str = "phase",
        init_scale: float = 1e-3,
        out_activation: str = "softplus",
    ):
        super().__init__()
        if not (0.0 < float(freq_range) <= 1.0):
            raise ValueError(f"freq_range must be in (0, 1], got {freq_range}")
        if mode not in ("phase", "complex"):
            raise ValueError(f"mode must be 'phase' or 'complex', got {mode}")

        self.h = int(h)
        self.w = int(w)
        self.freq_range = float(freq_range)
        self.mode = mode
        self.out_activation = out_activation

        fh = max(1, int(math.floor(self.h * self.freq_range)))
        fw = max(1, int(math.floor(self.w * self.freq_range)))
        self.fh = fh
        self.fw = fw

        if self.mode == "phase":
            self.phase = nn.Parameter(torch.randn(1, 1, fh, fw) * float(init_scale))
        else:
            self.real = nn.Parameter(torch.randn(1, 1, fh, fw) * float(init_scale))
            self.imag = nn.Parameter(torch.randn(1, 1, fh, fw) * float(init_scale))

    def _build_mask_shifted(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.mode == "phase":
            coeff = self.phase.to(device=device, dtype=torch.float32)
            coeff_c = torch.complex(torch.cos(coeff), torch.sin(coeff))
        else:
            real = self.real.to(device=device, dtype=torch.float32)
            imag = self.imag.to(device=device, dtype=torch.float32)
            coeff_c = torch.complex(real, imag)

        pad_h_total = self.h - self.fh
        pad_w_total = self.w - self.fw
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left

        mask = F.pad(coeff_c, (pad_left, pad_right, pad_top, pad_bottom))
        mask = mask.to(device=device, dtype=torch.complex64)
        return mask

    def _apply_out_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_activation == "none" or self.out_activation is None:
            return x
        if self.out_activation == "relu":
            return F.relu(x)
        if self.out_activation == "softplus":
            return F.softplus(x)
        if self.out_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.out_activation == "abs":
            return torch.abs(x)
        raise ValueError(f"Unknown out_activation: {self.out_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected [B,1,H,W] or [B,H,W], got {tuple(x.shape)}")
        if x.shape[-2] != self.h or x.shape[-1] != self.w:
            raise ValueError(f"Expected spatial {(self.h, self.w)}, got {(x.shape[-2], x.shape[-1])}")

        x_is_complex = torch.is_complex(x)
        x_c = x.to(torch.complex64)
        X = torch.fft.fft2(x_c, dim=(-2, -1))
        Xs = _fftshift2d(X)
        mask = self._build_mask_shifted(device=Xs.device, dtype=Xs.dtype)
        Ys = Xs * mask
        Y = _ifftshift2d(Ys)
        y_c = torch.fft.ifft2(Y, dim=(-2, -1))
        if x_is_complex:
            if self.out_activation not in ("none", None):
                raise ValueError("out_activation must be 'none' when applying spectral mixer in complex field domain")
            return y_c[:, 0]
        y = y_c.real
        y = self._apply_out_activation(y)
        return y[:, 0]


class PSFMixer(nn.Module):
    def __init__(
        self,
        out_activation: str = "softplus",
        coherent: bool = False,
        fourier: bool = True,
        freq_range: float = 0.5,
        wavelength: float = 532e-9,
    ):
        super().__init__()
        act = None
        if out_activation == "relu":
            act = F.relu
        elif out_activation == "softplus":
            act = F.softplus
        elif out_activation == "sigmoid":
            act = torch.sigmoid
        elif out_activation == "abs":
            act = torch.abs
        elif out_activation == "none" or out_activation is None:
            act = None
        else:
            raise ValueError(f"Unknown out_activation: {out_activation}")



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        y = self.layer(x)
        return y[:, 0]


class SeparableMixer(nn.Module):
    def __init__(
        self,
        h: int,
        w: int,
        kernel_size: int = 9,
        out_activation: str = "softplus",
    ):
        super().__init__()
        self.h = int(h)
        self.w = int(w)
        k = int(kernel_size)
        pad = k // 2
        self.row = nn.Conv1d(1, 1, kernel_size=k, padding=pad, bias=False)
        self.col = nn.Conv1d(1, 1, kernel_size=k, padding=pad, bias=False)
        nn.init.normal_(self.row.weight, std=0.01)
        nn.init.normal_(self.col.weight, std=0.01)
        self.out_activation = out_activation

    def _apply_out_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_activation == "none" or self.out_activation is None:
            return x
        if self.out_activation == "relu":
            return F.relu(x)
        if self.out_activation == "softplus":
            return F.softplus(x)
        if self.out_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.out_activation == "abs":
            return torch.abs(x)
        raise ValueError(f"Unknown out_activation: {self.out_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x[:, 0]
        if x.ndim != 3:
            raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(x.shape)}")
        if x.shape[-2] != self.h or x.shape[-1] != self.w:
            raise ValueError(f"Expected spatial {(self.h, self.w)}, got {(x.shape[-2], x.shape[-1])}")

        b, h, w = x.shape
        xr = x.reshape(b * h, 1, w)
        yr = self.row(xr).reshape(b, h, w)
        xc = yr.transpose(1, 2).reshape(b * w, 1, h)
        yc = self.col(xc).reshape(b, w, h).transpose(1, 2)
        yc = self._apply_out_activation(yc)
        return yc


class ScatterMixerNetwork(nn.Module):
    def __init__(
        self,
        input_hw: tuple[int, int] = (256, 256),
        output_hw: tuple[int, int] = (256, 256),
        num_layers: int = 5,
        return_intensity: bool = True,
        tmatrix_compute_dtype: torch.dtype | str | None = "bf16",
        tmatrix_sparsity: float = 0.98,
        tp_enabled: bool = False,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        activation: str = "abs",
        activation_params: dict | None = None,
        normalize_negative: bool = False,
        phase_dropout: float = 0.0,
        mixer_type: str = "spectral",
        mixer_out_activation: str = "softplus",
        mixer_freq_range: float = 0.5,
        mixer_mode: str = "phase",
        mixer_init_scale: float = 1e-3,
        mixer_kernel_size: int = 9,
        mixer_residual: bool = True,
        mixer_gate_init: float = 0.0,
        mixer_norm: str = "log1p",
        mixer_norm_eps: float = 1e-6,
        mixer_domain: str = "intensity",
    ):
        super().__init__()
        tmatrix_dtype = tmatrix_compute_dtype
        if isinstance(tmatrix_compute_dtype, str):
            if tmatrix_compute_dtype == "fp32":
                tmatrix_dtype = None
            elif tmatrix_compute_dtype == "bf16":
                tmatrix_dtype = torch.bfloat16
            elif tmatrix_compute_dtype == "fp16":
                tmatrix_dtype = torch.float16
            else:
                raise ValueError(f"Unknown tmatrix_compute_dtype string: {tmatrix_compute_dtype}")

        self.backbone = ScatterNeuralNetwork(
            input_hw=input_hw,
            output_hw=output_hw,
            num_layers=int(num_layers),
            return_intensity=bool(return_intensity),
            tmatrix_compute_dtype=tmatrix_dtype,
            tmatrix_sparsity=float(tmatrix_sparsity),
            tp_enabled=bool(tp_enabled),
            tp_rank=int(tp_rank),
            tp_world_size=int(tp_world_size),
            activation=str(activation),
            activation_params=activation_params,
            normalize_negative=bool(normalize_negative),
            phase_dropout=float(phase_dropout),
        )

        self.mixer_type = mixer_type
        self.mixer_residual = bool(mixer_residual)
        self.mixer_norm = str(mixer_norm)
        self.mixer_norm_eps = float(mixer_norm_eps)
        self.mixer_domain = str(mixer_domain)
        h_out, w_out = int(output_hw[0]), int(output_hw[1])
        if mixer_type == "none" or mixer_type is None:
            self.mixer = None
        elif mixer_type == "spectral":
            self.mixer = SpectralMixer(
                h=h_out,
                w=w_out,
                freq_range=float(mixer_freq_range),
                mode=str(mixer_mode),
                init_scale=float(mixer_init_scale),
                out_activation=str(mixer_out_activation),
            )
        elif mixer_type == "psf":
            self.mixer = PSFMixer(
                out_activation=str(mixer_out_activation),
                coherent=False,
                fourier=True,
                freq_range=float(mixer_freq_range),
            )
        elif mixer_type == "separable":
            self.mixer = SeparableMixer(
                h=h_out,
                w=w_out,
                kernel_size=int(mixer_kernel_size),
                out_activation=str(mixer_out_activation),
            )
        else:
            raise ValueError(f"Unknown mixer_type: {mixer_type}")

        if self.mixer_domain not in ("intensity", "field"):
            raise ValueError(f"Unknown mixer_domain: {self.mixer_domain}")
        if self.mixer_domain == "field" and (self.mixer is not None) and self.mixer_type != "spectral":
            raise ValueError("mixer_domain='field' currently supports mixer_type='spectral' only")

        if self.mixer is None or not self.mixer_residual:
            self.mixer_gate = None
        else:
            self.mixer_gate = nn.Parameter(torch.tensor(float(mixer_gate_init), dtype=torch.float32))

    def _normalize_for_mixer(self, x: torch.Tensor) -> torch.Tensor:
        name = self.mixer_norm
        eps = self.mixer_norm_eps
        if name == "none" or name is None:
            return x

        if not torch.is_complex(x):
            if name == "log1p":
                return torch.log1p(torch.clamp(x, min=0.0))
            if name == "standardize":
                mean = x.mean(dim=(-2, -1), keepdim=True)
                std = x.std(dim=(-2, -1), keepdim=True)
                return (x - mean) / (std + eps)
            if name == "minmax":
                x_min = x.amin(dim=(-2, -1), keepdim=True)
                x_max = x.amax(dim=(-2, -1), keepdim=True)
                return (x - x_min) / (x_max - x_min + eps)
            raise ValueError(f"Unknown mixer_norm: {name}")

        mag = torch.abs(x)
        if name == "log1p":
            scale = torch.log1p(mag) / (mag + eps)
            return x * scale
        if name == "standardize":
            std = mag.std(dim=(-2, -1), keepdim=True)
            return x / (std + eps)
        if name == "minmax":
            m_min = mag.amin(dim=(-2, -1), keepdim=True)
            m_max = mag.amax(dim=(-2, -1), keepdim=True)
            m01 = (mag - m_min) / (m_max - m_min + eps)
            scale = m01 / (mag + eps)
            return x * scale
        raise ValueError(f"Unknown mixer_norm: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)
        if self.mixer is None:
            return y

        if self.mixer_domain == "field":
            if not torch.is_complex(y):
                y = y.to(torch.complex64)
            y_for_mixer = self._normalize_for_mixer(y)
            if self.mixer_residual and self.mixer_gate is not None:
                return y + self.mixer_gate * self.mixer(y_for_mixer)
            return self.mixer(y_for_mixer)

        y_intensity = y.real.pow(2) + y.imag.pow(2) if torch.is_complex(y) else y
        y_for_mixer = self._normalize_for_mixer(y_intensity)
        if self.mixer_residual and self.mixer_gate is not None:
            return y_intensity + self.mixer_gate * self.mixer(y_for_mixer)
        return self.mixer(y_for_mixer)

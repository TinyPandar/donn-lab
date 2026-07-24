from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatMulConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            k_h, k_w = kernel_size, kernel_size
        else:
            k_h, k_w = int(kernel_size[0]), int(kernel_size[1])
        if isinstance(stride, int):
            s_h, s_w = stride, stride
        else:
            s_h, s_w = int(stride[0]), int(stride[1])
        if isinstance(padding, int):
            p_h, p_w = padding, padding
        else:
            p_h, p_w = int(padding[0]), int(padding[1])
        if isinstance(dilation, int):
            d_h, d_w = dilation, dilation
        else:
            d_h, d_w = int(dilation[0]), int(dilation[1])

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.k_h, self.k_w = int(k_h), int(k_w)
        self.s_h, self.s_w = int(s_h), int(s_w)
        self.p_h, self.p_w = int(p_h), int(p_w)
        self.d_h, self.d_w = int(d_h), int(d_w)

        weight = torch.empty(self.out_channels, self.in_channels, self.k_h, self.k_w)
        nn.init.kaiming_uniform_(weight, a=5**0.5)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {c}")
        cols = F.unfold(
            x,
            kernel_size=(self.k_h, self.k_w),
            dilation=(self.d_h, self.d_w),
            padding=(self.p_h, self.p_w),
            stride=(self.s_h, self.s_w),
        )
        l = int(cols.shape[-1])
        cols_t = cols.transpose(1, 2)
        w_mat = self.weight.reshape(self.out_channels, -1).t()
        out = cols_t.matmul(w_mat)
        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)
        out = out.transpose(1, 2)
        out_h = (h + 2 * self.p_h - self.d_h * (self.k_h - 1) - 1) // self.s_h + 1
        out_w = (w + 2 * self.p_w - self.d_w * (self.k_w - 1) - 1) // self.s_w + 1
        if out_h * out_w != l:
            out_h = int(out.shape[-1] // max(out_w, 1))
        return out.reshape(b, self.out_channels, int(out_h), int(out_w))


class MatMulCNN(nn.Module):
    def __init__(self, input_hw: tuple[int, int], output_hw: tuple[int, int]):
        super().__init__()
        h_out, w_out = int(output_hw[0]), int(output_hw[1])
        self.h_out = h_out
        self.w_out = w_out
        self.layers = nn.ModuleList(
            [
                MatMulConv2d(1, 4, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
        )
        self.head = nn.Sequential(
            MatMulConv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MatMulConv2d(4, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected input shape [B,3,H,W]")
        x = x.float()
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        amp = (x * weights).sum(dim=1, keepdim=True)
        amp = torch.clamp(amp, min=0.0)
        out = amp
        for layer in self.layers:
            out = layer(out)
        feat = out
        heat = self.head(feat)
        heat = F.softplus(heat)
        heat = F.interpolate(heat, size=(self.h_out, self.w_out), mode="bilinear", align_corners=False)
        return heat[:, 0], feat


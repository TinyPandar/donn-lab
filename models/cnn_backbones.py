from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(name: str, channels: int) -> nn.Module:
    name = str(name).lower()
    if name in ("none", "identity", ""):
        return nn.Identity()
    if name in ("batchnorm", "bn", "batch_norm"):
        return nn.BatchNorm2d(channels)
    if name in ("instancenorm", "in", "instance_norm"):
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unsupported cnn_encoder_norm: {name}")


def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name in ("none", "identity", ""):
        return nn.Identity()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("leaky_relu", "leaky"):
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported cnn_encoder_activation: {name}")


class HybridCNNEncoder(nn.Module):
    """Small electronic encoder that emits a single non-negative amplitude map."""

    def __init__(
        self,
        input_hw: tuple[int, int],
        *,
        in_channels: int = 3,
        depth: int = 1,
        width: int = 8,
        norm: str = "batchnorm",
        activation: str = "relu",
        out_activation: str = "softplus",
    ) -> None:
        super().__init__()
        self.h_in, self.w_in = int(input_hw[0]), int(input_hw[1])
        self.depth = int(depth)
        self.width = int(width)
        self.out_activation = str(out_activation).lower()
        self.eps = 1e-8

        if self.depth not in (1, 2):
            raise ValueError(f"cnn_encoder_depth must be 1 or 2, got {self.depth}")
        if self.width <= 0:
            raise ValueError(f"cnn_encoder_width must be positive, got {self.width}")

        blocks: list[nn.Module] = []
        curr_channels = int(in_channels)
        for _ in range(self.depth):
            blocks.extend(
                [
                    nn.Conv2d(curr_channels, self.width, kernel_size=3, padding=1),
                    _make_norm(norm, self.width),
                    _make_activation(activation),
                ]
            )
            curr_channels = self.width
        self.body = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(curr_channels, 1, kernel_size=1)

    def _apply_out_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_activation == "softplus":
            return F.softplus(x)
        if self.out_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.out_activation == "relu":
            return F.relu(x)
        if self.out_activation in ("none", "identity", ""):
            return x
        raise ValueError(f"Unsupported cnn_encoder_out_activation: {self.out_activation}")

    def _normalize_spatial_per_sample(self, x: torch.Tensor) -> torch.Tensor:
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        amp = self.proj(self.body(x.float()))
        amp = self._apply_out_activation(amp)
        amp = self._normalize_spatial_per_sample(amp)
        if amp.shape[-2:] != (self.h_in, self.w_in):
            amp = F.interpolate(amp, size=(self.h_in, self.w_in), mode="bilinear", align_corners=False)
        return amp


class _TinyHeatmapCNN(nn.Module):
    def __init__(self, output_hw: tuple[int, int], width: int = 32):
        super().__init__()
        self.h_out, self.w_out = int(output_hw[0]), int(output_hw[1])
        self.body = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.body(x.float())
        heat = self.head(feat)
        heat = F.softplus(heat)
        heat = F.interpolate(heat, size=(self.h_out, self.w_out), mode="bilinear", align_corners=False)
        return heat[:, 0], feat


class SimpleCNN(nn.Module):
    def __init__(self, input_hw: tuple[int, int], output_hw: tuple[int, int]):
        super().__init__()
        _ = input_hw
        self.net = _TinyHeatmapCNN(output_hw=output_hw, width=24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heat, _ = self.net(x)
        return heat


class SimpleDistCNN(nn.Module):
    def __init__(self, input_hw: tuple[int, int], output_hw: tuple[int, int]):
        super().__init__()
        _ = input_hw
        self.net = _TinyHeatmapCNN(output_hw=output_hw, width=24)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ResNetHeatmap(nn.Module):
    def __init__(
        self,
        input_hw: tuple[int, int],
        output_hw: tuple[int, int],
        variant: str = "resnet18",
        pretrained: bool = False,
    ):
        super().__init__()
        _ = input_hw
        self.h_out, self.w_out = int(output_hw[0]), int(output_hw[1])
        self.variant = str(variant)
        self.pretrained = bool(pretrained)
        self._build_backbone()

    def _build_backbone(self) -> None:
        try:
            from torchvision import models

            weights = None
            if self.pretrained:
                if self.variant == "resnet18":
                    weights = models.ResNet18_Weights.DEFAULT
                elif self.variant == "resnet34":
                    weights = models.ResNet34_Weights.DEFAULT
                elif self.variant == "resnet50":
                    weights = models.ResNet50_Weights.DEFAULT
            if self.variant == "resnet34":
                backbone = models.resnet34(weights=weights)
                out_ch = 512
            elif self.variant == "resnet50":
                backbone = models.resnet50(weights=weights)
                out_ch = 2048
            else:
                backbone = models.resnet18(weights=weights)
                out_ch = 512
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            self.head = nn.Sequential(
                nn.Conv2d(out_ch, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1, kernel_size=1),
            )
        except Exception:
            # Keep repository self-contained when torchvision is unavailable.
            self.backbone = _TinyHeatmapCNN(output_hw=(self.h_out, self.w_out), width=48).body
            self.head = nn.Sequential(
                nn.Conv2d(48, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x.float())
        heat = self.head(feat)
        heat = F.softplus(heat)
        heat = F.interpolate(heat, size=(self.h_out, self.w_out), mode="bilinear", align_corners=False)
        return heat[:, 0]

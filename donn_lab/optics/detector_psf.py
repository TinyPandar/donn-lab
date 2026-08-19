"""Effective detector point-spread functions.

The operator in this module is intentionally an intensity-plane correction,
not an additional coherent propagation plane.  It approximates the residual
camera/registration blur observed between measured-TM predictions and real
captures while remaining fixed, differentiable, and checkpoint-configurable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_detector_psf_sigma(value: float) -> float:
    sigma = float(value)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("detector_psf_sigma must be finite and non-negative")
    return sigma


class GaussianIntensityPSF(nn.Module):
    """Apply a fixed Gaussian PSF to ``[B,C,H,W]`` detector intensities.

    Reflect padding matches the offline calibration scan and avoids inventing a
    dark border at the cropped camera ROI.  A zero sigma is an exact no-op.
    """

    def __init__(self, sigma: float = 0.0, truncate: float = 4.0) -> None:
        super().__init__()
        self.sigma = validate_detector_psf_sigma(sigma)
        self.truncate = float(truncate)
        if not math.isfinite(self.truncate) or self.truncate <= 0.0:
            raise ValueError("truncate must be finite and positive")

        if self.sigma == 0.0:
            self.radius = 0
            kernel = torch.ones((1, 1, 1, 1), dtype=torch.float32)
        else:
            # Same radius convention as scipy.ndimage.gaussian_filter.
            self.radius = max(1, int(self.truncate * self.sigma + 0.5))
            coordinate = torch.arange(
                -self.radius, self.radius + 1, dtype=torch.float32
            )
            one_dimensional = torch.exp(
                -0.5 * torch.square(coordinate / self.sigma)
            )
            one_dimensional /= one_dimensional.sum()
            kernel = torch.outer(one_dimensional, one_dimensional)
            kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        self.register_buffer("kernel", kernel, persistent=False)

    @property
    def kernel_size(self) -> int:
        return int(2 * self.radius + 1)

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        if self.radius == 0:
            return intensity
        if intensity.ndim != 4:
            raise ValueError("detector intensity must have shape [B,C,H,W]")
        if intensity.shape[-2] <= self.radius or intensity.shape[-1] <= self.radius:
            raise ValueError(
                "detector image is too small for the configured PSF radius {}".format(
                    self.radius
                )
            )
        channels = int(intensity.shape[1])
        kernel = self.kernel.to(device=intensity.device, dtype=intensity.dtype)
        kernel = kernel.expand(channels, 1, self.kernel_size, self.kernel_size)
        padded = F.pad(
            intensity,
            (self.radius, self.radius, self.radius, self.radius),
            mode="reflect",
        )
        return F.conv2d(padded, kernel, groups=channels)


__all__ = ["GaussianIntensityPSF", "validate_detector_psf_sigma"]

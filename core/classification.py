from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_intensity(pred: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(pred):
        pred = pred.real.square() + pred.imag.square()
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if pred.ndim != 3:
        raise ValueError(f"Classification expects [B,H,W], got {tuple(pred.shape)}")
    return pred


@dataclass(frozen=True)
class DetectorRegion:
    class_index: int
    top: int
    left: int
    height: int
    width: int


def build_detector_regions(
    *,
    output_hw: tuple[int, int],
    num_classes: int,
    grid_rows: int,
    grid_cols: int,
    roi_hw: tuple[int, int],
    margin: int,
) -> list[DetectorRegion]:
    h, w = int(output_hw[0]), int(output_hw[1])
    roi_h, roi_w = int(roi_hw[0]), int(roi_hw[1])
    rows, cols = int(grid_rows), int(grid_cols)
    margin = int(margin)
    if min(h, w, roi_h, roi_w, rows, cols, num_classes) <= 0:
        raise ValueError("Detector dimensions and num_classes must be positive")
    if num_classes > rows * cols:
        raise ValueError("num_classes cannot exceed grid_rows * grid_cols")
    available_h, available_w = h - 2 * margin, w - 2 * margin
    if available_h <= 0 or available_w <= 0:
        raise ValueError("detector_margin leaves no usable output area")
    cell_h, cell_w = available_h / rows, available_w / cols
    if roi_h > cell_h or roi_w > cell_w:
        raise ValueError(
            f"ROI {(roi_h, roi_w)} does not fit detector cell "
            f"({cell_h:.1f}, {cell_w:.1f})"
        )

    regions: list[DetectorRegion] = []
    for class_index in range(num_classes):
        row, col = divmod(class_index, cols)
        center_y = margin + (row + 0.5) * cell_h
        center_x = margin + (col + 0.5) * cell_w
        top = int(round(center_y - roi_h / 2.0))
        left = int(round(center_x - roi_w / 2.0))
        top = min(max(top, 0), h - roi_h)
        left = min(max(left, 0), w - roi_w)
        regions.append(DetectorRegion(class_index, top, left, roi_h, roi_w))
    return regions


class DetectorRegionReadout(nn.Module):
    """Convert a camera intensity map into class logits using fixed optical ROIs."""

    def __init__(
        self,
        *,
        output_hw: tuple[int, int],
        num_classes: int = 10,
        grid_rows: int = 2,
        grid_cols: int = 5,
        roi_hw: tuple[int, int] = (16, 16),
        margin: int = 12,
        log_energy: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.output_hw = (int(output_hw[0]), int(output_hw[1]))
        self.log_energy = bool(log_energy)
        self.eps = float(eps)
        self.regions = build_detector_regions(
            output_hw=self.output_hw,
            num_classes=int(num_classes),
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
            roi_hw=(int(roi_hw[0]), int(roi_hw[1])),
            margin=int(margin),
        )

        masks = torch.zeros(len(self.regions), *self.output_hw, dtype=torch.float32)
        for region in self.regions:
            masks[
                region.class_index,
                region.top : region.top + region.height,
                region.left : region.left + region.width,
            ] = 1.0
        self.register_buffer("masks", masks, persistent=False)
        self.register_buffer("areas", masks.sum(dim=(1, 2)).clamp_min(1.0), persistent=False)

    def forward(self, pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intensity = to_intensity(pred).clamp_min(0.0)
        if tuple(intensity.shape[-2:]) != self.output_hw:
            raise ValueError(
                f"Readout expects output {self.output_hw}, got {tuple(intensity.shape[-2:])}"
            )
        masks = self.masks.to(dtype=intensity.dtype)
        region_sums = torch.einsum("bhw,khw->bk", intensity, masks)
        mean_energies = region_sums / self.areas.to(dtype=intensity.dtype).unsqueeze(0)
        logits = torch.log(mean_energies + self.eps) if self.log_energy else mean_energies
        total_energy = intensity.sum(dim=(1, 2)).clamp_min(self.eps)
        efficiency = region_sums.sum(dim=1) / total_energy
        return logits, mean_energies, efficiency


def classification_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    efficiency: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    efficiency_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = targets.reshape(-1).long()
    ce = F.cross_entropy(logits, labels, label_smoothing=float(label_smoothing))
    efficiency_loss = 1.0 - efficiency.mean()
    loss = ce + float(efficiency_weight) * efficiency_loss
    return loss, ce, efficiency_loss


@torch.no_grad()
def classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    efficiency: torch.Tensor,
) -> dict[str, float]:
    labels = targets.reshape(-1).long()
    pred_classes = logits.argmax(dim=1)
    accuracy = (pred_classes == labels).float().mean()
    probs = F.softmax(logits, dim=1)
    correct_prob = probs.gather(1, labels[:, None]).squeeze(1)
    wrong_probs = probs.clone()
    wrong_probs.scatter_(1, labels[:, None], -1.0)
    margin = correct_prob - wrong_probs.max(dim=1).values
    return {
        "accuracy": float(accuracy.cpu()),
        "detector_efficiency": float(efficiency.mean().cpu()),
        "class_margin": float(margin.mean().cpu()),
    }


__all__ = [
    "DetectorRegion",
    "DetectorRegionReadout",
    "build_detector_regions",
    "classification_metrics",
    "classification_objective",
    "to_intensity",
]

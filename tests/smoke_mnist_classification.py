from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.schema import ExperimentConfig
from core.classification import DetectorRegionReadout
from models.factory import DistContext
from pipelines.classification import ClassificationPipeline


class TinyDetectorModel(nn.Module):
    def __init__(self, output_hw: tuple[int, int]) -> None:
        super().__init__()
        self.detector_logits = nn.Parameter(torch.zeros(1, *output_hw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample_gain = 1.0 + 0.01 * x.mean(dim=(1, 2, 3), keepdim=True)
        return F.softplus(self.detector_logits) * sample_gain.reshape(-1, 1, 1)


def main() -> None:
    output_hw = (32, 40)
    readout = DetectorRegionReadout(
        output_hw=output_hw,
        num_classes=10,
        grid_rows=2,
        grid_cols=5,
        roi_hw=(4, 4),
        margin=2,
    )
    intensity = torch.full((1, *output_hw), 1e-4)
    target_region = readout.regions[7]
    intensity[
        0,
        target_region.top : target_region.top + target_region.height,
        target_region.left : target_region.left + target_region.width,
    ] = 10.0
    logits, _energy, efficiency = readout(intensity)
    assert logits.shape == (1, 10)
    assert int(logits.argmax(dim=1).item()) == 7
    assert 0.0 < float(efficiency.item()) <= 1.0

    cfg = ExperimentConfig()
    cfg.pipeline = "classification"
    cfg.dataset = "mnist"
    cfg.data.dataset = "mnist"
    cfg.data.mnist_target_mode = "class"
    cfg.data.h_out, cfg.data.w_out = output_hw
    cfg.classification.roi_h = 4
    cfg.classification.roi_w = 4
    cfg.classification.detector_margin = 2
    cfg.loss.label_smoothing = 0.0

    model = TinyDetectorModel(output_hw)
    pipeline = ClassificationPipeline()
    pipeline.validate_config(cfg)
    pipeline.setup(cfg, model, torch.device("cpu"), DistContext(False, 0, 1))
    x = torch.rand(2, 1, 24, 32)
    labels = torch.tensor([0, 9], dtype=torch.long)
    out = pipeline.training_step((x, labels), model, cfg)
    assert torch.isfinite(out.loss)
    assert 0.0 <= out.metrics["accuracy"] <= 1.0
    out.loss.backward()
    assert model.detector_logits.grad is not None
    assert torch.isfinite(model.detector_logits.grad).all()
    print("MNIST classification smoke test passed")


if __name__ == "__main__":
    main()

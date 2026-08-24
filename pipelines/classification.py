from __future__ import annotations

import torch

from config.schema import ExperimentConfig
from core.classification import (
    DetectorRegionReadout,
    classification_metrics,
    classification_objective,
)
from models.factory import DistContext
from .base import StepOutput, TrainingPipeline


class ClassificationPipeline(TrainingPipeline):
    name = "classification"

    def __init__(self) -> None:
        self.readout: DetectorRegionReadout | None = None

    def validate_config(self, cfg: ExperimentConfig) -> None:
        if str(cfg.dataset).lower() == "mnist" and str(cfg.data.mnist_target_mode).lower() != "class":
            raise ValueError("MNIST classification requires data.mnist_target_mode: class")
        if int(cfg.classification.num_classes) != 10 and str(cfg.dataset).lower() == "mnist":
            raise ValueError("MNIST requires classification.num_classes: 10")

    def setup(
        self,
        cfg: ExperimentConfig,
        model: torch.nn.Module,
        device: torch.device,
        dist_ctx: DistContext,
    ) -> None:
        _ = model, dist_ctx
        c = cfg.classification
        self.readout = DetectorRegionReadout(
            output_hw=(int(cfg.data.h_out), int(cfg.data.w_out)),
            num_classes=int(c.num_classes),
            grid_rows=int(c.grid_rows),
            grid_cols=int(c.grid_cols),
            roi_hw=(int(c.roi_h), int(c.roi_w)),
            margin=int(c.detector_margin),
            log_energy=bool(c.log_energy),
        ).to(device)

    def _step(self, batch, model: torch.nn.Module, cfg: ExperimentConfig) -> StepOutput:
        if self.readout is None:
            raise RuntimeError("ClassificationPipeline.setup() has not been called")
        x_batch, labels = batch
        pred = self._unwrap_pred(model(x_batch))
        logits, _energies, efficiency = self.readout(pred)
        loss, ce, efficiency_loss = classification_objective(
            logits,
            labels,
            efficiency,
            label_smoothing=float(cfg.loss.label_smoothing),
            efficiency_weight=float(cfg.classification.efficiency_weight),
        )
        metrics = {
            "task_loss": float(ce.detach().cpu()),
            "efficiency_loss": float(efficiency_loss.detach().cpu()),
        }
        metrics.update(classification_metrics(logits, labels, efficiency))
        return StepOutput(loss=loss, metrics=metrics)

    def training_step(self, batch, model: torch.nn.Module, cfg: ExperimentConfig) -> StepOutput:
        return self._step(batch, model, cfg)

    def validation_step(self, batch, model: torch.nn.Module, cfg: ExperimentConfig) -> StepOutput:
        out = self._step(batch, model, cfg)
        out.metrics["val_loss"] = float(out.loss.detach().cpu())
        return out


__all__ = ["ClassificationPipeline"]

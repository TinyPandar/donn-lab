from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from config.schema import ExperimentConfig
from core.losses import compute_argmax_coord_metrics, compute_loss_by_name
from models.factory import DistContext, create_registered_model


@dataclass
class StepOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


class TrainingPipeline:
    name = "base"

    def validate_config(self, cfg: ExperimentConfig) -> None:
        _ = cfg

    def build_model(self, cfg: ExperimentConfig, device: torch.device, dist_ctx: DistContext):
        return create_registered_model(cfg.model, cfg, device, dist_ctx)

    def setup(self, cfg: ExperimentConfig, model: torch.nn.Module, device: torch.device, dist_ctx: DistContext) -> None:
        _ = cfg, model, device, dist_ctx

    def _unwrap_pred(self, out: Any) -> torch.Tensor:
        if isinstance(out, (tuple, list)):
            if len(out) == 0:
                raise RuntimeError("model returned empty tuple/list")
            out = out[0]
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"Expected Tensor output, got {type(out).__name__}")
        return out

    def training_step(self, batch, model: torch.nn.Module, cfg: ExperimentConfig) -> StepOutput:
        x_batch, coords_batch = batch
        pred = self._unwrap_pred(model(x_batch))
        loss = compute_loss_by_name(pred, coords_batch, cfg)
        return StepOutput(loss=loss, metrics={"task_loss": float(loss.detach().cpu())})

    def validation_step(self, batch, model: torch.nn.Module, cfg: ExperimentConfig) -> StepOutput:
        x_batch, coords_batch = batch
        pred = self._unwrap_pred(model(x_batch))
        loss = compute_loss_by_name(pred, coords_batch, cfg, test=True)
        metrics = {"val_loss": float(loss.detach().cpu())}
        metrics.update(compute_argmax_coord_metrics(pred, coords_batch))
        return StepOutput(loss=loss, metrics=metrics)

    def on_epoch_end(self, epoch: int, model: torch.nn.Module, cfg: ExperimentConfig) -> dict[str, float]:
        _ = epoch, model, cfg
        return {}

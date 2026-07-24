from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from config.schema import ExperimentConfig
from core.losses import compute_argmax_coord_metrics, compute_loss_by_name
from models.factory import DistContext, create_registered_model
from core.tensor_utils import kl_divergence_loss

from .base import StepOutput, TrainingPipeline


def _to_intensity(pred: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(pred):
        return pred.real.pow(2) + pred.imag.pow(2)
    return pred


def _forward_with_features(model: torch.nn.Module, x: torch.Tensor):
    out = model(x)
    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            raise RuntimeError("model returned empty tuple/list")
        if len(out) == 1:
            return out[0], None
        return out[0], out[1]
    return out, None


def _make_kd_feature(feat, pred_intensity: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    if feat is None:
        feat = pred_intensity
    if isinstance(feat, torch.Tensor) and torch.is_complex(feat):
        feat = feat.real.pow(2) + feat.imag.pow(2)
    if not isinstance(feat, torch.Tensor):
        raise TypeError("feat must be tensor or None")
    if feat.ndim == 2:
        return feat.float()
    if feat.ndim == 3:
        return feat.reshape(int(feat.shape[0]), -1).float()
    if feat.ndim == 4:
        if feat.shape[1] != 1:
            feat = feat.mean(dim=1, keepdim=True)
        feat = F.interpolate(feat, size=out_hw, mode="bilinear", align_corners=False)
        return feat[:, 0].reshape(int(feat.shape[0]), -1).float()
    return feat.reshape(int(feat.shape[0]), -1).float()


class DistillPipeline(TrainingPipeline):
    name = "distill"

    def __init__(self) -> None:
        super().__init__()
        self.teacher: torch.nn.Module | None = None
        self.warned_feat_mismatch = False

    def setup(self, cfg: ExperimentConfig, model: torch.nn.Module, device: torch.device, dist_ctx: DistContext) -> None:
        _ = model, dist_ctx
        teacher_model_name = str(cfg.distill.teacher_model).lower()
        if teacher_model_name in ("", "none"):
            self.teacher = None
            return

        teacher = create_registered_model(teacher_model_name, cfg, device, DistContext(False, 0, 1))
        if cfg.distill.teacher_ckpt:
            ckpt = torch.load(cfg.distill.teacher_ckpt, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            stripped = {}
            if isinstance(state, dict):
                for k, v in state.items():
                    stripped[k[7:] if isinstance(k, str) and k.startswith("module.") else k] = v
                state = stripped
            missing, unexpected = teacher.load_state_dict(state, strict=False)
            if len(missing) > 0 or len(unexpected) > 0:
                print(f"[distill] teacher checkpoint loaded with missing={len(missing)} unexpected={len(unexpected)}")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        self.teacher = teacher

    def training_step(self, batch, model: torch.nn.Module, cfg: ExperimentConfig) -> StepOutput:
        x_batch, coords_batch = batch
        pred, feat = _forward_with_features(model, x_batch)
        if self.teacher is None:
            loss = compute_loss_by_name(pred, coords_batch, cfg)
            return StepOutput(loss=loss, metrics={"task_loss": float(loss.detach().cpu())})

        with torch.no_grad():
            pred_t, feat_t = _forward_with_features(self.teacher, x_batch)

        task_w = float(cfg.distill.task_w)
        kd_pred_w = float(cfg.distill.kd_pred_w)
        kd_feat_w = float(cfg.distill.kd_feat_w)
        kd_mode = str(cfg.distill.kd_mode).lower()
        kd_temperature = max(float(cfg.distill.kd_temperature), 1e-6)

        pred_i = _to_intensity(pred)
        pred_t_i = _to_intensity(pred_t).to(dtype=pred_i.dtype)
        if pred_t_i.shape != pred_i.shape:
            pred_t_i = F.interpolate(
                pred_t_i.unsqueeze(1),
                size=(int(pred_i.shape[-2]), int(pred_i.shape[-1])),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        if kd_mode == "kl":
            kd_pred = kl_divergence_loss(pred_i.reshape(pred_i.shape[0], -1) / kd_temperature, pred_t_i.reshape(pred_i.shape[0], -1) / kd_temperature) * (
                kd_temperature * kd_temperature
            )
        else:
            kd_pred = F.mse_loss(pred_i, pred_t_i.detach())

        feat_vec = _make_kd_feature(feat, pred_i, (int(cfg.data.h_out), int(cfg.data.w_out)))
        feat_t_vec = _make_kd_feature(feat_t, pred_t_i, (int(cfg.data.h_out), int(cfg.data.w_out)))
        if feat_vec.shape == feat_t_vec.shape:
            kd_feat = F.mse_loss(feat_vec, feat_t_vec.detach())
        else:
            kd_feat = torch.zeros((), device=pred_i.device, dtype=pred_i.dtype)
            if not self.warned_feat_mismatch:
                print(f"[distill] Skip kd_feat due to shape mismatch student={tuple(feat_vec.shape)} teacher={tuple(feat_t_vec.shape)}")
                self.warned_feat_mismatch = True

        if task_w > 0:
            task_loss = compute_loss_by_name(pred, coords_batch, cfg)
        else:
            task_loss = torch.zeros((), device=pred_i.device, dtype=pred_i.dtype)
        loss = task_w * task_loss + kd_pred_w * kd_pred + kd_feat_w * kd_feat
        return StepOutput(
            loss=loss,
            metrics={
                "task_loss": float(task_loss.detach().cpu()),
                "kd_pred": float(kd_pred.detach().cpu()),
                "kd_feat": float(kd_feat.detach().cpu()),
            },
        )

    def validation_step(self, batch, model: torch.nn.Module, cfg: ExperimentConfig) -> StepOutput:
        x_batch, coords_batch = batch
        pred, _ = _forward_with_features(model, x_batch)
        loss = compute_loss_by_name(pred, coords_batch, cfg, test=True)
        metrics = {"val_loss": float(loss.detach().cpu())}
        metrics.update(compute_argmax_coord_metrics(pred, coords_batch))
        return StepOutput(loss=loss, metrics=metrics)

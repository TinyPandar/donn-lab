from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from core.tensor_utils import kl_divergence_loss


def _to_intensity(pred: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(pred):
        return pred.real.pow(2) + pred.imag.pow(2)
    return pred


def _unwrap_prediction(pred: Any) -> torch.Tensor:
    if isinstance(pred, (tuple, list)):
        if len(pred) == 0:
            raise ValueError("Empty model output tuple/list")
        pred = pred[0]
    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor prediction, got {type(pred).__name__}")
    return pred


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    node = cfg
    for part in key.split("."):
        if hasattr(node, part):
            node = getattr(node, part)
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


def _flatten_logits(pred: torch.Tensor) -> torch.Tensor:
    p = _to_intensity(pred)
    if p.ndim != 3:
        raise ValueError(f"Expected [B,H,W], got {tuple(p.shape)}")
    return p.reshape(p.shape[0], -1)


def _targets_from_coords(coords: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError(f"Expected coords [B,2], got {tuple(coords.shape)}")
    rr = coords[:, 0].long().clamp(0, h - 1)
    cc = coords[:, 1].long().clamp(0, w - 1)
    return rr * w + cc


def _pbr_loss(pred: torch.Tensor, coords: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    b, h, w = pred.shape
    rr = coords[:, 0].long().clamp(0, h - 1)
    cc = coords[:, 1].long().clamp(0, w - 1)
    batch_idx = torch.arange(b, device=pred.device)

    peaks = pred[batch_idx, rr, cc]
    total_sum = pred.reshape(b, -1).sum(dim=1)
    background_sum = total_sum - peaks
    background_count = max(h * w - 1, 1)
    background_mean = background_sum / background_count

    pbr = peaks.clamp_min(eps) / background_mean.clamp_min(eps)
    return -torch.log(pbr).mean()


def _gaussian_targets(coords: torch.Tensor, h: int, w: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    b = int(coords.shape[0])
    y = torch.arange(h, device=device, dtype=torch.float32).view(1, h, 1)
    x = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, w)
    cy = coords[:, 0].float().view(b, 1, 1)
    cx = coords[:, 1].float().view(b, 1, 1)
    g = torch.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2.0 * max(float(sigma), 1e-6) ** 2))
    g = g / (g.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return g.to(dtype=dtype)


def _soft_argmax_xy(pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p = _to_intensity(pred)
    b, h, w = p.shape
    probs = F.softmax(p.reshape(b, -1), dim=1).reshape(b, h, w)
    ys = torch.arange(h, device=p.device, dtype=probs.dtype).view(1, h, 1)
    xs = torch.arange(w, device=p.device, dtype=probs.dtype).view(1, 1, w)
    y_hat = (probs * ys).sum(dim=(1, 2))
    x_hat = (probs * xs).sum(dim=(1, 2))
    return y_hat, x_hat


def _match_target_map(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError(f"Expected target map [B,H,W], got {tuple(target.shape)}")
    target = target.to(device=pred.device, dtype=pred.dtype)
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(target.unsqueeze(1), size=pred.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return target.clamp(0.0, 1.0)


def _compute_map_loss(name: str, pred: torch.Tensor, target: torch.Tensor, cfg: Any) -> torch.Tensor:
    _ = cfg
    target_map = _match_target_map(target, pred)
    if name in ("mse", "map_mse", "mask_mse", "box_mse"):
        return F.mse_loss(pred, target_map)
    if name in ("nrmse", "map_nrmse", "mask_nrmse"):
        mse = F.mse_loss(pred, target_map)
        denom = torch.sqrt(target_map.square().mean() + 1e-8)
        return torch.sqrt(mse + 1e-8) / denom
    if name in ("bce", "mask_bce"):
        pred_prob = pred / (pred.amax(dim=(1, 2), keepdim=True) + 1e-8)
        pred_prob = pred_prob.clamp(1e-6, 1.0 - 1e-6)
        return F.binary_cross_entropy(pred_prob, target_map)
    if name in ("soft_iou", "iou"):
        pred_prob = pred / (pred.amax(dim=(1, 2), keepdim=True) + 1e-8)
        inter = (pred_prob * target_map).sum(dim=(1, 2))
        union = (pred_prob + target_map - pred_prob * target_map).sum(dim=(1, 2))
        return (1.0 - (inter + 1e-6) / (union + 1e-6)).mean()
    raise ValueError(f"Unsupported map loss: {name}")


def _compute_single_loss(name: str, pred: torch.Tensor, coords: torch.Tensor, cfg: Any) -> torch.Tensor:
    p = _to_intensity(pred)
    b, h, w = p.shape
    if coords.ndim in (3, 4):
        return _compute_map_loss(name, p, coords, cfg)
    logits = p.reshape(b, -1)
    targets = _targets_from_coords(coords, h, w)

    if name == "pbr":
        return _pbr_loss(p, coords)
    if name == "xent":
        return F.cross_entropy(logits, targets)
    if name == "xent_smooth":
        smoothing = float(_cfg_get(cfg, "loss.label_smoothing", _cfg_get(cfg, "label_smoothing", 0.1)))
        return F.cross_entropy(logits, targets, label_smoothing=smoothing)
    if name == "focal":
        alpha = float(_cfg_get(cfg, "loss.focal_alpha", _cfg_get(cfg, "focal_alpha", 0.25)))
        gamma = float(_cfg_get(cfg, "loss.focal_gamma", _cfg_get(cfg, "focal_gamma", 2.0)))
        logp = F.log_softmax(logits, dim=1)
        p_t = logp.gather(1, targets.view(-1, 1)).exp().squeeze(1)
        ce = F.nll_loss(logp, targets, reduction="none")
        focal = (alpha * (1.0 - p_t).pow(gamma) * ce).mean()
        return focal
    if name in ("mse", "nrmse", "kl"):
        sigma = float(_cfg_get(cfg, "loss.gauss_sigma", _cfg_get(cfg, "gauss_sigma", 1.5)))
        target_map = _gaussian_targets(coords, h, w, sigma, p.device, p.dtype)
        if name == "mse":
            return F.mse_loss(p, target_map)
        if name == "nrmse":
            mse = F.mse_loss(p, target_map)
            denom = torch.sqrt((target_map.square().mean()) + 1e-8)
            return torch.sqrt(mse + 1e-8) / denom
        return kl_divergence_loss(logits, target_map.reshape(b, -1))
    if name in ("coord_mse", "coord_mse_norm"):
        y_hat, x_hat = _soft_argmax_xy(p)
        yy = coords[:, 0].float()
        xx = coords[:, 1].float()
        if name == "coord_mse_norm":
            y_hat = y_hat / max(h - 1, 1)
            x_hat = x_hat / max(w - 1, 1)
            yy = yy / max(h - 1, 1)
            xx = xx / max(w - 1, 1)
        return 0.5 * (F.mse_loss(y_hat, yy) + F.mse_loss(x_hat, xx))
    raise ValueError(f"Unsupported loss: {name}")


def compute_loss_by_name(pred: Any, coords: torch.Tensor, cfg: Any, test: bool = False) -> torch.Tensor:
    _ = test
    pred_t = _unwrap_prediction(pred)
    if pred_t.ndim == 4 and pred_t.shape[1] == 1:
        pred_t = pred_t[:, 0]
    if pred_t.ndim != 3:
        raise ValueError(f"Loss expects prediction [B,H,W], got {tuple(pred_t.shape)}")

    loss_name = str(_cfg_get(cfg, "loss.name", _cfg_get(cfg, "loss", "pbr"))).lower()
    if loss_name == "mix":
        a = str(_cfg_get(cfg, "loss.mix_loss_a", _cfg_get(cfg, "mix_loss_a", "pbr"))).lower()
        b = str(_cfg_get(cfg, "loss.mix_loss_b", _cfg_get(cfg, "mix_loss_b", "mse"))).lower()
        alpha = float(_cfg_get(cfg, "loss.mix_alpha", _cfg_get(cfg, "mix_alpha", 0.5)))
        la = _compute_single_loss(a, pred_t, coords, cfg)
        lb = _compute_single_loss(b, pred_t, coords, cfg)
        return alpha * la + (1.0 - alpha) * lb
    return _compute_single_loss(loss_name, pred_t, coords, cfg)


def compute_argmax_coord_metrics(pred: Any, coords: torch.Tensor) -> dict[str, float]:
    pred_t = _unwrap_prediction(pred)
    pred_i = _to_intensity(pred_t)
    if pred_i.ndim == 4 and pred_i.shape[1] == 1:
        pred_i = pred_i[:, 0]
    if pred_i.ndim != 3 or coords.ndim != 2 or coords.shape[-1] != 2:
        return {}

    b, _h, w = pred_i.shape
    flat_idx = pred_i.reshape(b, -1).argmax(dim=1)
    pred_y = (flat_idx // w).float()
    pred_x = (flat_idx % w).float()
    target_y = coords[:, 0].float()
    target_x = coords[:, 1].float()

    dy = pred_y - target_y
    dx = pred_x - target_x
    coord_mse = 0.5 * (dy.square() + dx.square())
    distance = torch.sqrt(dy.square() + dx.square())
    return {
        "coord_mse_argmax": float(coord_mse.mean().detach().cpu()),
        "mean_pixel_distance": float(distance.mean().detach().cpu()),
    }


__all__ = ["compute_loss_by_name", "compute_argmax_coord_metrics"]

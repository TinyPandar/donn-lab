from __future__ import annotations

from typing import Any

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore[assignment]

from config.schema import ExperimentConfig
from core.classification import DetectorRegionReadout, to_intensity
from data.datamodule import DataModule


def _to_intensity(pred: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(pred):
        return pred.real.pow(2) + pred.imag.pow(2)
    return pred


def _unwrap_pred(pred):
    if isinstance(pred, (tuple, list)):
        if len(pred) == 0:
            raise ValueError("Empty output tuple/list")
        return pred[0]
    return pred


def _tensor_to_display_rgb(x: torch.Tensor) -> np.ndarray:
    x0 = x[0].detach().cpu().float().clamp(0.0, 1.0)
    if x0.ndim != 3:
        raise ValueError(f"Expected visual input [C,H,W], got {tuple(x0.shape)}")
    if x0.shape[0] == 1:
        img = x0.repeat(3, 1, 1)
    else:
        img = x0[:3]
    return (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _coord_to_display(row: float, col: float, out_h: int, out_w: int, display_h: int, display_w: int) -> tuple[float, float]:
    y = float(row) / max(out_h - 1, 1) * max(display_h - 1, 1)
    x = float(col) / max(out_w - 1, 1) * max(display_w - 1, 1)
    return y, x


def _target_center(target: torch.Tensor, default_h: int, default_w: int) -> tuple[float, float, int, int] | None:
    t = target.detach().cpu()
    if t.ndim == 2 and t.shape[-1] == 2:
        return float(t[0, 0]), float(t[0, 1]), int(default_h), int(default_w)
    if t.ndim == 3 and t.shape[0] == 1 and t.shape[-1] == 2:
        return float(t[0, 0, 0]), float(t[0, 0, 1]), int(default_h), int(default_w)
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    if t.ndim == 3 and t.shape[0] == 1:
        t = t[0]
    if t.ndim != 2:
        return None
    total = float(t.sum().item())
    h, w = int(t.shape[0]), int(t.shape[1])
    if total <= 0:
        flat_idx = int(t.reshape(-1).argmax().item())
        return float(flat_idx // w), float(flat_idx % w), h, w
    ys = torch.arange(h, dtype=torch.float32).view(h, 1)
    xs = torch.arange(w, dtype=torch.float32).view(1, w)
    row = float((t.float() * ys).sum().item() / total)
    col = float((t.float() * xs).sum().item() / total)
    return row, col, h, w


@torch.no_grad()
def visualize_epoch_samples(
    *,
    model: torch.nn.Module,
    datamodule: DataModule,
    cfg: ExperimentConfig,
    epoch: int,
    writer,
    clearml_logger: Any | None = None,
    tag_prefix: str = "",
) -> None:
    if writer is None:
        return
    if str(cfg.pipeline).lower() == "classification":
        _visualize_classification_samples(
            model=model,
            datamodule=datamodule,
            cfg=cfg,
            epoch=epoch,
            writer=writer,
            clearml_logger=clearml_logger,
            tag_prefix=tag_prefix,
        )
        return
    model.eval()
    for split in ("Train", "Test"):
        samples = datamodule.sample_for_vis(split, int(cfg.data.vis_samples))
        if not samples:
            continue
        rows = int(np.ceil(len(samples) / 3))
        cols = 3
        fig = plt.figure(figsize=(cols * 4, rows * 3.5))
        for i, sample_data in enumerate(samples):
            if isinstance(sample_data, dict):
                x_in = sample_data["x"]
                target = sample_data.get("target")
            else:
                raise TypeError("Visualization samples must be preprocessed dicts from DataModule.sample_for_vis")
            img_rgb = _tensor_to_display_rgb(x_in)
            orig_h, orig_w = img_rgb.shape[:2]
            x_in = x_in.to(next(model.parameters()).device)
            pred = _unwrap_pred(model(x_in))
            pred = _to_intensity(pred)
            if pred.ndim == 4 and pred.shape[1] == 1:
                pred = pred[:, 0]
            intensity = pred[0].detach().cpu().numpy()

            flat_idx = int(intensity.reshape(-1).argmax())
            peak_r = flat_idx // intensity.shape[1]
            peak_c = flat_idx % intensity.shape[1]
            pred_cy, pred_cx = _coord_to_display(
                peak_r,
                peak_c,
                int(intensity.shape[0]),
                int(intensity.shape[1]),
                orig_h,
                orig_w,
            )

            heat = (intensity - intensity.min()) / (np.ptp(intensity) + 1e-12)
            heat_u8 = (heat * 255.0).astype(np.uint8)
            heat_u8 = cv2.resize(heat_u8, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            heat_color_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
            vis = cv2.addWeighted(heat_color_rgb, 0.45, img_rgb, 0.55, 0)

            if isinstance(target, torch.Tensor):
                center = _target_center(target, int(cfg.data.h_out), int(cfg.data.w_out))
                if center is not None:
                    gt_r, gt_c, target_h, target_w = center
                    gt_cy, gt_cx = _coord_to_display(gt_r, gt_c, target_h, target_w, orig_h, orig_w)
                    cv2.circle(vis, (int(gt_cx), int(gt_cy)), 6, (0, 255, 0), 2)

            cv2.drawMarker(
                vis,
                (int(pred_cx), int(pred_cy)),
                (255, 255, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=16,
                thickness=2,
            )
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(vis)
            ax.set_title(f"Sample {i + 1}")
            ax.axis("off")

        fig.suptitle(f"Epoch {epoch} [{split}] GT (green) vs Pred (white)")
        fig.tight_layout()
        writer.add_figure(f"{tag_prefix}{split.lower()}/epoch_visualization", fig, global_step=epoch)
        if clearml_logger is not None:
            title = f"{tag_prefix}{split.lower()}/epoch_visualization"
            try:
                clearml_logger.report_matplotlib_figure(
                    title=title,
                    series="final",
                    figure=fig,
                    iteration=epoch,
                    report_image=True,
                )
            except TypeError:
                try:
                    clearml_logger.report_matplotlib_figure(
                        title=title,
                        series="final",
                        figure=fig,
                        iteration=epoch,
                    )
                except Exception as exc:
                    print(f"Warning: failed to upload {title} to ClearML: {exc}")
            except Exception as exc:
                print(f"Warning: failed to upload {title} to ClearML: {exc}")
        plt.close(fig)


@torch.no_grad()
def _visualize_classification_samples(
    *,
    model: torch.nn.Module,
    datamodule: DataModule,
    cfg: ExperimentConfig,
    epoch: int,
    writer,
    clearml_logger: Any | None,
    tag_prefix: str,
) -> None:
    model.eval()
    device = next(model.parameters()).device
    c = cfg.classification
    readout = DetectorRegionReadout(
        output_hw=(int(cfg.data.h_out), int(cfg.data.w_out)),
        num_classes=int(c.num_classes),
        grid_rows=int(c.grid_rows),
        grid_cols=int(c.grid_cols),
        roi_hw=(int(c.roi_h), int(c.roi_w)),
        margin=int(c.detector_margin),
        log_energy=bool(c.log_energy),
    ).to(device)

    for split in ("Train", "Test"):
        samples = datamodule.sample_for_vis(split, int(cfg.data.vis_samples))
        if not samples:
            continue
        fig, axes = plt.subplots(len(samples), 2, figsize=(7, 3 * len(samples)), squeeze=False)
        for row, sample_data in enumerate(samples):
            x_in = sample_data["x"]
            target = sample_data["target"]
            label = int(target.reshape(-1)[0].item())
            pred_map = _unwrap_pred(model(x_in.to(device)))
            logits, _energies, _efficiency = readout(pred_map)
            pred_class = int(logits.argmax(dim=1)[0].item())
            intensity = to_intensity(pred_map)[0].detach().cpu().numpy()

            axes[row, 0].imshow(_tensor_to_display_rgb(x_in))
            axes[row, 0].set_title(f"Input · label {label}")
            axes[row, 0].axis("off")

            ax = axes[row, 1]
            ax.imshow(intensity, cmap="inferno")
            for region in readout.regions:
                is_gt = region.class_index == label
                is_pred = region.class_index == pred_class
                color = "lime" if is_gt else ("cyan" if is_pred else "white")
                width = 2.5 if (is_gt or is_pred) else 0.7
                ax.add_patch(
                    Rectangle(
                        (region.left, region.top),
                        region.width,
                        region.height,
                        fill=False,
                        edgecolor=color,
                        linewidth=width,
                    )
                )
                ax.text(region.left + 1, region.top + 10, str(region.class_index), color=color, fontsize=8)
            ax.set_title(f"Detector · pred {pred_class}")
            ax.axis("off")

        fig.suptitle(f"Epoch {epoch} [{split}] GT (green) / prediction (cyan)")
        fig.tight_layout()
        title = f"{tag_prefix}{split.lower()}/epoch_classification"
        writer.add_figure(title, fig, global_step=epoch)
        if clearml_logger is not None:
            try:
                clearml_logger.report_matplotlib_figure(
                    title=title,
                    series="final",
                    figure=fig,
                    iteration=epoch,
                    report_image=True,
                )
            except TypeError:
                clearml_logger.report_matplotlib_figure(
                    title=title,
                    series="final",
                    figure=fig,
                    iteration=epoch,
                )
            except Exception as exc:
                print(f"Warning: failed to upload {title} to ClearML: {exc}")
        plt.close(fig)

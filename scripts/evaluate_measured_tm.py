from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp import autocast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.loader import load_experiment_config
from config.schema import ExperimentConfig, update_config_from_dict
from core.losses import compute_loss_by_name
from data import register_builtin_datasets
from engine.checkpoint import CheckpointIO
from models import register_builtin_models
from models.factory import DistContext
from pipelines import create_registered_pipeline, register_builtin_pipelines
from registry import create_dataset


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(
        run_dir.glob("epoch_*.pth"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No epoch_*.pth checkpoints found in {run_dir}")
    return checkpoints[-1]


def load_run_config(run_dir: Path) -> ExperimentConfig:
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = ExperimentConfig()
    update_config_from_dict(cfg, payload)
    return cfg


def coords_from_target(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 2 and target.shape[-1] == 2:
        return target.long()
    target_map = target[:, 0] if target.ndim == 4 and target.shape[1] == 1 else target
    if target_map.ndim != 3:
        raise ValueError(f"Cannot derive coords from target shape {tuple(target.shape)}")
    batch, _height, width = target_map.shape
    flat_idx = target_map.reshape(batch, -1).argmax(dim=1)
    return torch.stack((flat_idx // width, flat_idx % width), dim=1).long()


def intensity_from_pred(pred: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(pred):
        pred = pred.real.square() + pred.imag.square()
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if pred.ndim != 3:
        raise ValueError(f"Expected prediction [B,H,W], got {tuple(pred.shape)}")
    return pred


def evaluate(cfg: ExperimentConfig, checkpoint: Path | None, device: torch.device) -> dict[str, Any]:
    register_builtin_models()
    register_builtin_datasets()
    register_builtin_pipelines()

    pipeline = create_registered_pipeline(cfg.pipeline)
    pipeline.validate_config(cfg)
    dist_ctx = DistContext(False, 0, 1)
    model = pipeline.build_model(cfg, device, dist_ctx)
    if checkpoint is not None:
        CheckpointIO.load(path=str(checkpoint), model=model, map_location=device, strict=False)
    pipeline.setup(cfg, model, device, dist_ctx)
    model.eval()

    datamodule = create_dataset(cfg.dataset, cfg)
    autocast_enabled = device.type == "cuda" and str(cfg.model_cfg.tmatrix_compute_dtype) in ("bf16", "fp16")
    autocast_dtype = torch.bfloat16 if str(cfg.model_cfg.tmatrix_compute_dtype) == "bf16" else torch.float16

    if str(cfg.pipeline).lower() == "classification":
        metric_sums: dict[str, float] = {}
        sample_count = 0
        with torch.no_grad():
            for batch_idx, (x_batch, target_batch) in enumerate(datamodule.val_iter(), start=1):
                x_batch = x_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)
                batch_size = int(x_batch.shape[0])
                with autocast("cuda", enabled=autocast_enabled, dtype=autocast_dtype if autocast_enabled else None):
                    out = pipeline.validation_step((x_batch, target_batch), model, cfg)
                values = {"configured_loss": float(out.loss.detach().cpu())}
                values.update({k: float(v) for k, v in out.metrics.items() if k not in {"val_loss", "task_loss"}})
                for key, value in values.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size
                sample_count += batch_size
                if cfg.data.max_test_batches > 0 and batch_idx >= int(cfg.data.max_test_batches):
                    break
        n = max(sample_count, 1)
        return {"n": sample_count, **{key: value / n for key, value in metric_sums.items()}}

    sums = {
        "n": 0.0,
        "configured_loss": 0.0,
        "xent_loss": 0.0,
        "coord_mse_argmax": 0.0,
        "mean_pixel_distance": 0.0,
        "within_5px": 0.0,
        "within_10px": 0.0,
        "target_pbr": 0.0,
    }

    with torch.no_grad():
        for batch_idx, (x_batch, target_batch) in enumerate(datamodule.val_iter(), start=1):
            x_batch = x_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            with autocast("cuda", enabled=autocast_enabled, dtype=autocast_dtype if autocast_enabled else None):
                pred = model(x_batch)

            intensity = intensity_from_pred(pred)
            coords = coords_from_target(target_batch).to(device)
            batch, height, width = intensity.shape
            flat_idx = intensity.reshape(batch, -1).argmax(dim=1)
            pred_y = (flat_idx // width).float()
            pred_x = (flat_idx % width).float()
            target_y = coords[:, 0].float()
            target_x = coords[:, 1].float()
            dy = pred_y - target_y
            dx = pred_x - target_x
            distance = torch.sqrt(dy.square() + dx.square())
            coord_mse = 0.5 * (dy.square() + dx.square())

            batch_indices = torch.arange(batch, device=device)
            target_intensity = intensity[
                batch_indices,
                coords[:, 0].clamp(0, height - 1),
                coords[:, 1].clamp(0, width - 1),
            ]
            total_sum = intensity.reshape(batch, -1).sum(dim=1)
            background_mean = (total_sum - target_intensity) / max(height * width - 1, 1)
            pbr = target_intensity.clamp_min(1e-12) / background_mean.clamp_min(1e-12)

            class_targets = (coords[:, 0].clamp(0, height - 1) * width + coords[:, 1].clamp(0, width - 1)).long()
            xent = F.cross_entropy(intensity.reshape(batch, -1), class_targets)
            configured_loss = compute_loss_by_name(intensity, coords, cfg, test=True)

            sums["n"] += float(batch)
            sums["configured_loss"] += float(configured_loss.detach().cpu()) * batch
            sums["xent_loss"] += float(xent.detach().cpu()) * batch
            sums["coord_mse_argmax"] += float(coord_mse.sum().detach().cpu())
            sums["mean_pixel_distance"] += float(distance.sum().detach().cpu())
            sums["within_5px"] += float((distance <= 5.0).sum().detach().cpu())
            sums["within_10px"] += float((distance <= 10.0).sum().detach().cpu())
            sums["target_pbr"] += float(pbr.sum().detach().cpu())

            if cfg.data.max_test_batches > 0 and batch_idx >= int(cfg.data.max_test_batches):
                break

    n = max(sums.pop("n"), 1.0)
    return {"n": int(n), **{key: value / n for key, value in sums.items()}}


def write_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a measured-TM DONN checkpoint on a chosen TM file.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path for measured_tm_scatter.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Existing run dir with config.json/checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to load. Defaults to latest in --run-dir.")
    parser.add_argument("--tmatrix-path", type=str, default=None, help="Override cfg.model_cfg.tmatrix_path, e.g. H0.npy or H1.npy.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=Path("reports/measured_tm_eval/metrics.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir is None and args.config is None:
        raise SystemExit("Provide --config or --run-dir")

    if args.run_dir is not None:
        cfg = load_run_config(args.run_dir)
        checkpoint = args.checkpoint or latest_checkpoint(args.run_dir)
    else:
        cfg, _ = load_experiment_config(["--config", str(args.config), "--model", "measured_tm_scatter"])
        checkpoint = args.checkpoint

    if args.tmatrix_path is not None:
        cfg.model_cfg.tmatrix_path = args.tmatrix_path
    if args.batch_size is not None:
        cfg.data.batch_size = int(args.batch_size)
    if args.max_test_batches is not None:
        cfg.data.max_test_batches = int(args.max_test_batches)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics = evaluate(cfg, checkpoint, device)
    metrics["checkpoint"] = str(checkpoint) if checkpoint is not None else ""
    metrics["tmatrix_path"] = str(cfg.model_cfg.tmatrix_path)
    write_metrics(metrics, args.out)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

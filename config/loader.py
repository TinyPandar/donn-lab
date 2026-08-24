from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from .schema import ExperimentConfig, dataclass_to_dict, update_config_from_dict

DATASET_ROOTS = {
    "inria": "/home/limingfei/speckle/donn/datasets/INRIAPerson",
    "pennfudan": "/home/limingfei/speckle/donn/datasets/PennFudanPed",
    "mnist": "/home/limingfei/speckle/donn/datasets/mnist",
    "syn": "/home/limingfei/speckle/donn/datasets/synthetic",
    "synped": "/home/limingfei/speckle/donn/datasets/synped",
    "vehicle": "/home/limingfei/speckle/donn/datasets/dataset_simple",
    "tracking_vehicle": "/home/limingfei/speckle/donn/datasets/tracking_vehicle",
    "syntrack": "/home/limingfei/speckle/donn/datasets/tracking_vehicle",
    "tracking": "/home/limingfei/speckle/donn/datasets/tracking_vehicle",
}

RUN_DIR_RE = re.compile(r"^20\d{6}-\d{6}(?:_.+)?$")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DONN unified training entrypoint")
    parser.add_argument("--config", "--configs", dest="config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--pipeline", type=str, default=None, choices=["base", "classification", "distill", "stn"])

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--vis_samples", type=int, default=None)
    parser.add_argument("--label_filter", type=str, default=None)
    parser.add_argument("--mnist_target_mode", type=str, default=None, choices=["coord", "class"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--clearml", dest="clearml_enabled", action="store_true", default=None)
    parser.add_argument("--clearml_project_name", type=str, default=None)
    parser.add_argument("--clearml_task_name", type=str, default=None)
    parser.add_argument("--clearml_tags", type=str, default=None)
    parser.add_argument("--clearml_output_uri", type=str, default=None)
    parser.add_argument("--clearml_offline", action="store_true", default=None)
    parser.add_argument("--clearml_auto_connect_tensorboard", action="store_true", default=None)
    parser.add_argument("--clearml_auto_connect_pytorch", action="store_true", default=None)
    parser.add_argument("--clearml_report_scalars", action="store_true", default=None)
    parser.add_argument("--clearml_report_batch_scalars", action="store_true", default=None)
    parser.add_argument("--h_in", type=int, default=None)
    parser.add_argument("--w_in", type=int, default=None)
    parser.add_argument("--h_out", type=int, default=None)
    parser.add_argument("--w_out", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--tmatrix_compute_dtype", type=str, default=None)
    parser.add_argument("--tmatrix_sparsity", type=float, default=None)
    parser.add_argument("--tp", action="store_true", default=None)
    parser.add_argument("--save_every", type=int, default=None)

    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--focal_alpha", type=float, default=None)
    parser.add_argument("--focal_gamma", type=float, default=None)
    parser.add_argument("--gauss_sigma", type=float, default=None)
    parser.add_argument("--mix_loss_a", type=str, default=None)
    parser.add_argument("--mix_loss_b", type=str, default=None)
    parser.add_argument("--mix_alpha", type=float, default=None)

    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--class_grid_rows", type=int, default=None)
    parser.add_argument("--class_grid_cols", type=int, default=None)
    parser.add_argument("--class_roi_h", type=int, default=None)
    parser.add_argument("--class_roi_w", type=int, default=None)
    parser.add_argument("--detector_margin", type=int, default=None)
    parser.add_argument("--classification_log_energy", action="store_true", default=None)
    parser.add_argument("--efficiency_weight", type=float, default=None)

    parser.add_argument("--enable_aug", action="store_true", default=None)
    parser.add_argument("--disable_aug", action="store_true", default=None)
    parser.add_argument("--aug_hflip_p", type=float, default=None)
    parser.add_argument("--aug_color_p", type=float, default=None)
    parser.add_argument("--aug_blur_p", type=float, default=None)
    parser.add_argument("--aug_noise_p", type=float, default=None)
    parser.add_argument("--zoom_crop_mode", type=str, default=None)
    parser.add_argument("--zoom_crop_factor", type=float, default=None)
    parser.add_argument("--multiple_objects", action="store_true", default=None)
    parser.add_argument("--only_single", action="store_true", default=None)
    parser.add_argument("--vehicle_channel_mode", type=str, default=None)
    parser.add_argument("--vehicle_channel", type=str, default=None)
    parser.add_argument("--vehicle_channel_invert", action="store_true", default=None)
    parser.add_argument("--vehicle_channel_p_low", type=float, default=None)
    parser.add_argument("--vehicle_channel_p_high", type=float, default=None)

    parser.add_argument("--activation", type=str, default=None)
    parser.add_argument("--activation_params", type=str, default=None)
    parser.add_argument("--phase_dropout", type=float, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--resume_from_last", action="store_true", default=None)
    parser.add_argument("--memory_snapshot", action="store_true", default=None)
    parser.add_argument("--memory_snapshot_batch", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--amp_dtype", type=str, default=None)

    parser.add_argument("--teacher_model", type=str, default=None)
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--task_w", type=float, default=None)
    parser.add_argument("--kd_pred_w", type=float, default=None)
    parser.add_argument("--kd_feat_w", type=float, default=None)
    parser.add_argument("--kd_mode", type=str, default=None)
    parser.add_argument("--kd_temperature", type=float, default=None)

    parser.add_argument("--resnet_variant", type=str, default=None)
    parser.add_argument("--resnet_pretrained", action="store_true", default=None)
    parser.add_argument("--normalize_negative", action="store_true", default=None)
    parser.add_argument("--return_intensity", action="store_true", default=None)
    parser.add_argument("--normalize_input", action="store_true", default=None)
    parser.add_argument("--input_amplitude_normalization", type=str, default=None)
    parser.add_argument("--sqrt_amplitude", action="store_true", default=None)
    parser.add_argument("--detector_psf_sigma", type=float, default=None)
    parser.add_argument("--input_mode", type=str, default=None)
    parser.add_argument("--phase_init", type=str, default=None)
    parser.add_argument("--tmatrix_scale", type=float, default=None)
    parser.add_argument("--tmatrix_path", type=str, default=None)
    parser.add_argument("--tmatrix_shape", type=str, default=None)
    parser.add_argument("--tmatrix_dtype", type=str, default=None)
    parser.add_argument("--tmatrix_layout", type=str, default=None)
    parser.add_argument("--tmatrix_normalization", type=str, default=None)
    parser.add_argument("--tmatrix_input_h", type=int, default=None)
    parser.add_argument("--tmatrix_input_w", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--tile_h", type=int, default=None)
    parser.add_argument("--tile_w", type=int, default=None)
    parser.add_argument("--tile_layout_mode", type=str, default=None)
    parser.add_argument("--learnable_tile_gains", action="store_true", default=None)
    parser.add_argument("--no_tile_energy_normalize", dest="tile_energy_normalize", action="store_false", default=None)
    parser.add_argument("--learnable_amplitude_bias", action="store_true", default=None)
    parser.add_argument("--cnn_encoder_depth", type=int, default=None)
    parser.add_argument("--cnn_encoder_width", type=int, default=None)
    parser.add_argument("--cnn_encoder_norm", type=str, default=None)
    parser.add_argument("--cnn_encoder_activation", type=str, default=None)
    parser.add_argument("--cnn_encoder_out_activation", type=str, default=None)
    parser.add_argument("--mixer_type", type=str, default=None)
    parser.add_argument("--mixer_out_activation", type=str, default=None)
    parser.add_argument("--mixer_freq_range", type=float, default=None)
    parser.add_argument("--mixer_mode", type=str, default=None)
    parser.add_argument("--mixer_init_scale", type=float, default=None)
    parser.add_argument("--mixer_kernel_size", type=int, default=None)
    parser.add_argument("--mixer_residual", action="store_true", default=None)
    parser.add_argument("--mixer_gate_init", type=float, default=None)
    parser.add_argument("--mixer_norm", type=str, default=None)
    parser.add_argument("--mixer_norm_eps", type=float, default=None)
    parser.add_argument("--mixer_domain", type=str, default=None)
    return parser


def _set_if_not_none(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def _load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must parse to dict, got {type(payload).__name__}")
    return payload


def _payload_has_path(payload: dict[str, Any], *parts: str) -> bool:
    node: Any = payload
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _payload_get_path(payload: dict[str, Any], *parts: str) -> Any:
    node: Any = payload
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _apply_yaml_payload(cfg: ExperimentConfig, payload: dict[str, Any]) -> None:
    if not payload:
        return
    # Support both nested schema and old flat schema.
    if any(k in payload for k in ("data", "model_cfg", "optim", "runtime", "logging", "loss", "classification", "distill", "output", "clearml")):
        update_config_from_dict(cfg, payload)
    else:
        _apply_flat_overrides(cfg, payload)


def _log_root_from_maybe_run_dir(log_dir: str, pipeline: str, model_group: str, dataset: str) -> str:
    path = os.path.abspath(log_dir)
    parts = path.split(os.sep)
    if len(parts) >= 5 and RUN_DIR_RE.match(parts[-1]):
        if parts[-4:-1] == [pipeline, model_group, dataset]:
            return os.sep.join(parts[:-4]) or os.sep
    return path


def _flatten_map_from_cli(args: argparse.Namespace) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    args_dict = vars(args)
    for k, v in args_dict.items():
        _set_if_not_none(flat, k, v)
    return flat


def _apply_flat_overrides(cfg: ExperimentConfig, flat: dict[str, Any]) -> list[str]:
    deprecated_hits: list[str] = []
    if "disable_aug" in flat:
        flat["enable_aug"] = not bool(flat["disable_aug"])
        deprecated_hits.append("--disable_aug")
    if "only_single" in flat:
        flat["multiple_objects"] = False
        deprecated_hits.append("--only_single")
    if isinstance(flat.get("clearml_tags"), str):
        flat["clearml_tags"] = [tag.strip() for tag in flat["clearml_tags"].split(",") if tag.strip()]

    if "pipeline" in flat:
        cfg.pipeline = str(flat["pipeline"])
    if "dataset" in flat:
        cfg.dataset = str(flat["dataset"])
        cfg.data.dataset = str(flat["dataset"])
    if "model" in flat:
        cfg.model = str(flat["model"])
        cfg.model_cfg.name = str(flat["model"])
    if "data_root" in flat:
        cfg.data_root = str(flat["data_root"])
        cfg.data.data_root = str(flat["data_root"])

    direct_map = {
        "label_filter": ("data", "label_filter"),
        "mnist_target_mode": ("data", "mnist_target_mode"),
        "multiple_objects": ("data", "multiple_objects"),
        "batch_size": ("data", "batch_size"),
        "max_train_batches": ("data", "max_train_batches"),
        "max_test_batches": ("data", "max_test_batches"),
        "h_in": ("data", "h_in"),
        "w_in": ("data", "w_in"),
        "h_out": ("data", "h_out"),
        "w_out": ("data", "w_out"),
        "vis_samples": ("data", "vis_samples"),
        "input_mode": ("data", "input_mode"),
        "enable_aug": ("data", "enable_aug"),
        "aug_hflip_p": ("data", "aug_hflip_p"),
        "aug_color_p": ("data", "aug_color_p"),
        "aug_blur_p": ("data", "aug_blur_p"),
        "aug_noise_p": ("data", "aug_noise_p"),
        "zoom_crop_mode": ("data", "zoom_crop_mode"),
        "zoom_crop_factor": ("data", "zoom_crop_factor"),
        "vehicle_channel_mode": ("data", "vehicle_channel_mode"),
        "vehicle_channel": ("data", "vehicle_channel"),
        "vehicle_channel_invert": ("data", "vehicle_channel_invert"),
        "vehicle_channel_p_low": ("data", "vehicle_channel_p_low"),
        "vehicle_channel_p_high": ("data", "vehicle_channel_p_high"),
        "vehicle_target_mode": ("data", "vehicle_target_mode"),
        "vehicle_target_column": ("data", "vehicle_target_column"),
        "num_layers": ("model_cfg", "num_layers"),
        "seed": ("model_cfg", "seed"),
        "activation": ("model_cfg", "activation"),
        "activation_params": ("model_cfg", "activation_params"),
        "phase_dropout": ("model_cfg", "phase_dropout"),
        "normalize_input": ("model_cfg", "normalize_input"),
        "input_amplitude_normalization": ("model_cfg", "input_amplitude_normalization"),
        "sqrt_amplitude": ("model_cfg", "sqrt_amplitude"),
        "detector_psf_sigma": ("model_cfg", "detector_psf_sigma"),
        "phase_init": ("model_cfg", "phase_init"),
        "tmatrix_scale": ("model_cfg", "tmatrix_scale"),
        "tmatrix_path": ("model_cfg", "tmatrix_path"),
        "tmatrix_shape": ("model_cfg", "tmatrix_shape"),
        "tmatrix_dtype": ("model_cfg", "tmatrix_dtype"),
        "tmatrix_layout": ("model_cfg", "tmatrix_layout"),
        "tmatrix_normalization": ("model_cfg", "tmatrix_normalization"),
        "tmatrix_input_h": ("model_cfg", "tmatrix_input_h"),
        "tmatrix_input_w": ("model_cfg", "tmatrix_input_w"),
        "tmatrix_compute_dtype": ("model_cfg", "tmatrix_compute_dtype"),
        "tmatrix_sparsity": ("model_cfg", "tmatrix_sparsity"),
        "tp": ("model_cfg", "tp"),
        "return_intensity": ("model_cfg", "return_intensity"),
        "normalize_negative": ("model_cfg", "normalize_negative"),
        "resnet_variant": ("model_cfg", "resnet_variant"),
        "resnet_pretrained": ("model_cfg", "resnet_pretrained"),
        "tile_h": ("model_cfg", "tile_h"),
        "tile_w": ("model_cfg", "tile_w"),
        "tile_layout_mode": ("model_cfg", "tile_layout_mode"),
        "learnable_tile_gains": ("model_cfg", "learnable_tile_gains"),
        "tile_energy_normalize": ("model_cfg", "tile_energy_normalize"),
        "learnable_amplitude_bias": ("model_cfg", "learnable_amplitude_bias"),
        "cnn_encoder_depth": ("model_cfg", "cnn_encoder_depth"),
        "cnn_encoder_width": ("model_cfg", "cnn_encoder_width"),
        "cnn_encoder_norm": ("model_cfg", "cnn_encoder_norm"),
        "cnn_encoder_activation": ("model_cfg", "cnn_encoder_activation"),
        "cnn_encoder_out_activation": ("model_cfg", "cnn_encoder_out_activation"),
        "mixer_type": ("model_cfg", "mixer_type"),
        "mixer_out_activation": ("model_cfg", "mixer_out_activation"),
        "mixer_freq_range": ("model_cfg", "mixer_freq_range"),
        "mixer_mode": ("model_cfg", "mixer_mode"),
        "mixer_init_scale": ("model_cfg", "mixer_init_scale"),
        "mixer_kernel_size": ("model_cfg", "mixer_kernel_size"),
        "mixer_residual": ("model_cfg", "mixer_residual"),
        "mixer_gate_init": ("model_cfg", "mixer_gate_init"),
        "mixer_norm": ("model_cfg", "mixer_norm"),
        "mixer_norm_eps": ("model_cfg", "mixer_norm_eps"),
        "mixer_domain": ("model_cfg", "mixer_domain"),
        "lr": ("optim", "lr"),
        "epochs": ("optim", "epochs"),
        "save_every": ("optim", "save_every"),
        "device": ("runtime", "device"),
        "amp": ("runtime", "amp"),
        "amp_dtype": ("runtime", "amp_dtype"),
        "resume_ckpt": ("runtime", "resume_ckpt"),
        "resume_from_last": ("runtime", "resume_from_last"),
        "memory_snapshot": ("runtime", "memory_snapshot"),
        "memory_snapshot_batch": ("runtime", "memory_snapshot_batch"),
        "log_dir": ("logging", "log_dir"),
        "ckpt_dir": ("logging", "ckpt_dir"),
        "comment": ("logging", "comment"),
        "clearml_enabled": ("clearml", "enabled"),
        "clearml_project_name": ("clearml", "project_name"),
        "clearml_task_name": ("clearml", "task_name"),
        "clearml_tags": ("clearml", "tags"),
        "clearml_output_uri": ("clearml", "output_uri"),
        "clearml_offline": ("clearml", "offline"),
        "clearml_auto_connect_tensorboard": ("clearml", "auto_connect_tensorboard"),
        "clearml_auto_connect_pytorch": ("clearml", "auto_connect_pytorch"),
        "clearml_report_scalars": ("clearml", "report_scalars"),
        "clearml_report_batch_scalars": ("clearml", "report_batch_scalars"),
        "loss": ("loss", "name"),
        "label_smoothing": ("loss", "label_smoothing"),
        "focal_alpha": ("loss", "focal_alpha"),
        "focal_gamma": ("loss", "focal_gamma"),
        "gauss_sigma": ("loss", "gauss_sigma"),
        "mix_loss_a": ("loss", "mix_loss_a"),
        "mix_loss_b": ("loss", "mix_loss_b"),
        "mix_alpha": ("loss", "mix_alpha"),
        "num_classes": ("classification", "num_classes"),
        "class_grid_rows": ("classification", "grid_rows"),
        "class_grid_cols": ("classification", "grid_cols"),
        "class_roi_h": ("classification", "roi_h"),
        "class_roi_w": ("classification", "roi_w"),
        "detector_margin": ("classification", "detector_margin"),
        "classification_log_energy": ("classification", "log_energy"),
        "efficiency_weight": ("classification", "efficiency_weight"),
        "teacher_model": ("distill", "teacher_model"),
        "teacher_ckpt": ("distill", "teacher_ckpt"),
        "task_w": ("distill", "task_w"),
        "kd_pred_w": ("distill", "kd_pred_w"),
        "kd_feat_w": ("distill", "kd_feat_w"),
        "kd_mode": ("distill", "kd_mode"),
        "kd_temperature": ("distill", "kd_temperature"),
    }
    for k, v in flat.items():
        if k not in direct_map:
            continue
        group, name = direct_map[k]
        obj = getattr(cfg, group)
        setattr(obj, name, v)

    cfg.dataset = cfg.data.dataset
    cfg.model = cfg.model_cfg.name
    cfg.data_root = cfg.data.data_root

    if cfg.data.max_train_batches < 0:
        cfg.data.max_train_batches = 0
    if cfg.data.max_test_batches < 0:
        cfg.data.max_test_batches = 0
    return deprecated_hits


def _is_optical_input_model(model: str) -> bool:
    return str(model).lower() in {"scatter", "scatter_tile", "stn", "cnn_scatter", "scatter_mixer", "measured_tm_scatter"}


def _is_auto_input_mode(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "auto", "default"}


def _finalize_derived_fields(cfg: ExperimentConfig) -> None:
    # Keep top-level aliases and nested config in sync.
    if cfg.model and (not cfg.model_cfg.name or cfg.model_cfg.name == "scatter"):
        cfg.model_cfg.name = cfg.model
    elif cfg.model_cfg.name:
        cfg.model = cfg.model_cfg.name
    if cfg.dataset and (not cfg.data.dataset or cfg.data.dataset == "synped"):
        cfg.data.dataset = cfg.dataset
    elif cfg.data.dataset:
        cfg.dataset = cfg.data.dataset

    if cfg.runtime.device is None:
        cfg.runtime.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(cfg.runtime.device, str) and cfg.runtime.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.runtime.device = "cpu"

    if (not cfg.data.data_root) and cfg.dataset in DATASET_ROOTS:
        cfg.data.data_root = DATASET_ROOTS[cfg.dataset]
    if cfg.data.data_root in DATASET_ROOTS.values() and cfg.dataset in DATASET_ROOTS:
        # Keep old behavior: follow dataset switch when still using default preset paths.
        cfg.data.data_root = DATASET_ROOTS[cfg.dataset]
    cfg.data_root = cfg.data.data_root

    if cfg.pipeline == "stn" and cfg.model not in ("scatter_tile", "scattertile", "stn"):
        cfg.model = "scatter_tile"
        cfg.model_cfg.name = "scatter_tile"

    data_input_explicit = bool(cfg.extras.get("data_input_mode_explicit", False))
    legacy_model_input_mode = cfg.extras.get("legacy_model_input_mode")
    if _is_auto_input_mode(cfg.data.input_mode):
        if legacy_model_input_mode and _is_optical_input_model(cfg.model):
            cfg.data.input_mode = str(legacy_model_input_mode)
            cfg.extras.setdefault("deprecated_flags", []).append(
                "model_cfg.input_mode is legacy; mapped it to data.input_mode for this optical model."
            )
        elif _is_optical_input_model(cfg.model):
            cfg.data.input_mode = "gray"
        else:
            cfg.data.input_mode = "rgb"
    elif not data_input_explicit and legacy_model_input_mode and str(cfg.data.input_mode) != str(legacy_model_input_mode):
        cfg.extras.setdefault("deprecated_flags", []).append(
            "Both data.input_mode and legacy model_cfg.input_mode are present; data.input_mode takes precedence."
        )

    ts = time.strftime("%Y%m%d-%H%M%S")
    if cfg.logging.comment:
        ts = f"{ts}_{cfg.logging.comment}"

    model_group = "cnn" if cfg.model in ("simple_cnn", "resnet", "matmul_cnn") else cfg.model
    if cfg.runtime.resume_ckpt or cfg.runtime.resume_from_last:
        cfg.logging.ckpt_dir = os.path.abspath(cfg.logging.ckpt_dir)
        log_root = _log_root_from_maybe_run_dir(cfg.logging.log_dir, cfg.pipeline, model_group, cfg.dataset)
        cfg.logging.log_dir = os.path.join(log_root, cfg.pipeline, model_group, cfg.dataset, ts)
    else:
        cfg.logging.log_dir = os.path.join(cfg.logging.log_dir, cfg.pipeline, model_group, cfg.dataset, ts)
        cfg.logging.ckpt_dir = os.path.join(cfg.logging.ckpt_dir, cfg.pipeline, model_group, cfg.dataset, ts)

    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    os.makedirs(cfg.logging.ckpt_dir, exist_ok=True)


def load_experiment_config(argv: list[str] | None = None) -> tuple[ExperimentConfig, argparse.Namespace]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = ExperimentConfig()
    default_payload = _load_yaml(str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None)
    _apply_yaml_payload(cfg, default_payload)

    yaml_payload = _load_yaml(args.config)
    data_input_mode_explicit = _payload_has_path(yaml_payload, "data", "input_mode") or ("input_mode" in yaml_payload)
    legacy_model_input_mode = _payload_get_path(yaml_payload, "model_cfg", "input_mode")
    _apply_yaml_payload(cfg, yaml_payload)

    cli_flat = _flatten_map_from_cli(args)
    if cli_flat.get("input_mode") is not None:
        data_input_mode_explicit = True
    deprecated = _apply_flat_overrides(cfg, cli_flat)
    cfg.extras["data_input_mode_explicit"] = data_input_mode_explicit
    cfg.extras["legacy_model_input_mode"] = legacy_model_input_mode
    cfg.extras["deprecated_flags"] = deprecated
    _finalize_derived_fields(cfg)

    cfg.extras["loaded_from_config"] = args.config
    cfg.extras["deprecated_flags"] = list(cfg.extras.get("deprecated_flags", []))
    cfg.extras["raw_cli"] = vars(args)
    cfg.extras["resolved_config_json"] = json.dumps(dataclass_to_dict(cfg), indent=2, ensure_ascii=False)
    return cfg, args

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import cv2

from vehicle_center_loader import VehicleCenterDataset


@dataclass(frozen=True)
class VehicleExperimentSample:
    """One stable, preprocessed sample used by the optical experiment."""

    index: int
    sample_id: str
    image_path: Path
    input_chw: np.ndarray
    target_rc: Tuple[int, int]


def _nested(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key, {})
    return value if isinstance(value, dict) else {}


def configure_vehicle_preprocessing(config_dump: Dict[str, Any]) -> None:
    """Validate the loader settings serialized in a training checkpoint."""

    data_cfg = _nested(config_dump, "data")
    if str(data_cfg.get("vehicle_target_mode", "center")).lower() not in {"center", "coord", "coordinate"}:
        raise ValueError("The optical localization experiment requires vehicle_target_mode=center")


def _prepare_image_bgr(config_dump: Dict[str, Any], image_bgr: np.ndarray) -> np.ndarray:
    """Bit-for-bit equivalent of data_loader.prepare_image_bgr_to_tensor."""

    data_cfg = _nested(config_dump, "data")
    height = int(data_cfg.get("h_in", 128))
    width = int(data_cfg.get("w_in", 128))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
    values = image_rgb.astype(np.float32) / np.float32(255.0)
    mode = str(data_cfg.get("input_mode", "rgb")).lower()
    if mode in {"auto", "default", "rgb", "bgr", "none", "off"}:
        model_array = values
    elif mode in {"gray", "grey", "grayscale", "luminance", "lum"}:
        weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float32).reshape(1, 1, 3)
        model_array = (values * weights).sum(axis=2, keepdims=True)
    elif mode == "mean":
        model_array = values.mean(axis=2, keepdims=True)
    else:
        channels = {
            "r": 0,
            "red": 0,
            "0": 0,
            "g": 1,
            "green": 1,
            "1": 1,
            "b": 2,
            "blue": 2,
            "2": 2,
        }
        if mode not in channels:
            raise ValueError("Unsupported data.input_mode: {0!r}".format(mode))
        channel = channels[mode]
        model_array = values[:, :, channel : channel + 1]
    return np.ascontiguousarray(model_array.transpose(2, 0, 1), dtype=np.float32)


def resolve_data_root(config_dump: Dict[str, Any], override: Optional[Path] = None) -> Path:
    if override is not None:
        resolved = override.expanduser().resolve()
    else:
        data_cfg = _nested(config_dump, "data")
        raw = data_cfg.get("data_root", config_dump.get("data_root"))
        if not raw:
            raise ValueError("Training configuration does not contain data_root")
        path = Path(str(raw)).expanduser()
        if path.is_absolute():
            resolved = path.resolve()
        else:
            cwd_candidate = path.resolve()
            if cwd_candidate.exists():
                resolved = cwd_candidate
            else:
                repository_root = Path(__file__).resolve().parents[2]
                resolved = (repository_root / path).resolve()
    if not (resolved / "annotations.csv").is_file():
        variant = resolved / "random"
        if (variant / "annotations.csv").is_file():
            resolved = variant
    return resolved


def _stable_sample_id(index: int, image_path: Path, dataset_root: Path) -> str:
    try:
        rel = image_path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        rel = image_path.resolve().as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in image_path.stem)
    return "{0:06d}_{1}_{2}".format(index, safe_stem[:48], digest)


def iter_vehicle_samples(
    config_dump: Dict[str, Any],
    *,
    data_root: Optional[Path] = None,
    split: str = "val",
    indices: Optional[Sequence[int]] = None,
    start_index: int = 0,
    max_samples: int = 0,
) -> Iterator[VehicleExperimentSample]:
    """Yield samples without losing the filename/index identity like batch loaders do."""

    if str(config_dump.get("dataset", "vehicle")).lower() != "vehicle":
        raise ValueError("This experiment runner currently requires dataset='vehicle'")
    configure_vehicle_preprocessing(config_dump)
    root = resolve_data_root(config_dump, data_root)
    dataset = VehicleCenterDataset(str(root), split=split)
    data_cfg = _nested(config_dump, "data")
    h_out = int(data_cfg.get("h_out", 128))
    w_out = int(data_cfg.get("w_out", 128))
    require_single = not bool(data_cfg.get("multiple_objects", False))

    if indices is None:
        wanted = range(max(int(start_index), 0), len(dataset.samples))
    else:
        wanted = [int(index) for index in indices]

    emitted = 0
    for index in wanted:
        if index < 0 or index >= len(dataset.samples):
            raise IndexError("Sample index {0} is outside [0, {1})".format(index, len(dataset.samples)))
        image_path_raw, centers = dataset.samples[index]
        if require_single and (centers.ndim != 2 or centers.shape[0] != 1):
            continue
        image_bgr = VehicleCenterDataset._read_image_bgr(image_path_raw)
        if image_bgr is None:
            raise RuntimeError("Failed to read image: {0}".format(image_path_raw))
        if centers.size == 0:
            continue

        original_h, original_w = image_bgr.shape[:2]
        center_x = float(centers[0, 0])
        center_y = float(centers[0, 1])
        row = int(np.clip(np.round(center_y / original_h * h_out), 0, h_out - 1))
        col = int(np.clip(np.round(center_x / original_w * w_out), 0, w_out - 1))
        input_chw = _prepare_image_bgr(config_dump, image_bgr)
        image_path = Path(os.path.abspath(image_path_raw))
        yield VehicleExperimentSample(
            index=index,
            sample_id=_stable_sample_id(index, image_path, root),
            image_path=image_path,
            input_chw=input_chw,
            target_rc=(row, col),
        )
        emitted += 1
        if max_samples > 0 and emitted >= int(max_samples):
            return


__all__ = [
    "VehicleExperimentSample",
    "configure_vehicle_preprocessing",
    "iter_vehicle_samples",
    "resolve_data_root",
]

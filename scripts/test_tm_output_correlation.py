#!/usr/bin/env python
"""Compare measured camera outputs against ``abs(TM @ field) ** 2``.

The experiment deliberately includes several input-field families so that a
low score can be localized to the measured TM/coordinate path, the DMD complex
field encoder, or the PNG-amplitude network input.  Hardware access is guarded
by ``--arm-hardware``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TMATRIX = REPO_ROOT / "tm.npy"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "runs/linked_optical_vehicle/20260817-155503-png-amplitude-10ep/"
    "training_checkpoints/base/measured_tm_scatter/vehicle/"
    "20260817-155506_png-amplitude-10ep/epoch_010.pth"
)
DEFAULT_V4_ROOT = Path(r"C:\Users\smart\Documents\TMCalib")
DEFAULT_PROBE_FILE = (
    DEFAULT_V4_ROOT
    / "pregenerated_patterns_128_px4_active512_8N_full/probe.npy"
)
DEFAULT_V4_MODULE = REPO_ROOT / "combined_app_v4_128.py"
DEFAULT_DLL_PARENT = Path(r"C:\Users\smart\Documents")
GROUPS = ("calibration_probe", "random_phase", "phase_conjugate", "png_layer0")


@dataclass(frozen=True)
class FieldRecord:
    group: str
    label: str
    field: np.ndarray
    metadata: Mapping[str, Any]


def _resolved(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0 or not math.isfinite(denominator):
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _max_normalized(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.maximum(np.asarray(array, dtype=np.float64), 0.0)
    maximum = float(values.max(initial=0.0))
    if maximum <= eps:
        return np.zeros_like(values)
    return values / maximum


def _best_dihedral(
    measured: np.ndarray,
    predicted: np.ndarray,
) -> Tuple[str, float]:
    candidates = []  # type: List[Tuple[str, np.ndarray]]
    for turns in range(4):
        rotated = np.rot90(predicted, turns)
        candidates.append(("rot{0}".format(turns * 90), rotated))
        candidates.append(("rot{0}_flip_lr".format(turns * 90), np.fliplr(rotated)))
    scores = [(name, _pearson(measured, transformed)) for name, transformed in candidates]
    finite = [item for item in scores if math.isfinite(item[1])]
    return max(finite, key=lambda item: item[1]) if finite else ("undefined", float("nan"))


def _metrics(
    predicted: np.ndarray,
    measured_raw: np.ndarray,
    measured_corrected: np.ndarray,
    measured_raw_repeats: np.ndarray,
    measured_corrected_repeats: np.ndarray,
) -> Dict[str, Any]:
    predicted_norm = _max_normalized(predicted)
    measured_norm = _max_normalized(measured_corrected)
    difference = predicted_norm - measured_norm
    predicted_peak = np.unravel_index(int(np.argmax(predicted)), predicted.shape)
    measured_peak = np.unravel_index(int(np.argmax(measured_corrected)), measured_corrected.shape)
    peak_distance = math.hypot(
        float(predicted_peak[0] - measured_peak[0]),
        float(predicted_peak[1] - measured_peak[1]),
    )
    transform_name, transform_correlation = _best_dihedral(
        measured_corrected, predicted
    )
    repeat_correlations = []  # type: List[float]
    for first in range(measured_corrected_repeats.shape[0]):
        for second in range(first + 1, measured_corrected_repeats.shape[0]):
            repeat_correlations.append(
                _pearson(
                    measured_corrected_repeats[first],
                    measured_corrected_repeats[second],
                )
            )
    repeat_pearson = (
        float(np.nanmean(np.asarray(repeat_correlations, dtype=np.float64)))
        if repeat_correlations
        else float("nan")
    )
    saturated_fraction = float(np.mean(measured_raw_repeats >= 255.0))
    correlation_valid = bool(
        saturated_fraction <= 0.001
        and float(np.max(measured_raw) - np.min(measured_raw)) >= 5.0
    )
    return {
        "pearson": _pearson(measured_corrected, predicted),
        "correlation_valid": correlation_valid,
        "repeat_pearson": repeat_pearson,
        "normalized_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "predicted_peak_row": int(predicted_peak[0]),
        "predicted_peak_col": int(predicted_peak[1]),
        "measured_peak_row": int(measured_peak[0]),
        "measured_peak_col": int(measured_peak[1]),
        "peak_distance_px": float(peak_distance),
        "best_dihedral": transform_name,
        "best_dihedral_pearson": float(transform_correlation),
        "measured_min": float(np.min(measured_raw)),
        "measured_max": float(np.max(measured_raw)),
        "measured_dynamic_range": float(np.max(measured_raw) - np.min(measured_raw)),
        "measured_saturated_fraction": saturated_fraction,
        "predicted_mean": float(np.mean(predicted)),
        "measured_corrected_mean": float(np.mean(measured_corrected)),
    }


def _load_tm(path: Path) -> np.ndarray:
    matrix = np.load(str(path), mmap_mode="r", allow_pickle=False)
    if matrix.shape != (128 * 128, 128 * 128):
        raise ValueError("TM must have shape (16384,16384), got {0}".format(matrix.shape))
    if matrix.dtype != np.complex64:
        raise ValueError("TM must be complex64, got {0}".format(matrix.dtype))
    return matrix


def _calibration_probe_records(
    path: Path,
    count: int,
    rng: np.random.Generator,
) -> List[FieldRecord]:
    probes = np.load(str(path), mmap_mode="r", allow_pickle=False)
    if probes.ndim != 3 or tuple(probes.shape[1:]) != (128, 128):
        raise ValueError("Calibration probes must have shape [N,128,128]")
    if probes.dtype != np.complex64:
        raise ValueError("Calibration probes must be complex64")
    indices = np.sort(rng.choice(int(probes.shape[0]), size=count, replace=False))
    return [
        FieldRecord(
            group="calibration_probe",
            label="calibration_{0:06d}".format(int(index)),
            field=np.ascontiguousarray(probes[int(index)], dtype=np.complex64),
            metadata={"probe_index": int(index), "probe_file": str(path)},
        )
        for index in indices
    ]


def _random_phase_records(
    count: int,
    rng: np.random.Generator,
) -> List[FieldRecord]:
    records = []  # type: List[FieldRecord]
    for index in range(count):
        phase_level = rng.integers(0, 16, size=(128, 128), dtype=np.int16)
        phase = phase_level.astype(np.float32) * np.float32(2.0 * np.pi / 16.0)
        field = np.exp(1j * phase).astype(np.complex64)
        records.append(
            FieldRecord(
                group="random_phase",
                label="random_phase_{0:03d}".format(index),
                field=field,
                metadata={"phase_levels": 16},
            )
        )
    return records


def _focus_targets(count: int) -> List[Tuple[int, int]]:
    candidates = [
        (64, 64),
        (32, 32),
        (32, 96),
        (96, 32),
        (96, 96),
        (64, 24),
        (24, 64),
        (104, 64),
        (64, 104),
    ]
    if count <= len(candidates):
        return candidates[:count]
    extra = []
    for index in range(count - len(candidates)):
        extra.append(((17 * index + 11) % 128, (43 * index + 29) % 128))
    return candidates + extra


def _phase_conjugate_records(
    matrix: np.ndarray,
    count: int,
) -> List[FieldRecord]:
    records = []  # type: List[FieldRecord]
    for row, col in _focus_targets(count):
        tm_row = np.asarray(matrix[row * 128 + col], dtype=np.complex64)
        field = np.exp(-1j * np.angle(tm_row)).reshape(128, 128).astype(np.complex64)
        records.append(
            FieldRecord(
                group="phase_conjugate",
                label="focus_r{0:03d}_c{1:03d}".format(row, col),
                field=np.ascontiguousarray(field),
                metadata={"target_row": row, "target_col": col},
            )
        )
    return records


def _png_layer0_records(
    checkpoint_path: Path,
    data_root: Optional[Path],
    split: str,
    count: int,
) -> List[FieldRecord]:
    from donn_lab.experiment.optical_inference import (
        initial_amplitude,
        load_checkpoint_phases,
        normalize_spatial_per_sample,
    )
    from donn_lab.experiment.vehicle_data import iter_vehicle_samples

    checkpoint = load_checkpoint_phases(checkpoint_path, strict=True)
    samples = list(
        iter_vehicle_samples(
            dict(checkpoint.config_dump),
            data_root=data_root,
            split=split,
            max_samples=count,
        )
    )
    if len(samples) != count:
        raise RuntimeError("Requested {0} PNG samples, found {1}".format(count, len(samples)))
    images = np.stack([sample.input_chw for sample in samples], axis=0).astype(np.float32)
    amplitude = initial_amplitude(
        images,
        input_hw=checkpoint.input_hw,
        mode_hw=checkpoint.mode_hw,
        normalize_input=checkpoint.normalize_input,
        input_amplitude_normalization=checkpoint.input_amplitude_normalization,
        sqrt_amplitude=checkpoint.sqrt_amplitude,
    )
    normalization_mode = (
        checkpoint.input_amplitude_normalization
        if checkpoint.normalize_input
        else "minmax"
    )
    amplitude = normalize_spatial_per_sample(amplitude, mode=normalization_mode)
    phase = torch.from_numpy(np.array(checkpoint.phases[0], copy=True)).view(1, 1, 128, 128)
    fields = torch.polar(amplitude.float(), phase)[:, 0].numpy().astype(np.complex64)
    records = []  # type: List[FieldRecord]
    for index, sample in enumerate(samples):
        records.append(
            FieldRecord(
                group="png_layer0",
                label="png_{0}".format(sample.sample_id),
                field=np.ascontiguousarray(fields[index]),
                metadata={
                    "dataset_index": int(sample.index),
                    "image_path": str(sample.image_path),
                    "target_row": int(sample.target_rc[0]),
                    "target_col": int(sample.target_rc[1]),
                    "checkpoint": str(checkpoint_path),
                    "input_amplitude_normalization": checkpoint.input_amplitude_normalization,
                    "sqrt_amplitude": bool(checkpoint.sqrt_amplitude),
                },
            )
        )
    return records


def _build_records(
    args: argparse.Namespace,
    matrix: np.ndarray,
) -> List[FieldRecord]:
    rng = np.random.default_rng(int(args.seed))
    records = []  # type: List[FieldRecord]
    selected = set(args.groups)
    if "calibration_probe" in selected:
        records.extend(_calibration_probe_records(args.probe_file, args.samples_per_group, rng))
    if "random_phase" in selected:
        records.extend(_random_phase_records(args.samples_per_group, rng))
    if "phase_conjugate" in selected:
        records.extend(_phase_conjugate_records(matrix, args.samples_per_group))
    if "png_layer0" in selected:
        records.extend(
            _png_layer0_records(
                args.checkpoint,
                args.data_root,
                args.split,
                args.samples_per_group,
            )
        )
    if not records:
        raise ValueError("No input fields were generated")
    return records


def _predict(
    tmatrix: Path,
    fields: np.ndarray,
    device: str,
    stream_tm: bool,
    chunk_rows: Optional[int],
) -> np.ndarray:
    from donn_lab.hardware.torch_tm_backend import TorchTMBackend

    backend = TorchTMBackend(
        tmatrix_path=tmatrix,
        input_hw=(128, 128),
        output_hw=(128, 128),
        device=device,
        cache_on_device=not stream_tm,
        chunk_rows=chunk_rows,
    )
    try:
        backend.open()
        return np.ascontiguousarray(
            backend.project_and_capture_fields(fields), dtype=np.float32
        )
    finally:
        backend.close()


def _capture(
    args: argparse.Namespace,
    fields: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    from donn_lab.hardware.v4_128_backend import V4128Backend

    backend = V4128Backend(
        v4_root=args.v4_root,
        save_path=args.output_dir / "camera",
        device_name=args.dmd_device,
        camera_index=args.camera_index,
        module_path=args.v4_module,
        dll_parent=args.dll_parent,
        max_retries=args.max_retries,
    )
    try:
        backend.open(arm=True)
        dark = backend.capture_dark(
            batch_size=1,
            capture_repeats=args.dark_frames,
        )[0]
        field_indices = np.repeat(np.arange(fields.shape[0]), args.capture_repeats)
        repeat_indices = np.tile(np.arange(args.capture_repeats), fields.shape[0])
        capture_order = np.random.default_rng(args.seed + 991).permutation(
            field_indices.shape[0]
        )
        ordered = backend.project_and_capture_fields(
            np.ascontiguousarray(fields[field_indices[capture_order]], dtype=np.complex64),
            capture_repeats=1,
        )
        repeats = np.empty(
            (fields.shape[0], args.capture_repeats, 128, 128),
            dtype=np.float32,
        )
        repeats[
            field_indices[capture_order], repeat_indices[capture_order]
        ] = ordered
        measured = repeats.mean(axis=1, dtype=np.float32)
        metadata_before_close = dict(backend.metadata)
        return (
            np.ascontiguousarray(measured, dtype=np.float32),
            np.ascontiguousarray(repeats, dtype=np.float32),
            np.ascontiguousarray(dark, dtype=np.float32),
            metadata_before_close,
        )
    finally:
        backend.close()


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary = {}  # type: Dict[str, Any]
    for group in sorted({str(row["group"]) for row in rows}):
        selected = [row for row in rows if row["group"] == group]
        pearson = np.asarray([float(row["pearson"]) for row in selected], dtype=np.float64)
        valid_pearson = np.asarray(
            [float(row["pearson"]) for row in selected if bool(row["correlation_valid"])],
            dtype=np.float64,
        )
        repeat_pearson = np.asarray(
            [float(row["repeat_pearson"]) for row in selected], dtype=np.float64
        )
        rmse = np.asarray([float(row["normalized_rmse"]) for row in selected], dtype=np.float64)
        distance = np.asarray([float(row["peak_distance_px"]) for row in selected], dtype=np.float64)
        summary[group] = {
            "count": len(selected),
            "pearson_mean": float(np.nanmean(pearson)),
            "pearson_median": float(np.nanmedian(pearson)),
            "pearson_min": float(np.nanmin(pearson)),
            "pearson_max": float(np.nanmax(pearson)),
            "valid_count": int(valid_pearson.size),
            "valid_pearson_mean": (
                None if valid_pearson.size == 0 else float(np.nanmean(valid_pearson))
            ),
            "repeat_pearson_mean": float(np.nanmean(repeat_pearson)),
            "normalized_rmse_mean": float(np.mean(rmse)),
            "peak_distance_mean_px": float(np.mean(distance)),
            "identity_was_best_dihedral_fraction": float(
                np.mean([row["best_dihedral"] == "rot0" for row in selected])
            ),
            "max_saturated_fraction": float(
                max(float(row["measured_saturated_fraction"]) for row in selected)
            ),
        }
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "index",
        "group",
        "label",
        "pearson",
        "correlation_valid",
        "repeat_pearson",
        "normalized_rmse",
        "predicted_peak_row",
        "predicted_peak_col",
        "measured_peak_row",
        "measured_peak_col",
        "peak_distance_px",
        "best_dihedral",
        "best_dihedral_pearson",
        "measured_min",
        "measured_max",
        "measured_dynamic_range",
        "measured_saturated_fraction",
        "predicted_mean",
        "measured_corrected_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(
    output_dir: Path,
    records: Sequence[FieldRecord],
    predicted: np.ndarray,
    measured: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []  # type: List[str]
    for group in sorted({record.group for record in records}):
        indices = [index for index, record in enumerate(records) if record.group == group]
        figure, axes = plt.subplots(
            len(indices),
            3,
            figsize=(10.5, max(3.0, 3.0 * len(indices))),
            squeeze=False,
        )
        for row_index, item_index in enumerate(indices):
            pred = _max_normalized(predicted[item_index])
            meas = _max_normalized(measured[item_index])
            diff = meas - pred
            axes[row_index, 0].imshow(pred, cmap="inferno", vmin=0.0, vmax=1.0)
            axes[row_index, 1].imshow(meas, cmap="inferno", vmin=0.0, vmax=1.0)
            axes[row_index, 2].imshow(diff, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            axes[row_index, 0].set_title("TM prediction")
            axes[row_index, 1].set_title(
                "camera | r={0:.3f}".format(float(rows[item_index]["pearson"]))
            )
            axes[row_index, 2].set_title(
                "camera - prediction | peak d={0:.1f}".format(
                    float(rows[item_index]["peak_distance_px"])
                )
            )
            axes[row_index, 0].set_ylabel(records[item_index].label, fontsize=8)
            for axis in axes[row_index]:
                axis.set_xticks([])
                axis.set_yticks([])
        figure.tight_layout()
        path = output_dir / "comparison_{0}.png".format(group)
        figure.savefig(str(path), dpi=150)
        plt.close(figure)
        files.append(str(path))
    return files


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmatrix", type=Path, default=DEFAULT_TMATRIX)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--probe-file", type=Path, default=DEFAULT_PROBE_FILE)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--samples-per-group", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stream-tm", action="store_true")
    parser.add_argument("--tm-chunk-rows", type=int, default=None)
    parser.add_argument("--capture-repeats", type=int, default=2)
    parser.add_argument("--dark-frames", type=int, default=16)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--dmd-device", default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--v4-root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument("--v4-module", type=Path, default=DEFAULT_V4_MODULE)
    parser.add_argument("--dll-parent", type=Path, default=DEFAULT_DLL_PARENT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--arm-hardware", action="store_true")
    args = parser.parse_args(argv)
    if args.samples_per_group <= 0:
        parser.error("--samples-per-group must be positive")
    if args.capture_repeats <= 0 or args.dark_frames <= 0:
        parser.error("capture repeats and dark frames must be positive")
    if args.tm_chunk_rows is not None and args.tm_chunk_rows <= 0:
        parser.error("--tm-chunk-rows must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.tmatrix = _resolved(args.tmatrix)
    args.checkpoint = _resolved(args.checkpoint)
    args.probe_file = _resolved(args.probe_file)
    args.v4_root = _resolved(args.v4_root)
    args.v4_module = _resolved(args.v4_module)
    args.dll_parent = _resolved(args.dll_parent)
    args.data_root = None if args.data_root is None else _resolved(args.data_root)
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = REPO_ROOT / "runs/tm_output_correlation" / stamp
    else:
        args.output_dir = _resolved(args.output_dir)

    required = {"tmatrix": args.tmatrix, "v4_root": args.v4_root, "v4_module": args.v4_module}
    if "calibration_probe" in args.groups:
        required["probe_file"] = args.probe_file
    if "png_layer0" in args.groups:
        required["checkpoint"] = args.checkpoint
    missing = ["{0}={1}".format(name, path) for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + "; ".join(missing))

    preflight = {
        "status": "ready" if args.arm_hardware else "preflight_only",
        "tmatrix": str(args.tmatrix),
        "checkpoint": str(args.checkpoint),
        "probe_file": str(args.probe_file),
        "groups": list(args.groups),
        "samples_per_group": int(args.samples_per_group),
        "capture_repeats": int(args.capture_repeats),
        "dark_frames": int(args.dark_frames),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(preflight, indent=2, ensure_ascii=False), flush=True)
    if not args.arm_hardware:
        print("No camera or DMD object was constructed; add --arm-hardware to run.")
        return 0
    if args.output_dir.exists():
        raise FileExistsError("Output directory already exists: {0}".format(args.output_dir))
    args.output_dir.mkdir(parents=True)

    matrix = _load_tm(args.tmatrix)
    records = _build_records(args, matrix)
    fields = np.stack([record.field for record in records], axis=0).astype(np.complex64)
    print("Predicting {0} fields with measured TM...".format(len(records)), flush=True)
    predicted = _predict(
        args.tmatrix,
        fields,
        args.device,
        args.stream_tm,
        args.tm_chunk_rows,
    )
    print("Capturing {0} fields on hardware...".format(len(records)), flush=True)
    measured_raw, measured_raw_repeats, dark, backend_metadata = _capture(args, fields)
    measured_corrected = np.maximum(measured_raw - dark[None, :, :], 0.0).astype(np.float32)
    measured_corrected_repeats = np.maximum(
        measured_raw_repeats - dark[None, None, :, :], 0.0
    ).astype(np.float32)

    rows = []  # type: List[Dict[str, Any]]
    for index, record in enumerate(records):
        row = {
            "index": index,
            "group": record.group,
            "label": record.label,
        }
        row.update(
            _metrics(
                predicted[index],
                measured_raw[index],
                measured_corrected[index],
                measured_raw_repeats[index],
                measured_corrected_repeats[index],
            )
        )
        rows.append(row)
        print(
            "[{0:02d}] {1:18s} r={2:.4f} repeat_r={3:.4f} "
            "nrmse={4:.4f} peak_d={5:.2f}px valid={6}".format(
                index,
                record.group,
                float(row["pearson"]),
                float(row["repeat_pearson"]),
                float(row["normalized_rmse"]),
                float(row["peak_distance_px"]),
                bool(row["correlation_valid"]),
            ),
            flush=True,
        )

    np.save(str(args.output_dir / "fields.npy"), fields)
    np.save(str(args.output_dir / "predicted_intensity.npy"), predicted)
    np.save(str(args.output_dir / "measured_raw.npy"), measured_raw)
    np.save(str(args.output_dir / "measured_raw_repeats.npy"), measured_raw_repeats)
    np.save(str(args.output_dir / "dark.npy"), dark)
    np.save(str(args.output_dir / "measured_corrected.npy"), measured_corrected)
    np.save(
        str(args.output_dir / "measured_corrected_repeats.npy"),
        measured_corrected_repeats,
    )
    _write_csv(args.output_dir / "records.csv", rows)
    plots = _write_plots(
        args.output_dir,
        records,
        predicted,
        measured_corrected,
        rows,
    )
    summary = {
        "status": "complete",
        "created_at_utc": _utc_now(),
        "tmatrix": str(args.tmatrix),
        "tmatrix_stat": {
            "size_bytes": int(args.tmatrix.stat().st_size),
            "mtime_ns": int(args.tmatrix.stat().st_mtime_ns),
        },
        "checkpoint": str(args.checkpoint),
        "probe_file": str(args.probe_file),
        "groups": _group_summary(rows),
        "num_fields": len(records),
        "capture_repeats": int(args.capture_repeats),
        "dark_frames": int(args.dark_frames),
        "dark": {
            "minimum": float(np.min(dark)),
            "maximum": float(np.max(dark)),
            "mean": float(np.mean(dark)),
        },
        "backend_metadata": dict(backend_metadata),
        "records": [
            {
                "index": index,
                "group": record.group,
                "label": record.label,
                "metadata": dict(record.metadata),
                "metrics": rows[index],
            }
            for index, record in enumerate(records)
        ],
        "plots": plots,
    }
    _json_dump(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["groups"], indent=2, ensure_ascii=False), flush=True)
    print("Artifacts: {0}".format(args.output_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from donn_lab.experiment.artifacts import RunArtifactWriter, quick_file_identity
from donn_lab.experiment.optical_inference import (
    CheckpointPhases,
    OpticalInferenceResult,
    OpticalInferenceRunner,
    OpticalInferenceSettings,
    load_checkpoint_phases,
)
from donn_lab.experiment.vehicle_data import (
    VehicleExperimentSample,
    iter_vehicle_samples,
    resolve_data_root,
)


DEFAULT_CHECKPOINT = REPO_ROOT / (
    "runs/checkpoints/base/measured_tm_scatter/vehicle/"
    "20260812-170342_vehicle-dark-intersection-measured-tm-scatter/epoch_300.pth"
)
DEFAULT_TMATRIX = REPO_ROOT / "tm.npy"
DEFAULT_V4_MODULE = REPO_ROOT / "combined_app_v4_128.py"
DEFAULT_V4_ROOT = Path(r"C:\Users\smart\Documents\TMCalib")
DEFAULT_DLL_PARENT = Path(r"C:\Users\smart\Documents")


def _path_arg(raw: str) -> Path:
    return Path(raw).expanduser()


def _parse_indices(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or not raw.strip():
        return None
    values = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise argparse.ArgumentTypeError("sample indices must be non-negative")
        values.append(value)
    if not values:
        return None
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("sample indices must not contain duplicates")
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a measured-TM phase checkpoint as a five-pass "
            "DMD/scattering-medium/camera optical network."
        )
    )
    parser.add_argument("--mode", choices=("simulate", "hardware"), default="simulate")
    parser.add_argument("--checkpoint", type=_path_arg, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config",
        type=_path_arg,
        default=None,
        help="Optional JSON/YAML config to cross-check against checkpoint config_dump.",
    )
    parser.add_argument("--tmatrix", type=_path_arg, default=DEFAULT_TMATRIX)
    parser.add_argument("--data-root", type=_path_arg, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sample-indices", default=None, help="Comma-separated indices within the selected split.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=1, help="0 means all selected samples.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--capture-repeats", type=int, default=1)
    parser.add_argument(
        "--dark-frames",
        type=int,
        default=None,
        help="Repeated black-pattern captures; defaults to 0 in simulation and 16 on hardware.",
    )
    parser.add_argument("--saturation-level", type=float, default=None)
    parser.add_argument("--max-saturated-fraction", type=float, default=0.001)
    parser.add_argument("--allow-saturation", action="store_true")
    parser.add_argument(
        "--min-dynamic-range",
        type=float,
        default=None,
        help="Minimum raw camera max-min; defaults to 5 native 8-bit codes in hardware mode.",
    )
    parser.add_argument("--allow-low-dynamic-range", action="store_true")
    parser.add_argument("--save-layers", choices=("none", "frames", "all"), default="frames")
    parser.add_argument("--output-dir", type=_path_arg, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hash-tmatrix", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")

    simulation = parser.add_argument_group("ideal measured-TM simulator")
    simulation.add_argument("--device", default="cuda:0")
    simulation.add_argument("--tmatrix-shape", default=None)
    simulation.add_argument("--tmatrix-dtype", default="complex64")
    simulation.add_argument("--tmatrix-layout", choices=("out_in", "in_out"), default=None)
    simulation.add_argument("--sim-noise-std", type=float, default=0.0)
    simulation.add_argument("--sim-dark-level", type=float, default=0.0)
    simulation.add_argument("--sim-quantize-8bit", action="store_true")
    simulation.add_argument("--sim-seed", type=int, default=42)
    simulation.add_argument(
        "--stream-tm",
        action="store_true",
        help="Keep the TM memory-mapped on host instead of caching it on the device.",
    )
    simulation.add_argument("--tm-chunk-rows", type=int, default=None)

    hardware = parser.add_argument_group("V4 128x128 DMD/camera hardware")
    hardware.add_argument("--arm-hardware", action="store_true")
    hardware.add_argument("--v4-module", type=_path_arg, default=DEFAULT_V4_MODULE)
    hardware.add_argument(
        "--v4-root",
        type=_path_arg,
        default=DEFAULT_V4_ROOT,
        help="Directory containing dmd_pattern_128.py and the other V4 helper modules.",
    )
    hardware.add_argument("--dll-parent", type=_path_arg, default=DEFAULT_DLL_PARENT)
    hardware.add_argument("--camera-index", type=int, default=0)
    hardware.add_argument("--camera-save-path", type=_path_arg, default=REPO_ROOT / "camera_1")
    hardware.add_argument("--dmd-device", default=None)
    hardware.add_argument("--max-retries", type=int, default=2)

    args = parser.parse_args(argv)
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    if args.batch_size <= 0 or args.batch_size > 1000:
        parser.error("--batch-size must be between 1 and 1000")
    if args.capture_repeats <= 0:
        parser.error("--capture-repeats must be positive")
    if args.batch_size * args.capture_repeats > 1000 and args.mode == "hardware":
        parser.error("hardware batch-size * capture-repeats must not exceed 1000")
    if args.dark_frames is not None and args.dark_frames < 0:
        parser.error("--dark-frames must be non-negative")
    if args.dark_frames is not None and args.dark_frames > 1000 and args.mode == "hardware":
        parser.error("--dark-frames must not exceed the V4 sequence limit of 1000")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    if args.tm_chunk_rows is not None and args.tm_chunk_rows <= 0:
        parser.error("--tm-chunk-rows must be positive")
    if args.min_dynamic_range is not None and args.min_dynamic_range < 0:
        parser.error("--min-dynamic-range must be non-negative")
    try:
        args.sample_indices = _parse_indices(args.sample_indices)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    if args.mode == "hardware" and not args.preflight_only and not args.arm_hardware:
        parser.error("hardware execution is inert unless --arm-hardware is supplied")
    if args.mode == "simulate" and args.resume and args.sim_noise_std > 0.0:
        parser.error(
            "--resume with --sim-noise-std is disabled because RNG state is not persisted"
        )
    return args


def _load_config_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Config does not exist: {0}".format(path))
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "Reading YAML requires PyYAML in this Python environment; "
                "the checkpoint already contains config_dump, so --config may also be omitted"
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping")
    return payload


def _get_nested(payload: Mapping[str, Any], path: str) -> Any:
    value = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _cross_check_config(checkpoint_config: Mapping[str, Any], supplied: Mapping[str, Any]) -> None:
    semantic_paths = (
        "dataset",
        "model",
        "data.h_in",
        "data.w_in",
        "data.h_out",
        "data.w_out",
        "data.input_mode",
        "model_cfg.name",
        "model_cfg.num_layers",
        "model_cfg.activation",
        "model_cfg.normalize_input",
        "model_cfg.input_amplitude_normalization",
        "model_cfg.sqrt_amplitude",
        "model_cfg.detector_psf_sigma",
        "model_cfg.tmatrix_layout",
        "model_cfg.tmatrix_normalization",
    )
    differences = []
    for path in semantic_paths:
        left = _get_nested(checkpoint_config, path)
        right = _get_nested(supplied, path)
        if left is not None and right is not None and left != right:
            differences.append("{0}: checkpoint={1!r}, config={2!r}".format(path, left, right))
    if differences:
        raise ValueError("Config is incompatible with the checkpoint:\n  " + "\n  ".join(differences))


def _resolved_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_path = candidate.resolve()
    if cwd_path.exists():
        return cwd_path
    return (REPO_ROOT / candidate).resolve()


def _build_backend(args: argparse.Namespace, checkpoint: CheckpointPhases) -> Any:
    if args.mode == "simulate":
        from donn_lab.hardware.torch_tm_backend import TorchTMBackend

        model_cfg = checkpoint.config_dump.get("model_cfg", {})
        layout = args.tmatrix_layout or str(model_cfg.get("tmatrix_layout", "out_in"))
        shape = args.tmatrix_shape or model_cfg.get("tmatrix_shape")
        return TorchTMBackend(
            tmatrix_path=_resolved_path(args.tmatrix),
            input_hw=checkpoint.mode_hw,
            output_hw=checkpoint.output_hw,
            tmatrix_shape=shape,
            tmatrix_dtype=args.tmatrix_dtype,
            layout=layout,
            device=args.device,
            noise_std=float(args.sim_noise_std),
            dark_level=float(args.sim_dark_level),
            quantize8=bool(args.sim_quantize_8bit),
            seed=int(args.sim_seed),
            detector_psf_sigma=float(checkpoint.detector_psf_sigma),
            cache_on_device=not bool(args.stream_tm),
            chunk_rows=args.tm_chunk_rows,
        )

    from donn_lab.hardware.v4_128_backend import V4128Backend

    return V4128Backend(
        v4_root=_resolved_path(args.v4_root),
        save_path=_resolved_path(args.camera_save_path),
        device_name=args.dmd_device,
        camera_index=int(args.camera_index),
        module_path=_resolved_path(args.v4_module),
        dll_parent=_resolved_path(args.dll_parent),
        max_retries=int(args.max_retries),
    )


def _hardware_dependency_report(args: argparse.Namespace) -> Dict[str, Any]:
    v4_module = _resolved_path(args.v4_module)
    v4_root = _resolved_path(args.v4_root)
    dll_parent = _resolved_path(args.dll_parent)
    required = {
        "v4_module": v4_module,
        "dmd_pattern_128": v4_root / "dmd_pattern_128.py",
        "tm_reconstruction_128": v4_root / "tm_reconstruction_128.py",
        "partial_tm_focus_128": v4_root / "partial_tm_focus_128.py",
        "pixelwise_focus_report_128": v4_root / "pixelwise_focus_report_128.py",
        "measurement_quality_report": v4_root / "measurement_quality_report.py",
        "juopt_dll": dll_parent / "JUOPT_DLP V4.0.002 20250522 release/4.DLL/DLL/JUOPT_DLL_V4.dll",
        "pthread_dll": dll_parent / "JUOPT_DLP V4.0.002 20250522 release/4.DLL/DLL/pthreadVC2.dll",
        "hologram_encoder": v4_root.parent / "holograms" / "dmd_holograms.py",
        "hologram_lut": v4_root.parent / "holograms" / "generate_LUT.py",
    }
    report = {name: {"path": str(path), "exists": path.is_file()} for name, path in required.items()}
    try:
        import importlib.util

        report["PySpin"] = {"available": importlib.util.find_spec("PySpin") is not None}
    except Exception as exc:
        report["PySpin"] = {"available": False, "error": str(exc)}
    return report


def _default_output_dir(mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "runs" / "optical_vehicle" / (stamp + "-" + mode)


def _runtime_versions() -> Dict[str, Any]:
    import cv2
    import torch

    return {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "opencv": str(cv2.__version__),
        "torch": str(torch.__version__),
        "torch_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
    }


def _selected_sample_digest(samples: Sequence[VehicleExperimentSample]) -> Dict[str, Any]:
    """Fingerprint the exact preprocessed tensors and targets used by this run."""

    digest = hashlib.sha256()
    for sample in samples:
        array = np.ascontiguousarray(sample.input_chw, dtype=np.float32)
        descriptor = {
            "index": int(sample.index),
            "sample_id": sample.sample_id,
            "shape": list(array.shape),
            "target_rc": [int(sample.target_rc[0]), int(sample.target_rc[1])],
        }
        digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return {"count": len(samples), "sha256": digest.hexdigest()}


def _fingerprint_payload(
    args: argparse.Namespace,
    checkpoint: CheckpointPhases,
    data_root: Path,
    selected_samples: Sequence[VehicleExperimentSample],
) -> Dict[str, Any]:
    checkpoint_path = _resolved_path(args.checkpoint)
    payload = {
        "schema": 4,
        "mode": args.mode,
        "checkpoint": quick_file_identity(checkpoint_path, include_hash=True),
        "data_root": str(data_root),
        "annotations": quick_file_identity(data_root / "annotations.csv", include_hash=True),
        "split": args.split,
        "sample_indices": args.sample_indices,
        "start_index": int(args.start_index),
        "max_samples": int(args.max_samples),
        "num_layers": int(checkpoint.num_layers),
        "input_hw": list(checkpoint.input_hw),
        "mode_hw": list(checkpoint.mode_hw),
        "output_hw": list(checkpoint.output_hw),
        "normalize_input": bool(checkpoint.normalize_input),
        "input_amplitude_normalization": (
            checkpoint.input_amplitude_normalization
        ),
        "sqrt_amplitude": bool(checkpoint.sqrt_amplitude),
        "detector_psf_sigma": float(checkpoint.detector_psf_sigma),
        "capture_repeats": int(args.capture_repeats),
        "batch_size": int(args.batch_size),
        "dark_frames": int(args.dark_frames),
        "saturation_level": args.saturation_level,
        "max_saturated_fraction": float(args.max_saturated_fraction),
        "min_dynamic_range": args.min_dynamic_range,
        "allow_saturation": bool(args.allow_saturation),
        "allow_low_dynamic_range": bool(args.allow_low_dynamic_range),
        "save_layers": args.save_layers,
        "selected_data": _selected_sample_digest(selected_samples),
        "runtime_versions": _runtime_versions(),
    }
    source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "donn_lab/experiment/optical_inference.py",
        REPO_ROOT / "donn_lab/experiment/vehicle_data.py",
        REPO_ROOT / "donn_lab/experiment/artifacts.py",
        REPO_ROOT / "donn_lab/hardware/torch_tm_backend.py",
        REPO_ROOT / "donn_lab/hardware/v4_128_backend.py",
        REPO_ROOT / "vehicle_center_loader.py",
    )
    payload["implementation"] = {
        path.relative_to(REPO_ROOT).as_posix(): quick_file_identity(path, include_hash=True)
        for path in source_paths
    }
    if args.mode == "simulate":
        payload.update(
            {
                "tmatrix": quick_file_identity(
                    _resolved_path(args.tmatrix), include_hash=bool(args.hash_tmatrix)
                ),
                "tmatrix_shape": args.tmatrix_shape,
                "tmatrix_dtype": args.tmatrix_dtype,
                "tmatrix_layout": args.tmatrix_layout,
                "device": args.device,
                "sim_noise_std": float(args.sim_noise_std),
                "sim_dark_level": float(args.sim_dark_level),
                "sim_quantize_8bit": bool(args.sim_quantize_8bit),
                "sim_seed": int(args.sim_seed),
                "stream_tm": bool(args.stream_tm),
                "tm_chunk_rows": args.tm_chunk_rows,
            }
        )
    else:
        dependency_report = _hardware_dependency_report(args)
        dependency_identities = {}
        for name, item in dependency_report.items():
            raw_path = item.get("path")
            if raw_path and Path(str(raw_path)).is_file():
                dependency_identities[name] = quick_file_identity(
                    Path(str(raw_path)), include_hash=True
                )
        payload.update(
            {
                "v4_module": quick_file_identity(_resolved_path(args.v4_module), include_hash=True),
                "v4_root": str(_resolved_path(args.v4_root)),
                "dll_parent": str(_resolved_path(args.dll_parent)),
                "camera_index": int(args.camera_index),
                "dmd_device": args.dmd_device,
                "max_retries": int(args.max_retries),
                "hardware_dependencies": dependency_identities,
            }
        )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "inputs": payload,
    }


def _batched(samples: Iterable[VehicleExperimentSample], batch_size: int) -> Iterable[List[VehicleExperimentSample]]:
    batch = []
    for sample in samples:
        batch.append(sample)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _sample_prediction(
    result: OpticalInferenceResult,
    sample: VehicleExperimentSample,
    batch_index: int,
) -> Dict[str, Any]:
    metrics = result.metrics
    pred_row, pred_col = [int(value) for value in metrics.predicted_rc[batch_index]]
    item = {
        "dataset_index": int(sample.index),
        "image_path": str(sample.image_path),
        "target_row": int(sample.target_rc[0]),
        "target_col": int(sample.target_rc[1]),
        "predicted_row": pred_row,
        "predicted_col": pred_col,
        "peak_intensity": float(metrics.peak_intensity[batch_index]),
        "peak_pbr": float(metrics.peak_pbr[batch_index]),
    }
    if metrics.pixel_distance is not None:
        item.update(
            {
                "coord_mse_argmax": float(metrics.coord_mse_argmax[batch_index]),
                "pixel_distance": float(metrics.pixel_distance[batch_index]),
                "within_5px": bool(metrics.pixel_distance[batch_index] <= 5.0),
                "within_10px": bool(metrics.pixel_distance[batch_index] <= 10.0),
                "target_intensity": float(metrics.target_intensity[batch_index]),
                "target_pbr": float(metrics.target_pbr[batch_index]),
            }
        )
    layer_quality = []
    for quality in result.layer_quality:
        layer_item = {
            "layer_index": int(quality.layer_index),
            "minimum": float(quality.minimum[batch_index]),
            "maximum": float(quality.maximum[batch_index]),
            "mean": float(quality.mean[batch_index]),
            "std": float(quality.std[batch_index]),
            "nonfinite_count": int(quality.nonfinite_count[batch_index]),
            "saturated_count": int(quality.saturated_count[batch_index]),
            "saturated_fraction": float(quality.saturated_fraction[batch_index]),
            "below_dark_fraction": float(quality.below_dark_fraction[batch_index]),
        }
        layer_quality.append(layer_item)
        item["layer_{0}_saturated_fraction".format(quality.layer_index)] = layer_item[
            "saturated_fraction"
        ]
        item["layer_{0}_dynamic_range".format(quality.layer_index)] = (
            layer_item["maximum"] - layer_item["minimum"]
        )
    item["layer_quality"] = layer_quality
    return item


def _trace_arrays(result: OpticalInferenceResult, batch_index: int, save_layers: str) -> Dict[str, Optional[np.ndarray]]:
    if save_layers == "none":
        return {
            "layer_frames": None,
            "layer_raw_frames": None,
            "layer_amplitudes": None,
            "layer_fields": None,
        }
    frames = np.stack(
        [trace.dark_corrected_intensity[batch_index] for trace in result.traces], axis=0
    ).astype(np.float32, copy=False)
    if save_layers == "frames":
        return {
            "layer_frames": frames,
            "layer_raw_frames": None,
            "layer_amplitudes": None,
            "layer_fields": None,
        }
    raw_frames = np.stack(
        [trace.raw_intensity[batch_index] for trace in result.traces], axis=0
    ).astype(np.float32, copy=False)
    amplitudes = np.stack(
        [trace.projected_amplitude[batch_index] for trace in result.traces], axis=0
    ).astype(np.float32, copy=False)
    fields = np.stack(
        [trace.projected_field[batch_index] for trace in result.traces], axis=0
    ).astype(np.complex64, copy=False)
    return {
        "layer_frames": frames,
        "layer_raw_frames": raw_frames,
        "layer_amplitudes": amplitudes,
        "layer_fields": fields,
    }


def _summary_from_manifest(path: Path, elapsed_seconds: float) -> Dict[str, Any]:
    if not path.is_file():
        return {
            "num_samples": 0,
            "invocation_elapsed_seconds": float(elapsed_seconds),
        }
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    def numbers(name: str) -> List[float]:
        values = []
        for row in rows:
            raw = row.get(name)
            if raw not in (None, ""):
                values.append(float(raw))
        return values

    summary = {
        "num_samples": len(rows),
        "invocation_elapsed_seconds": float(elapsed_seconds),
    }
    mappings = {
        "coord_mse_argmax": "coord_mse_argmax",
        "mean_pixel_distance": "pixel_distance",
        "mean_target_pbr": "target_pbr",
        "mean_peak_pbr": "peak_pbr",
    }
    for output_name, field_name in mappings.items():
        values = numbers(field_name)
        if values:
            summary[output_name] = float(np.mean(values))
    distances = numbers("pixel_distance")
    if distances:
        values_array = np.asarray(distances)
        summary["within_5px"] = float(np.mean(values_array <= 5.0))
        summary["within_10px"] = float(np.mean(values_array <= 10.0))
    layer_index = 0
    while rows and "layer_{0}_saturated_fraction".format(layer_index) in rows[0]:
        saturation = numbers("layer_{0}_saturated_fraction".format(layer_index))
        dynamic_range = numbers("layer_{0}_dynamic_range".format(layer_index))
        if saturation:
            summary["layer_{0}_mean_saturated_fraction".format(layer_index)] = float(
                np.mean(saturation)
            )
            summary["layer_{0}_max_saturated_fraction".format(layer_index)] = float(
                np.max(saturation)
            )
        if dynamic_range:
            summary["layer_{0}_mean_dynamic_range".format(layer_index)] = float(
                np.mean(dynamic_range)
            )
            summary["layer_{0}_min_dynamic_range".format(layer_index)] = float(
                np.min(dynamic_range)
            )
        layer_index += 1
    return summary


def _print_preflight(
    args: argparse.Namespace,
    checkpoint: CheckpointPhases,
    config: Mapping[str, Any],
) -> bool:
    data_root = resolve_data_root(dict(config), args.data_root)
    first_sample = next(
        iter_vehicle_samples(
            dict(config),
            data_root=data_root,
            split=args.split,
            indices=args.sample_indices,
            start_index=args.start_index,
            max_samples=1,
        )
    )
    report = {
        "status": "preflight-only; no camera/DMD device object was constructed",
        "mode": args.mode,
        "python": sys.executable,
        "checkpoint": checkpoint.checkpoint_path,
        "checkpoint_epoch": checkpoint.epoch,
        "num_layers": checkpoint.num_layers,
        "phase_shape": list(checkpoint.phases.shape),
        "input_amplitude_normalization": (
            checkpoint.input_amplitude_normalization
        ),
        "sqrt_amplitude": bool(checkpoint.sqrt_amplitude),
        "data_root": str(data_root),
        "first_sample": {
            "sample_id": first_sample.sample_id,
            "input_shape": list(first_sample.input_chw.shape),
            "target_rc": list(first_sample.target_rc),
        },
    }
    if args.mode == "hardware":
        report["hardware_dependencies"] = _hardware_dependency_report(args)
        missing = [
            name
            for name, item in report["hardware_dependencies"].items()
            if not bool(item.get("exists", item.get("available", False)))
        ]
        report["hardware_dependencies_ok"] = not missing
        report["missing"] = missing
        if not missing:
            try:
                backend = _build_backend(args, checkpoint)
                report["software_stack"] = dict(backend.software_preflight())
                report["software_stack_ok"] = True
            except Exception as exc:
                report["software_stack_ok"] = False
                report["software_stack_error"] = "{0}: {1}".format(
                    type(exc).__name__, exc
                )
    else:
        report["tmatrix"] = quick_file_identity(_resolved_path(args.tmatrix), include_hash=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return bool(report.get("hardware_dependencies_ok", True)) and bool(
        report.get("software_stack_ok", True)
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.checkpoint = _resolved_path(args.checkpoint)
    args.tmatrix = _resolved_path(args.tmatrix)
    if args.dark_frames is None:
        args.dark_frames = 16 if args.mode == "hardware" else 0
    if args.saturation_level is None and args.mode == "hardware":
        args.saturation_level = 255.0
    if args.min_dynamic_range is None and args.mode == "hardware":
        # A 0/1 frame is indistinguishable from the observed dark read noise
        # and must never be accepted as an optical activation.
        args.min_dynamic_range = 5.0

    checkpoint = load_checkpoint_phases(
        args.checkpoint,
        expected_num_layers=5,
        expected_hw=(128, 128),
        strict=True,
    )
    if not checkpoint.config_dump:
        raise RuntimeError(
            "Checkpoint has no config_dump; exact physical replay cannot infer "
            "normalize/sqrt/spatial semantics from an external config after loading"
        )
    else:
        config = dict(checkpoint.config_dump)
        if args.config is not None:
            supplied = _load_config_file(_resolved_path(args.config))
            _cross_check_config(config, supplied)
    if str(config.get("dataset", "")).lower() != "vehicle":
        raise ValueError("The optical vehicle experiment requires dataset='vehicle'")

    if args.preflight_only:
        return 0 if _print_preflight(args, checkpoint, config) else 2

    data_root = resolve_data_root(config, args.data_root)
    # Validate sample selection and backend/QC construction before creating a
    # durable run directory. This avoids orphan run.json files on pure input
    # errors such as an out-of-range sample index or malformed TM arguments.
    selected_samples = list(
        iter_vehicle_samples(
            config,
            data_root=data_root,
            split=args.split,
            indices=args.sample_indices,
            start_index=args.start_index,
            max_samples=args.max_samples,
        )
    )
    backend = _build_backend(args, checkpoint)
    settings = OpticalInferenceSettings(
        capture_repeats=int(args.capture_repeats),
        auto_capture_dark=False,
        dark_capture_repeats=max(int(args.dark_frames), 1),
        saturation_level=args.saturation_level,
        max_saturated_fraction=float(args.max_saturated_fraction),
        fail_on_saturation=(args.mode == "hardware" and not args.allow_saturation),
        min_dynamic_range=args.min_dynamic_range,
        fail_on_low_dynamic_range=(
            args.mode == "hardware" and not args.allow_low_dynamic_range
        ),
        fail_on_nonfinite=True,
    )
    fingerprint = _fingerprint_payload(args, checkpoint, data_root, selected_samples)
    output_dir = _resolved_path(args.output_dir) if args.output_dir else _default_output_dir(args.mode)
    writer = RunArtifactWriter(
        output_dir,
        {
            "schema_version": 2,
            "fingerprint": fingerprint,
            "mode": args.mode,
            "python": sys.executable,
            "checkpoint_epoch": checkpoint.epoch,
            "checkpoint_global_step": checkpoint.global_step,
            "input_amplitude_normalization": (
                checkpoint.input_amplitude_normalization
            ),
            "sqrt_amplitude": bool(checkpoint.sqrt_amplitude),
            "resolved_config": config,
            "command": [sys.executable] + sys.argv,
        },
        resume=bool(args.resume),
    )
    writer.prepare()

    samples = (
        sample for sample in selected_samples if not writer.sample_done(sample.sample_id)
    )
    first_pending = next(samples, None)
    if first_pending is None:
        summary = _summary_from_manifest(writer.manifest_path, 0.0)
        summary["status"] = "complete"
        summary["note"] = "No pending samples; hardware and TM backend were not opened."
        prior_run = json.loads(writer.run_path.read_text(encoding="utf-8"))
        summary["cumulative_elapsed_seconds"] = float(
            prior_run.get("cumulative_elapsed_seconds", 0.0)
        )
        writer.write_summary(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("Artifacts: {0}".format(output_dir))
        return 0
    samples = itertools.chain((first_pending,), samples)
    started = time.monotonic()
    run_status = "failed"
    return_code = 0
    runner = None
    captured_error = None
    try:
        dark_path = output_dir / "calibration" / "dark.npy"
        dark_identity = None
        dark_frame = None
        if args.resume and args.dark_frames > 0:
            existing_run = json.loads(writer.run_path.read_text(encoding="utf-8"))
            if dark_path.is_file():
                expected_dark = existing_run.get("dark_frame", {}).get("file_identity")
                dark_identity = quick_file_identity(dark_path, include_hash=True)
                if (
                    isinstance(expected_dark, dict)
                    and expected_dark.get("sha256") != dark_identity.get("sha256")
                ):
                    raise RuntimeError("Dark calibration fingerprint differs from run.json")
                dark_frame = np.load(str(dark_path), allow_pickle=False)
            else:
                completed_before = int(
                    _summary_from_manifest(writer.manifest_path, 0.0).get("num_samples", 0)
                )
                if completed_before > 0:
                    raise RuntimeError(
                        "Cannot resume completed samples because calibration/dark.npy is missing"
                    )

        if args.mode == "hardware":
            backend.open(arm=True)
        else:
            backend.open()

        if dark_frame is None and args.dark_frames > 0:
            captured_dark = backend.capture_dark(
                batch_size=1, capture_repeats=int(args.dark_frames)
            )
            dark_frame = np.mean(np.asarray(captured_dark, dtype=np.float32), axis=0, keepdims=True)
            dark_path = writer.save_calibration_array("dark", dark_frame)
            dark_identity = quick_file_identity(dark_path, include_hash=True)
        elif dark_frame is None:
            dark_frame = np.zeros((1, checkpoint.output_hw[0], checkpoint.output_hw[1]), dtype=np.float32)

        runner = OpticalInferenceRunner(
            checkpoint=checkpoint,
            backend=backend,
            settings=settings,
            dark_frame=dark_frame,
        )
        runner.open()
        writer.update_run_metadata(
            {
                "backend": dict(runner.metadata),
                "dark_frame": {
                    "shape": list(dark_frame.shape),
                    "minimum": float(np.min(dark_frame)),
                    "maximum": float(np.max(dark_frame)),
                    "mean": float(np.mean(dark_frame)),
                    "file_identity": dark_identity,
                },
            }
        )

        processed_now = 0
        for batch_number, batch in enumerate(_batched(samples, int(args.batch_size)), start=1):
            images = np.stack([sample.input_chw for sample in batch], axis=0).astype(np.float32, copy=False)
            targets = np.asarray([sample.target_rc for sample in batch], dtype=np.int64)
            result = runner.infer(
                images,
                targets=targets,
                return_traces=args.save_layers != "none",
            )
            for batch_index, sample in enumerate(batch):
                prediction = _sample_prediction(result, sample, batch_index)
                trace_arrays = _trace_arrays(result, batch_index, args.save_layers)
                writer.write_sample(
                    sample_id=sample.sample_id,
                    input_chw=sample.input_chw,
                    final_intensity=result.final_intensity[batch_index],
                    prediction=prediction,
                    **trace_arrays
                )
                processed_now += 1
            batch_summary = result.metrics.summary()
            print(
                "batch={0} samples={1} processed_now={2} mean_distance={3:.4f}px mean_target_pbr={4:.4f}".format(
                    batch_number,
                    len(batch),
                    processed_now,
                    float(batch_summary.get("mean_pixel_distance", float("nan"))),
                    float(batch_summary.get("mean_target_pbr", float("nan"))),
                )
            )
        run_status = "complete"
    except KeyboardInterrupt:
        print("Interrupted; completed sample directories remain resumable.", file=sys.stderr)
        return_code = 130
        run_status = "interrupted"
    except Exception as exc:
        captured_error = exc
        run_status = "failed"
    finally:
        metadata_before_close = dict(backend.metadata) if hasattr(backend, "metadata") else {}
        close_exception = None
        try:
            if runner is not None:
                runner.close()
            else:
                backend.close()
        except Exception as exc:
            close_exception = exc
            if captured_error is None:
                captured_error = exc
            run_status = "failed_cleanup"
        metadata_after_close = dict(backend.metadata) if hasattr(backend, "metadata") else {}
        elapsed = time.monotonic() - started
        summary = _summary_from_manifest(writer.manifest_path, elapsed)
        cleanup_errors = metadata_after_close.get("last_close_errors", [])
        if close_exception is not None:
            cleanup_errors = list(cleanup_errors) + [
                "{0}: {1}".format(type(close_exception).__name__, close_exception)
            ]
        if cleanup_errors and run_status == "complete":
            run_status = "failed_cleanup"
            return_code = 1
        summary["status"] = run_status
        summary["invocation_elapsed_seconds"] = float(elapsed)
        prior_run = json.loads(writer.run_path.read_text(encoding="utf-8"))
        cumulative_elapsed = float(prior_run.get("cumulative_elapsed_seconds", 0.0)) + float(elapsed)
        summary["cumulative_elapsed_seconds"] = cumulative_elapsed
        summary["cleanup_errors"] = cleanup_errors
        if captured_error is not None:
            summary["error"] = "{0}: {1}".format(
                type(captured_error).__name__, captured_error
            )
        writer.write_summary(summary)
        writer.update_run_metadata(
            {
                "final_backend_before_close": metadata_before_close,
                "final_backend_after_close": metadata_after_close,
                "final_status": run_status,
                "last_invocation_elapsed_seconds": float(elapsed),
                "cumulative_elapsed_seconds": cumulative_elapsed,
            }
        )

    if captured_error is not None:
        raise captured_error

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Artifacts: {0}".format(output_dir))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from donn_lab.experiment.optical_inference import load_checkpoint_phases
from donn_lab.hardware.torch_tm_backend import TorchTMBackend
from donn_lab.hardware.v4_128_backend import V4128Backend
from donn_lab.optics.detector_psf import GaussianIntensityPSF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare arbitrary-amplitude and phase-only TM predictions on hardware."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tmatrix", type=Path, required=True)
    parser.add_argument(
        "--sample-dir",
        type=Path,
        required=True,
        help="A save-layers=all sample directory containing layer_amplitudes.npz.",
    )
    parser.add_argument("--target-row", type=int, required=True)
    parser.add_argument("--target-col", type=int, required=True)
    parser.add_argument("--mapped-amplitude-min", type=float, default=0.4)
    parser.add_argument("--mapped-amplitude-max", type=float, default=0.8)
    parser.add_argument("--skip-focus-control", action="store_true")
    parser.add_argument("--dark-frames", type=int, default=16)
    parser.add_argument("--capture-repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--v4-root", type=Path, default=Path(r"C:\Users\smart\Documents\TMCalib"))
    parser.add_argument("--v4-module", type=Path, default=REPO_ROOT / "combined_app_v4_128.py")
    parser.add_argument("--dll-parent", type=Path, default=Path(r"C:\Users\smart\Documents"))
    parser.add_argument("--camera-save-path", type=Path, default=REPO_ROOT / "camera_1")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--arm-hardware", action="store_true")
    args = parser.parse_args()
    if not args.arm_hardware:
        parser.error("--arm-hardware is required before opening the camera and DMD")
    if args.dark_frames <= 0 or args.capture_repeats <= 0:
        parser.error("--dark-frames and --capture-repeats must be positive")
    if not (0 <= args.target_row < 128 and 0 <= args.target_col < 128):
        parser.error("target row/column must be in [0, 127]")
    if not (
        math.isfinite(args.mapped_amplitude_min)
        and math.isfinite(args.mapped_amplitude_max)
        and 0.0 <= args.mapped_amplitude_min < args.mapped_amplitude_max <= 1.0
    ):
        parser.error("mapped amplitude range must satisfy 0 <= min < max <= 1")
    return args


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else 0.0


def peak_rc(image: np.ndarray) -> Tuple[int, int]:
    index = int(np.argmax(image))
    return index // image.shape[1], index % image.shape[1]


def pbr(image: np.ndarray, row: int, col: int) -> float:
    array = np.asarray(image, dtype=np.float64)
    target = float(array[row, col])
    background = float((array.sum() - target) / max(1, array.size - 1))
    return target / max(background, 1e-12)


def d4_correlations(prediction: np.ndarray, measured: np.ndarray) -> Dict[str, float]:
    transforms = {
        "identity": lambda x: x,
        "flip_ud": np.flipud,
        "flip_lr": np.fliplr,
        "rot180": lambda x: np.rot90(x, 2),
        "transpose": np.transpose,
        "transpose_flip_ud": lambda x: np.flipud(x.T),
        "transpose_flip_lr": lambda x: np.fliplr(x.T),
        "transpose_rot180": lambda x: np.rot90(x.T, 2),
    }
    return {name: pearson(prediction, transform(measured)) for name, transform in transforms.items()}


def make_figure(
    output: Path,
    names: Tuple[str, ...],
    predictions: np.ndarray,
    measurements: np.ndarray,
    target: Tuple[int, int],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(names), figsize=(5 * len(names), 9), constrained_layout=True)
    for index, name in enumerate(names):
        for row_index, (data, prefix) in enumerate(
            ((predictions[index], "TM prediction"), (measurements[index], "Hardware"))
        ):
            lower = float(np.percentile(data, 1.0))
            upper = float(np.percentile(data, 99.8))
            if upper <= lower:
                upper = lower + 1.0
            axes[row_index, index].imshow(data, cmap="inferno", vmin=lower, vmax=upper)
            axes[row_index, index].set_title("{}\n{}".format(prefix, name))
            axes[row_index, index].scatter(
                [target[1]], [target[0]], s=95, facecolors="none", edgecolors="lime", linewidths=1.5
            )
            peak = peak_rc(data)
            axes[row_index, index].scatter(
                [peak[1]], [peak[0]], s=75, marker="x", c="cyan", linewidths=1.5
            )
            axes[row_index, index].set_xticks([])
            axes[row_index, index].set_yticks([])
    fig.suptitle("Green: requested target; cyan: measured/predicted peak\nPanels use independent 1-99.8 percentile scaling")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    checkpoint = load_checkpoint_phases(args.checkpoint, expected_num_layers=5, expected_hw=(128, 128))
    amplitudes = np.load(args.sample_dir / "layer_amplitudes.npz")["amplitudes"]
    if amplitudes.shape != (5, 128, 128):
        raise ValueError("Expected layer amplitudes with shape (5,128,128), got {}".format(amplitudes.shape))

    phase0 = checkpoint.phases[0].astype(np.float32, copy=False)
    network_amplitude = amplitudes[0].astype(np.float32, copy=False)
    amplitude_span = float(network_amplitude.max() - network_amplitude.min())
    if amplitude_span <= 0.0:
        raise ValueError("The saved network amplitude is constant")
    mapped_amplitude = (
        float(args.mapped_amplitude_min)
        + (network_amplitude - float(network_amplitude.min()))
        / amplitude_span
        * (float(args.mapped_amplitude_max) - float(args.mapped_amplitude_min))
    ).astype(np.float32, copy=False)
    unit_amplitude = np.ones((128, 128), dtype=np.float32)

    field_list = [
        network_amplitude * np.exp(1j * phase0),
        mapped_amplitude * np.exp(1j * phase0),
        unit_amplitude * np.exp(1j * phase0),
    ]
    amplitude_list = [network_amplitude, mapped_amplitude, unit_amplitude]
    names = [
        "network_amplitude",
        "mapped_amplitude_{:.3g}_{:.3g}".format(
            args.mapped_amplitude_min, args.mapped_amplitude_max
        ),
        "same_phase_unit_amplitude",
    ]
    if not args.skip_focus_control:
        tmatrix = np.load(args.tmatrix, mmap_mode="r")
        if tmatrix.shape != (128 * 128, 128 * 128):
            raise ValueError("Expected a (16384,16384) TM, got {}".format(tmatrix.shape))
        target_index = int(args.target_row) * 128 + int(args.target_col)
        target_row = np.asarray(tmatrix[target_index], dtype=np.complex64)
        focus_phase = -np.angle(target_row).astype(np.float32, copy=False)
        del tmatrix
        field_list.append(unit_amplitude * np.exp(1j * focus_phase.reshape(128, 128)))
        amplitude_list.append(unit_amplitude)
        names.append("target_phase_conjugate")
    fields = np.stack(field_list, axis=0).astype(np.complex64, copy=False)
    names = tuple(names)

    simulator = TorchTMBackend(
        tmatrix_path=args.tmatrix,
        input_hw=(128, 128),
        output_hw=(128, 128),
        device=args.device,
        detector_psf_sigma=0.0,
    )
    with simulator:
        prediction_raw = simulator.project_and_capture_fields(fields)
    psf = GaussianIntensityPSF(float(checkpoint.detector_psf_sigma))
    import torch

    with torch.no_grad():
        prediction_psf = psf(torch.from_numpy(prediction_raw[:, None]))[:, 0].numpy()

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = REPO_ROOT / "runs" / "tm_phase_only_control" / stamp
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    backend = V4128Backend(
        v4_root=args.v4_root,
        save_path=args.camera_save_path,
        module_path=args.v4_module,
        dll_parent=args.dll_parent,
        max_retries=2,
    )
    hardware_raw = None
    dark = None
    before_close: Dict[str, Any] = {}
    try:
        backend.open(arm=True)
        dark_stack = backend.capture_dark(batch_size=args.dark_frames, capture_repeats=1)
        dark = np.median(dark_stack, axis=0).astype(np.float32)
        hardware_raw = backend.project_and_capture_fields(fields, capture_repeats=args.capture_repeats)
        before_close = dict(backend.metadata)
    finally:
        backend.close()
    after_close = dict(backend.metadata)
    if hardware_raw is None or dark is None:
        raise RuntimeError("Hardware capture did not produce data")
    hardware = np.maximum(hardware_raw - dark[None], 0.0).astype(np.float32)

    records = []
    for index, name in enumerate(names):
        d4 = d4_correlations(prediction_psf[index], hardware[index])
        best_transform = max(d4, key=d4.get)
        target = (int(args.target_row), int(args.target_col))
        pred_peak = peak_rc(prediction_psf[index])
        measured_peak = peak_rc(hardware[index])
        records.append(
            {
                "name": name,
                "input_amplitude_min": float(amplitude_list[index].min()),
                "input_amplitude_max": float(amplitude_list[index].max()),
                "input_amplitude_mean": float(amplitude_list[index].mean()),
                "pearson_raw_tm": pearson(prediction_raw[index], hardware[index]),
                "pearson_psf_tm": d4["identity"],
                "best_d4_transform": best_transform,
                "best_d4_pearson": float(d4[best_transform]),
                "predicted_peak_rc": list(pred_peak),
                "hardware_peak_rc": list(measured_peak),
                "target_pbr_prediction": pbr(prediction_psf[index], *target),
                "target_pbr_hardware": pbr(hardware[index], *target),
                "hardware_min": float(hardware[index].min()),
                "hardware_max": float(hardware[index].max()),
                "hardware_mean": float(hardware[index].mean()),
                "hardware_std": float(hardware[index].std()),
                "saturated_fraction": float(np.mean(hardware_raw[index] >= 255.0)),
            }
        )

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "tmatrix": str(args.tmatrix.resolve()),
        "sample_dir": str(args.sample_dir.resolve()),
        "target_rc": [int(args.target_row), int(args.target_col)],
        "detector_psf_sigma": float(checkpoint.detector_psf_sigma),
        "dark_frames": int(args.dark_frames),
        "capture_repeats": int(args.capture_repeats),
        "records": records,
        "hardware_metadata_before_close": before_close,
        "hardware_metadata_after_close": after_close,
    }
    np.savez_compressed(
        output_dir / "arrays.npz",
        fields=fields,
        prediction_raw=prediction_raw,
        prediction_psf=prediction_psf,
        hardware_raw=hardware_raw,
        dark=dark,
        hardware_corrected=hardware,
    )
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_figure(
        output_dir / "comparison.png",
        names,
        prediction_psf,
        hardware,
        (int(args.target_row), int(args.target_col)),
    )
    print(json.dumps({"output_dir": str(output_dir), "records": records}, indent=2))
    cleanup_errors = after_close.get("last_close_errors", [])
    return 2 if cleanup_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

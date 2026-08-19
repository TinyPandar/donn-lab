#!/usr/bin/env python
"""Measure the end-to-end complex-amplitude response of the real DMD path.

A small, spatially distributed set of input modes remains at unit amplitude so
that the external 4x4 superpixel encoder cannot divide away the commanded test
amplitude.  The remaining modes are swept over several amplitudes while their
phase and the reference mask stay fixed within each seed.

For every command level, the script compares the measured camera speckle with
``abs(TM @ field) ** 2``.  It also fits an effective amplitude by exploiting
the fact that detector intensity is quadratic in the swept amplitude.
Hardware access is guarded by ``--arm-hardware``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_tm_output_correlation import (  # noqa: E402
    DEFAULT_DLL_PARENT,
    DEFAULT_V4_MODULE,
    DEFAULT_V4_ROOT,
    FieldRecord,
    _capture,
    _json_dump,
    _metrics,
    _predict,
    _resolved,
    _utc_now,
)


DEFAULT_TMATRIX = Path(
    r"C:\Users\smart\Documents\TMCalib\reconstructed_field_128_px4_active512_8N.npy"
)
DEFAULT_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def _parse_levels(values: Sequence[float]) -> Tuple[float, ...]:
    levels = tuple(sorted({float(value) for value in values}))
    if not levels:
        raise ValueError("At least one amplitude level is required")
    if levels[0] < 0.0 or levels[-1] > 1.0:
        raise ValueError("Amplitude levels must be within [0, 1]")
    for required in (0.0, 0.5, 1.0):
        if not any(abs(level - required) <= 1e-8 for level in levels):
            raise ValueError(
                "Amplitude levels must include 0, 0.5, and 1 for response fitting"
            )
    return levels


def _make_records(
    levels: Sequence[float],
    phase_seeds: int,
    reference_fraction: float,
    seed: int,
) -> List[FieldRecord]:
    mode_count = 128 * 128
    reference_count = max(1, int(round(mode_count * reference_fraction)))
    seed_data = []  # type: List[Tuple[np.ndarray, np.ndarray]]
    for seed_index in range(phase_seeds):
        rng = np.random.default_rng(seed + seed_index * 1009)
        reference_flat = rng.choice(
            mode_count, size=reference_count, replace=False
        )
        reference_mask = np.zeros(mode_count, dtype=bool)
        reference_mask[reference_flat] = True
        reference_mask = reference_mask.reshape(128, 128)
        phase_levels = rng.integers(0, 16, size=(128, 128), dtype=np.int16)
        phase = phase_levels.astype(np.float32) * np.float32(
            2.0 * np.pi / 16.0
        )
        phasor = np.exp(1j * phase).astype(np.complex64)
        seed_data.append((reference_mask, phasor))

    records = []  # type: List[FieldRecord]
    for level in levels:
        for seed_index, (reference_mask, phasor) in enumerate(seed_data):
            amplitude = np.full((128, 128), level, dtype=np.float32)
            amplitude[reference_mask] = 1.0
            field = np.ascontiguousarray(amplitude * phasor, dtype=np.complex64)
            records.append(
                FieldRecord(
                    group="amplitude_response",
                    label="amplitude_{0:.3f}_seed_{1:02d}".format(
                        level, seed_index
                    ),
                    field=field,
                    metadata={
                        "commanded_amplitude": float(level),
                        "seed_index": int(seed_index),
                        "reference_fraction": float(reference_fraction),
                        "reference_count": int(reference_count),
                        "phase_levels": 16,
                    },
                )
            )
    return records


def _pearson_grid(
    measured: np.ndarray,
    constant: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
    candidates: np.ndarray,
) -> Tuple[float, float]:
    measured_flat = np.asarray(measured, dtype=np.float64).reshape(-1)
    measured_flat -= measured_flat.mean()
    measured_norm = float(np.linalg.norm(measured_flat))
    if measured_norm <= 0.0 or not math.isfinite(measured_norm):
        return float("nan"), float("nan")

    values = (
        constant[None, :, :]
        + candidates[:, None, None] * linear[None, :, :]
        + np.square(candidates[:, None, None]) * quadratic[None, :, :]
    ).reshape(candidates.size, -1)
    values = np.asarray(values, dtype=np.float64)
    values -= values.mean(axis=1, keepdims=True)
    denominators = np.linalg.norm(values, axis=1) * measured_norm
    correlations = np.full(candidates.shape, np.nan, dtype=np.float64)
    valid = denominators > 0.0
    correlations[valid] = (
        values[valid] @ measured_flat
    ) / denominators[valid]
    if not np.any(np.isfinite(correlations)):
        return float("nan"), float("nan")
    best_index = int(np.nanargmax(correlations))
    return float(candidates[best_index]), float(correlations[best_index])


def _fit_effective_amplitudes(
    records: Sequence[FieldRecord],
    predicted: np.ndarray,
    measured: np.ndarray,
    phase_seeds: int,
) -> List[Tuple[float, float]]:
    lookup = {}  # type: Dict[Tuple[int, float], int]
    for index, record in enumerate(records):
        lookup[
            (
                int(record.metadata["seed_index"]),
                float(record.metadata["commanded_amplitude"]),
            )
        ] = index

    candidates = np.linspace(0.0, 1.0, 201, dtype=np.float64)
    polynomial = {}  # type: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    for seed_index in range(phase_seeds):
        intensity_zero = np.asarray(
            predicted[lookup[(seed_index, 0.0)]], dtype=np.float64
        )
        intensity_half = np.asarray(
            predicted[lookup[(seed_index, 0.5)]], dtype=np.float64
        )
        intensity_one = np.asarray(
            predicted[lookup[(seed_index, 1.0)]], dtype=np.float64
        )
        difference_one = intensity_one - intensity_zero
        difference_half = intensity_half - intensity_zero
        quadratic = 2.0 * difference_one - 4.0 * difference_half
        linear = difference_one - quadratic
        polynomial[seed_index] = (intensity_zero, linear, quadratic)

    fitted = []  # type: List[Tuple[float, float]]
    for index, record in enumerate(records):
        seed_index = int(record.metadata["seed_index"])
        constant, linear, quadratic = polynomial[seed_index]
        fitted.append(
            _pearson_grid(
                measured[index], constant, linear, quadratic, candidates
            )
        )
    return fitted


def _level_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output = {}  # type: Dict[str, Any]
    levels = sorted({float(row["commanded_amplitude"]) for row in rows})
    for level in levels:
        selected = [
            row for row in rows
            if float(row["commanded_amplitude"]) == level
        ]
        pearson = np.asarray(
            [float(row["pearson"]) for row in selected], dtype=np.float64
        )
        repeat = np.asarray(
            [float(row["repeat_pearson"]) for row in selected], dtype=np.float64
        )
        fitted = np.asarray(
            [float(row["best_fit_amplitude"]) for row in selected],
            dtype=np.float64,
        )
        fitted_pearson = np.asarray(
            [float(row["best_fit_pearson"]) for row in selected],
            dtype=np.float64,
        )
        dynamic_range = np.asarray(
            [float(row["measured_dynamic_range"]) for row in selected],
            dtype=np.float64,
        )
        output["{0:.3f}".format(level)] = {
            "count": len(selected),
            "pearson_mean": float(np.nanmean(pearson)),
            "pearson_std": float(np.nanstd(pearson)),
            "repeat_pearson_mean": float(np.nanmean(repeat)),
            "best_fit_amplitude_mean": float(np.nanmean(fitted)),
            "best_fit_amplitude_std": float(np.nanstd(fitted)),
            "best_fit_pearson_mean": float(np.nanmean(fitted_pearson)),
            "dynamic_range_mean": float(np.nanmean(dynamic_range)),
            "max_saturated_fraction": float(
                max(float(row["measured_saturated_fraction"]) for row in selected)
            ),
        }
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "index",
        "label",
        "commanded_amplitude",
        "seed_index",
        "reference_fraction",
        "pearson",
        "repeat_pearson",
        "best_fit_amplitude",
        "best_fit_pearson",
        "amplitude_error",
        "normalized_rmse",
        "measured_dynamic_range",
        "measured_saturated_fraction",
        "correlation_valid",
        "best_dihedral",
        "best_dihedral_pearson",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[FieldRecord],
    predicted: np.ndarray,
    measured: np.ndarray,
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = sorted({float(row["commanded_amplitude"]) for row in rows})
    means = []
    stds = []
    fit_means = []
    fit_stds = []
    dynamics = []
    for level in levels:
        selected = [row for row in rows if float(row["commanded_amplitude"]) == level]
        correlations = np.asarray([row["pearson"] for row in selected], dtype=float)
        fits = np.asarray([row["best_fit_amplitude"] for row in selected], dtype=float)
        dynamic = np.asarray([row["measured_dynamic_range"] for row in selected], dtype=float)
        means.append(float(np.nanmean(correlations)))
        stds.append(float(np.nanstd(correlations)))
        fit_means.append(float(np.nanmean(fits)))
        fit_stds.append(float(np.nanstd(fits)))
        dynamics.append(float(np.nanmean(dynamic)))

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    axes[0].errorbar(levels, means, yerr=stds, marker="o", capsize=3)
    axes[0].set(xlabel="Commanded test amplitude", ylabel="Prediction-camera Pearson")
    axes[0].grid(alpha=0.3)
    axes[1].errorbar(levels, fit_means, yerr=fit_stds, marker="o", capsize=3)
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    axes[1].set(xlabel="Commanded test amplitude", ylabel="Best-fit effective amplitude")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].plot(levels, dynamics, marker="o")
    axes[2].set(xlabel="Commanded test amplitude", ylabel="Camera dynamic range (DN)")
    axes[2].grid(alpha=0.3)
    figure.tight_layout()
    response_path = output_dir / "amplitude_response.png"
    figure.savefig(str(response_path), dpi=170)
    plt.close(figure)

    example_indices = [
        index for index, record in enumerate(records)
        if int(record.metadata["seed_index"]) == 0
    ]
    figure, axes = plt.subplots(
        len(example_indices), 3,
        figsize=(10.5, max(3.0, 2.7 * len(example_indices))),
        squeeze=False,
    )
    for plot_row, index in enumerate(example_indices):
        pred = np.maximum(predicted[index], 0.0)
        meas = np.maximum(measured[index], 0.0)
        pred = pred / max(float(pred.max()), 1e-12)
        meas = meas / max(float(meas.max()), 1e-12)
        difference = meas - pred
        axes[plot_row, 0].imshow(pred, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[plot_row, 1].imshow(meas, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[plot_row, 2].imshow(difference, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[plot_row, 0].set_ylabel(
            "a={0:.2f}".format(float(records[index].metadata["commanded_amplitude"]))
        )
        for axis in axes[plot_row]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title("TM prediction")
    axes[0, 1].set_title("Camera")
    axes[0, 2].set_title("Camera - prediction")
    figure.tight_layout()
    comparison_path = output_dir / "seed0_comparisons.png"
    figure.savefig(str(comparison_path), dpi=150)
    plt.close(figure)
    return [str(response_path), str(comparison_path)]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmatrix", type=Path, default=DEFAULT_TMATRIX)
    parser.add_argument("--levels", type=float, nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--phase-seeds", type=int, default=4)
    parser.add_argument("--reference-fraction", type=float, default=1.0 / 16.0)
    parser.add_argument(
        "--laser-power-mw",
        type=float,
        default=None,
        help="Laser power recorded as experiment metadata; this does not control the laser.",
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stream-tm", action="store_true")
    parser.add_argument("--tm-chunk-rows", type=int, default=None)
    parser.add_argument("--capture-repeats", type=int, default=3)
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
    try:
        args.levels = _parse_levels(args.levels)
    except ValueError as error:
        parser.error(str(error))
    if args.phase_seeds <= 0:
        parser.error("--phase-seeds must be positive")
    if not 0.0 < args.reference_fraction < 1.0:
        parser.error("--reference-fraction must be between 0 and 1")
    if args.laser_power_mw is not None and args.laser_power_mw <= 0.0:
        parser.error("--laser-power-mw must be positive")
    if args.capture_repeats <= 0 or args.dark_frames <= 0:
        parser.error("capture repeats and dark frames must be positive")
    if args.tm_chunk_rows is not None and args.tm_chunk_rows <= 0:
        parser.error("--tm-chunk-rows must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.tmatrix = _resolved(args.tmatrix)
    args.v4_root = _resolved(args.v4_root)
    args.v4_module = _resolved(args.v4_module)
    args.dll_parent = _resolved(args.dll_parent)
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = REPO_ROOT / "runs/dmd_amplitude_response" / stamp
    else:
        args.output_dir = _resolved(args.output_dir)

    required = {
        "tmatrix": args.tmatrix,
        "v4_root": args.v4_root,
        "v4_module": args.v4_module,
        "dll_parent": args.dll_parent,
    }
    missing = [
        "{0}={1}".format(name, path)
        for name, path in required.items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + "; ".join(missing))

    preflight = {
        "status": "ready" if args.arm_hardware else "preflight_only",
        "tmatrix": str(args.tmatrix),
        "levels": list(args.levels),
        "phase_seeds": int(args.phase_seeds),
        "reference_fraction": float(args.reference_fraction),
        "laser_power_mw": args.laser_power_mw,
        "capture_repeats": int(args.capture_repeats),
        "dark_frames": int(args.dark_frames),
        "num_fields": int(len(args.levels) * args.phase_seeds),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(preflight, indent=2, ensure_ascii=False), flush=True)
    if not args.arm_hardware:
        print("Preflight only; no camera or DMD object was constructed.", flush=True)
        return 0
    if args.output_dir.exists():
        raise FileExistsError("Output directory already exists: {0}".format(args.output_dir))
    args.output_dir.mkdir(parents=True)

    records = _make_records(
        args.levels,
        args.phase_seeds,
        args.reference_fraction,
        args.seed,
    )
    fields = np.stack([record.field for record in records], axis=0).astype(np.complex64)
    print("Predicting {0} mixed-amplitude fields...".format(len(records)), flush=True)
    predicted = _predict(
        args.tmatrix,
        fields,
        args.device,
        args.stream_tm,
        args.tm_chunk_rows,
    )
    print("Capturing fields on hardware...", flush=True)
    measured_raw, measured_raw_repeats, dark, backend_metadata = _capture(args, fields)
    measured = np.maximum(measured_raw - dark[None, :, :], 0.0).astype(np.float32)
    measured_repeats = np.maximum(
        measured_raw_repeats - dark[None, None, :, :], 0.0
    ).astype(np.float32)

    fitted = _fit_effective_amplitudes(
        records, predicted, measured, args.phase_seeds
    )
    rows = []  # type: List[Dict[str, Any]]
    for index, record in enumerate(records):
        row = {
            "index": int(index),
            "label": record.label,
            "commanded_amplitude": float(record.metadata["commanded_amplitude"]),
            "seed_index": int(record.metadata["seed_index"]),
            "reference_fraction": float(record.metadata["reference_fraction"]),
        }
        row.update(
            _metrics(
                predicted[index],
                measured_raw[index],
                measured[index],
                measured_raw_repeats[index],
                measured_repeats[index],
            )
        )
        row["best_fit_amplitude"] = float(fitted[index][0])
        row["best_fit_pearson"] = float(fitted[index][1])
        row["amplitude_error"] = float(
            fitted[index][0] - float(record.metadata["commanded_amplitude"])
        )
        rows.append(row)

    by_level = _level_summary(rows)
    for key, values in by_level.items():
        print(
            "a={0} r={1:.4f}+/-{2:.4f} fit={3:.3f}+/-{4:.3f} "
            "fit_r={5:.4f} repeat={6:.4f} range={7:.1f}DN sat={8:.4%}".format(
                key,
                values["pearson_mean"],
                values["pearson_std"],
                values["best_fit_amplitude_mean"],
                values["best_fit_amplitude_std"],
                values["best_fit_pearson_mean"],
                values["repeat_pearson_mean"],
                values["dynamic_range_mean"],
                values["max_saturated_fraction"],
            ),
            flush=True,
        )

    np.save(str(args.output_dir / "fields.npy"), fields)
    np.save(str(args.output_dir / "predicted_intensity.npy"), predicted)
    np.save(str(args.output_dir / "measured_raw.npy"), measured_raw)
    np.save(str(args.output_dir / "measured_raw_repeats.npy"), measured_raw_repeats)
    np.save(str(args.output_dir / "dark.npy"), dark)
    np.save(str(args.output_dir / "measured_corrected.npy"), measured)
    _write_csv(args.output_dir / "records.csv", rows)
    plots = _write_plots(args.output_dir, rows, records, predicted, measured)
    summary = {
        "status": "complete",
        "created_at_utc": _utc_now(),
        "tmatrix": str(args.tmatrix),
        "tmatrix_stat": {
            "size_bytes": int(args.tmatrix.stat().st_size),
            "mtime_ns": int(args.tmatrix.stat().st_mtime_ns),
        },
        "method": {
            "description": "mixed amplitude with distributed unit-amplitude reference modes",
            "levels": list(args.levels),
            "phase_seeds": int(args.phase_seeds),
            "phase_levels": 16,
            "reference_fraction": float(args.reference_fraction),
            "laser_power_mw": args.laser_power_mw,
            "capture_repeats": int(args.capture_repeats),
            "dark_frames": int(args.dark_frames),
            "effective_amplitude_grid_step": 0.005,
        },
        "dark": {
            "minimum": float(np.min(dark)),
            "maximum": float(np.max(dark)),
            "mean": float(np.mean(dark)),
        },
        "by_level": by_level,
        "backend_metadata": dict(backend_metadata),
        "records": rows,
        "plots": plots,
    }
    _json_dump(args.output_dir / "summary.json", summary)
    print("Artifacts: {0}".format(args.output_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _centered_fft2(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(field), norm="ortho")
    ).astype(np.complex64)


def _centered_ifft2(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(field), norm="ortho")
    ).astype(np.complex64)


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def _dihedral_variants(image: np.ndarray) -> Iterable[Tuple[str, np.ndarray]]:
    yield "identity", image
    yield "flip_ud", np.flipud(image)
    yield "flip_lr", np.fliplr(image)
    yield "flip_both", np.flip(image, axis=(0, 1))
    transposed = image.T
    yield "transpose", transposed
    yield "transpose_flip_ud", np.flipud(transposed)
    yield "transpose_flip_lr", np.fliplr(transposed)
    yield "transpose_flip_both", np.flip(transposed, axis=(0, 1))


def _best_dihedral(
    measured: np.ndarray, predicted: np.ndarray
) -> Tuple[str, float]:
    candidates = [
        (name, _pearson(measured, transformed))
        for name, transformed in _dihedral_variants(predicted)
    ]
    return max(candidates, key=lambda item: item[1])


def _best_circular_shift(
    measured: np.ndarray, predicted: np.ndarray
) -> Tuple[str, float, Tuple[int, int]]:
    measured_zero = np.asarray(measured, dtype=np.float64) - float(
        np.mean(measured)
    )
    best = ("", -math.inf, (0, 0))
    for name, transformed in _dihedral_variants(predicted):
        predicted_zero = np.asarray(transformed, dtype=np.float64) - float(
            np.mean(transformed)
        )
        denominator = float(
            np.linalg.norm(measured_zero) * np.linalg.norm(predicted_zero)
        )
        if denominator <= 0.0:
            continue
        correlation = np.fft.ifft2(
            np.fft.fft2(measured_zero)
            * np.conj(np.fft.fft2(predicted_zero))
        ).real / denominator
        row, col = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
        row_shift = int(row if row <= correlation.shape[0] // 2 else row - correlation.shape[0])
        col_shift = int(col if col <= correlation.shape[1] // 2 else col - correlation.shape[1])
        value = float(correlation[row, col])
        if value > best[1]:
            best = (name, value, (row_shift, col_shift))
    return best


def _choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    candidates = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            free_bytes, _ = torch.cuda.mem_get_info()
        candidates.append((int(free_bytes), index))
    return torch.device("cuda:{0}".format(max(candidates)[1]))


def _load_npz_array(path: Path, key: str, layer_index: int) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as archive:
        if key not in archive:
            raise KeyError("{0} does not contain key {1!r}".format(path, key))
        array = np.asarray(archive[key])
    if array.ndim != 3 or not 0 <= layer_index < array.shape[0]:
        raise ValueError(
            "{0}:{1} must be [layers,H,W], got {2}".format(
                path, key, array.shape
            )
        )
    return np.ascontiguousarray(array[layer_index])


def _multiply_fields(
    tmatrix_path: Path,
    fields: Sequence[np.ndarray],
    *,
    device: torch.device,
    chunk_rows: int,
) -> np.ndarray:
    matrix = np.load(str(tmatrix_path), mmap_mode="r", allow_pickle=False)
    mode_count = int(fields[0].size)
    if matrix.shape != (mode_count, mode_count):
        raise ValueError(
            "TM shape {0} does not match field size {1}".format(
                matrix.shape, mode_count
            )
        )
    if matrix.dtype != np.dtype(np.complex64):
        raise ValueError("TM must be complex64, got {0}".format(matrix.dtype))

    vectors = np.stack(
        [np.asarray(field, dtype=np.complex64).reshape(-1) for field in fields]
    )
    vector_tensor = torch.from_numpy(vectors).to(device=device).transpose(0, 1)
    output = np.empty((len(fields), mode_count), dtype=np.complex64)
    with torch.inference_mode():
        for start in range(0, mode_count, chunk_rows):
            stop = min(start + chunk_rows, mode_count)
            host_chunk = np.array(matrix[start:stop], dtype=np.complex64, copy=True)
            matrix_chunk = torch.from_numpy(host_chunk).to(device=device)
            result = torch.matmul(matrix_chunk, vector_tensor).transpose(0, 1)
            output[:, start:stop] = result.cpu().numpy()
            del matrix_chunk, result, host_chunk
    return output


def _intensity(field: np.ndarray) -> np.ndarray:
    return (
        np.asarray(field.real, dtype=np.float64) ** 2
        + np.asarray(field.imag, dtype=np.float64) ** 2
    ).astype(np.float32)


def _fresnel_transfer(field: np.ndarray, beta: float) -> np.ndarray:
    """Same-sampling Fresnel transfer with dimensionless beta=lambda*z/pitch^2."""
    height, width = field.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    kernel = np.exp(-1j * np.pi * float(beta) * (fx * fx + fy * fy))
    return np.fft.ifft2(
        np.fft.fft2(field, norm="ortho") * kernel,
        norm="ortho",
    ).astype(np.complex64)


def _fresnel_betas(minimum: float, maximum: float, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    if minimum <= 0.0 or maximum <= minimum:
        raise ValueError("Fresnel beta bounds require 0 < minimum < maximum")
    positive = np.geomspace(minimum, maximum, count, dtype=np.float64)
    return np.concatenate((-positive[::-1], np.asarray([0.0]), positive))


def _normalise_for_display(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    low = float(np.min(image))
    high = float(np.max(image))
    return ((image - low) / (high - low + 1e-12)).astype(np.float32)


def _write_preview(
    output_path: Path,
    measured: np.ndarray,
    ranked: Sequence[Dict[str, object]],
    intensity_by_name: Dict[str, np.ndarray],
) -> None:
    chosen = list(ranked[:5])
    figure, axes = plt.subplots(2, 3, figsize=(11.5, 7.6), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(_normalise_for_display(measured), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Hardware camera")
    axes[0].axis("off")
    for axis, record in zip(axes[1:], chosen):
        name = str(record["name"])
        axis.imshow(
            _normalise_for_display(intensity_by_name[name]),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axis.set_title(
            "{0}\nr={1:.4f}, shifted={2:.4f}".format(
                name,
                float(record["pearson_identity"]),
                float(record["best_circular_correlation"]),
            ),
            fontsize=9,
        )
        axis.axis("off")
    figure.suptitle("TM propagation hypotheses (each image independently normalized)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare explicit FFT/IFFT hypotheses against a recorded camera frame."
    )
    parser.add_argument("--tmatrix", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--captured", type=Path, required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--fresnel-beta-min", type=float, default=1e-2)
    parser.add_argument("--fresnel-beta-max", type=float, default=1e5)
    parser.add_argument(
        "--fresnel-beta-count",
        type=int,
        default=0,
        help="Positive log-spaced beta samples; both signs plus zero are tested.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    tmatrix_path = _resolve(args.tmatrix)
    fields_path = _resolve(args.fields)
    captured_path = _resolve(args.captured)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logical_field = _load_npz_array(fields_path, "fields", args.layer_index).astype(
        np.complex64, copy=False
    )
    measured = _load_npz_array(captured_path, "frames", args.layer_index).astype(
        np.float32, copy=False
    )
    if logical_field.shape != measured.shape:
        raise ValueError(
            "Field shape {0} and capture shape {1} differ".format(
                logical_field.shape, measured.shape
            )
        )

    input_fields = {
        "input_direct": logical_field,
        "input_fft": _centered_fft2(logical_field),
        "input_ifft": _centered_ifft2(logical_field),
        "input_conjugate": np.conj(logical_field).astype(np.complex64),
    }
    device = _choose_device(str(args.device))
    print("Computing four TM matrix-vector products on {0}".format(device), flush=True)
    detector_vectors = _multiply_fields(
        tmatrix_path,
        list(input_fields.values()),
        device=device,
        chunk_rows=int(args.chunk_rows),
    )

    intensity_by_name: Dict[str, np.ndarray] = {}
    records: List[Dict[str, object]] = []
    output_operations = {
        "output_direct": lambda value: value,
        "output_fft": _centered_fft2,
        "output_ifft": _centered_ifft2,
    }
    height, width = logical_field.shape
    for input_index, input_name in enumerate(input_fields):
        detector_field = detector_vectors[input_index].reshape(height, width)
        for output_name, operation in output_operations.items():
            name = input_name + "__" + output_name
            candidate_field = operation(detector_field)
            predicted = _intensity(candidate_field)
            intensity_by_name[name] = predicted
            transform_name, transform_corr = _best_dihedral(measured, predicted)
            shift_transform, shift_corr, shift_rc = _best_circular_shift(
                measured, predicted
            )
            records.append(
                {
                    "name": name,
                    "pearson_identity": _pearson(measured, predicted),
                    "best_dihedral_transform": transform_name,
                    "best_dihedral_correlation": transform_corr,
                    "best_circular_transform": shift_transform,
                    "best_circular_correlation": shift_corr,
                    "best_circular_shift_rc": list(shift_rc),
                    "predicted_min": float(np.min(predicted)),
                    "predicted_max": float(np.max(predicted)),
                }
            )

    ranked = sorted(
        records,
        key=lambda record: float(record["best_circular_correlation"]),
        reverse=True,
    )
    fresnel_records: List[Dict[str, object]] = []
    betas = _fresnel_betas(
        float(args.fresnel_beta_min),
        float(args.fresnel_beta_max),
        int(args.fresnel_beta_count),
    )
    if betas.size:
        print("Scanning {0} dimensionless Fresnel beta values".format(betas.size), flush=True)
        direct_detector = detector_vectors[0].reshape(height, width)
        for beta in betas:
            propagated = _fresnel_transfer(direct_detector, float(beta))
            predicted = _intensity(propagated)
            transform_name, transform_corr = _best_dihedral(measured, predicted)
            fresnel_records.append(
                {
                    "beta": float(beta),
                    "pearson_identity": _pearson(measured, predicted),
                    "best_dihedral_transform": transform_name,
                    "best_dihedral_correlation": transform_corr,
                }
            )
        fresnel_records.sort(
            key=lambda record: float(record["best_dihedral_correlation"]),
            reverse=True,
        )
    payload = {
        "tmatrix": str(tmatrix_path),
        "fields": str(fields_path),
        "captured": str(captured_path),
        "layer_index": int(args.layer_index),
        "device": str(device),
        "measured_min": float(np.min(measured)),
        "measured_max": float(np.max(measured)),
        "ranking_metric": "best circular Pearson correlation across 8 dihedral transforms",
        "candidates": ranked,
        "fresnel_parameter": "beta = wavelength * propagation_distance / output_pitch^2",
        "fresnel_scan_warning": (
            "The GGS21 intensity-only reconstruction leaves an independent output-pixel "
            "phase gauge, so cross-pixel Fresnel propagation is only a numerical hypothesis."
        ),
        "fresnel_candidates": fresnel_records,
    }
    json_path = output_dir / "fft_hypotheses.json"
    temporary_path = json_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(str(temporary_path), str(json_path))
    np.savez_compressed(
        str(output_dir / "candidate_intensities.npz"),
        **{name: value for name, value in intensity_by_name.items()},
    )
    _write_preview(
        output_dir / "fft_hypotheses_preview.png",
        measured,
        ranked,
        intensity_by_name,
    )

    print("\nRanked candidates:")
    for record in ranked:
        print(
            "{0:42s} identity={1:+.5f} dihedral={2:+.5f} circular={3:+.5f} shift={4}".format(
                str(record["name"]),
                float(record["pearson_identity"]),
                float(record["best_dihedral_correlation"]),
                float(record["best_circular_correlation"]),
                record["best_circular_shift_rc"],
            )
        )
    if fresnel_records:
        print("\nTop dimensionless Fresnel candidates (no circular shift search):")
        for record in fresnel_records[:10]:
            print(
                "beta={0:+.8g} identity={1:+.5f} dihedral={2:+.5f} transform={3}".format(
                    float(record["beta"]),
                    float(record["pearson_identity"]),
                    float(record["best_dihedral_correlation"]),
                    record["best_dihedral_transform"],
                )
            )
    print("Results: {0}".format(json_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

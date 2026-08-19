"""Torch-backed camera simulator for a measured transmission matrix.

This module intentionally has no dependency on the training configuration or
model classes.  It implements the same optical operation used by the measured
TM model::

    intensity = abs(H @ field.reshape(-1)) ** 2

The matrix may stay memory-mapped on the host and be streamed to the selected
device in row chunks, or it may be cached on the device for repeated, low
latency captures.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from donn_lab.optics.detector_psf import GaussianIntensityPSF


ShapeLike = Optional[Union[str, Sequence[int]]]


def _parse_shape(shape: ShapeLike) -> Optional[Tuple[int, int]]:
    if shape is None:
        return None
    if isinstance(shape, str):
        parts = [part for part in re.split(r"[xX,\s*]+", shape.strip()) if part]
        if len(parts) != 2:
            raise ValueError("tmatrix_shape must contain exactly two dimensions")
        return int(parts[0]), int(parts[1])
    if len(shape) != 2:
        raise ValueError("tmatrix_shape must contain exactly two dimensions")
    return int(shape[0]), int(shape[1])


def _parse_hw(name: str, value: Sequence[int]) -> Tuple[int, int]:
    if len(value) != 2:
        raise ValueError("{} must be a (height, width) pair".format(name))
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError("{} dimensions must be positive".format(name))
    return height, width


def _numpy_dtype(name: str) -> np.dtype:
    choices = {
        "complex64": np.complex64,
        "complex128": np.complex128,
        "float32": np.float32,
        "float64": np.float64,
    }
    key = str(name).lower()
    if key not in choices:
        raise ValueError(
            "Unsupported tmatrix_dtype {!r}; expected one of {}".format(
                name, sorted(choices)
            )
        )
    return np.dtype(choices[key])


def _load_tmatrix_mmap(
    path: Path,
    shape: Optional[Tuple[int, int]],
    dtype: np.dtype,
) -> np.ndarray:
    """Load a TM without importing the Python-3.10-only training helpers."""
    try:
        loaded = np.load(str(path), mmap_mode="r", allow_pickle=False)
    except Exception:
        if shape is None:
            raise ValueError(
                "{} is not a NumPy array; tmatrix_shape is required for a raw file".format(
                    path
                )
            )
        return np.memmap(str(path), dtype=dtype, mode="r", shape=shape)

    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if not loaded.files:
                raise ValueError("TM archive is empty: {}".format(path))
            key = "H" if "H" in loaded.files else loaded.files[0]
            array = loaded[key]
        finally:
            loaded.close()
        return np.asarray(array)
    return np.asarray(loaded)


class TorchTMBackend:
    """Simulate a scattering medium and camera with a measured complex TM.

    Parameters are expressed in camera intensity units.  ``dark_level`` is
    added before optional Gaussian ``noise_std``.  When ``quantize8`` is true,
    every individual capture is rounded and clipped to ``[0, 255]`` before
    repeated captures are averaged.
    """

    def __init__(
        self,
        tmatrix_path: Union[str, Path],
        input_hw: Sequence[int] = (128, 128),
        output_hw: Sequence[int] = (128, 128),
        tmatrix_shape: ShapeLike = None,
        tmatrix_dtype: str = "complex64",
        layout: str = "out_in",
        device: Union[str, torch.device] = "auto",
        chunk_rows: Optional[int] = None,
        cache_on_device: bool = True,
        noise_std: float = 0.0,
        dark_level: float = 0.0,
        quantize8: bool = False,
        seed: Optional[int] = None,
        detector_psf_sigma: float = 0.0,
    ) -> None:
        self.tmatrix_path = Path(tmatrix_path).expanduser()
        self.input_hw = _parse_hw("input_hw", input_hw)
        self.output_hw = _parse_hw("output_hw", output_hw)
        self.tmatrix_shape = _parse_shape(tmatrix_shape)
        self.tmatrix_dtype = str(tmatrix_dtype).lower()
        self.layout = str(layout).lower()
        if self.layout not in {
            "out_in",
            "nout_nin",
            "rows_out",
            "in_out",
            "nin_nout",
            "cols_out",
        }:
            raise ValueError("layout must be 'out_in' or 'in_out'")

        if isinstance(device, str) and device.lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

        if chunk_rows is not None and int(chunk_rows) <= 0:
            raise ValueError("chunk_rows must be positive or None")
        if float(noise_std) < 0.0:
            raise ValueError("noise_std must be non-negative")
        if float(dark_level) < 0.0:
            raise ValueError("dark_level must be non-negative")

        self.chunk_rows = None if chunk_rows is None else int(chunk_rows)
        self.cache_on_device = bool(cache_on_device)
        self.noise_std = float(noise_std)
        self.dark_level = float(dark_level)
        self.quantize8 = bool(quantize8)
        self.seed = None if seed is None else int(seed)
        self.detector_psf = GaussianIntensityPSF(detector_psf_sigma).to(self.device)
        self.detector_psf_sigma = float(self.detector_psf.sigma)

        # Populated by open().  Keeping the mmap alive avoids loading a second
        # 2 GiB host copy when H is cached on CUDA.
        self._tm_array = None  # type: Optional[np.ndarray]
        self._tm_device = None  # type: Optional[torch.Tensor]
        self._generator = None  # type: Optional[torch.Generator]
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def metadata(self) -> Dict[str, Any]:
        matrix_shape = None
        if self._tm_array is not None:
            matrix_shape = [int(self._tm_array.shape[0]), int(self._tm_array.shape[1])]
        elif self.tmatrix_shape is not None:
            matrix_shape = [int(self.tmatrix_shape[0]), int(self.tmatrix_shape[1])]
        return {
            "backend": "torch_tm",
            "tmatrix_path": str(self.tmatrix_path.resolve()),
            "tmatrix_shape": matrix_shape,
            "tmatrix_dtype": self.tmatrix_dtype,
            "layout": "out_in",
            "source_layout": self.layout,
            "input_hw": [self.input_hw[0], self.input_hw[1]],
            "output_hw": [self.output_hw[0], self.output_hw[1]],
            "device": str(self.device),
            "chunk_rows": self.chunk_rows,
            "cache_on_device": self.cache_on_device,
            "noise_std": self.noise_std,
            "dark_level": self.dark_level,
            "quantize8": self.quantize8,
            "seed": self.seed,
            "detector_psf_sigma": self.detector_psf_sigma,
            "detector_psf_kernel_size": self.detector_psf.kernel_size,
            "is_open": self._is_open,
        }

    def open(self) -> "TorchTMBackend":
        """Open the matrix and optionally cache it on the compute device."""
        if self._is_open:
            return self
        if not self.tmatrix_path.is_file():
            raise FileNotFoundError("TM file not found: {}".format(self.tmatrix_path))

        parsed_dtype = _numpy_dtype(self.tmatrix_dtype)
        matrix = _load_tmatrix_mmap(
            self.tmatrix_path, shape=self.tmatrix_shape, dtype=parsed_dtype
        )
        if matrix.ndim == 3 and matrix.shape[-1] == 2 and not np.iscomplexobj(matrix):
            matrix = matrix[..., 0] + 1j * matrix[..., 1]
        if matrix.ndim != 2:
            raise ValueError("TM must be a 2D matrix, got shape {}".format(matrix.shape))
        if self.tmatrix_shape is not None and tuple(matrix.shape) != self.tmatrix_shape:
            raise ValueError(
                "TM shape mismatch: got {}, expected {}".format(
                    tuple(matrix.shape), self.tmatrix_shape
                )
            )
        if self.layout in {"in_out", "nin_nout", "cols_out"}:
            matrix = matrix.T

        expected_shape = (
            self.output_hw[0] * self.output_hw[1],
            self.input_hw[0] * self.input_hw[1],
        )
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                "TM shape must be [N_out,N_in] = {}, got {} after applying layout".format(
                    expected_shape, tuple(matrix.shape)
                )
            )
        self._tm_array = matrix

        if self.cache_on_device:
            try:
                self._tm_device = self._array_to_complex_tensor(matrix, self.device)
            except RuntimeError as exc:
                self._tm_device = None
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                self._tm_array = None
                raise RuntimeError(
                    "Could not cache the TM on {}. Set cache_on_device=False and "
                    "choose chunk_rows for streamed multiplication.".format(self.device)
                ) from exc

        if self.seed is not None:
            self._generator = torch.Generator(device=self.device.type)
            self._generator.manual_seed(self.seed)
        self._is_open = True
        return self

    @staticmethod
    def _array_to_complex_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
        with warnings.catch_warnings():
            if not array.flags.writeable:
                warnings.filterwarnings(
                    "ignore", message="The given NumPy array is not writable"
                )
            tensor = torch.as_tensor(array)
        if not torch.is_complex(tensor):
            tensor = torch.complex(tensor.float(), torch.zeros_like(tensor, dtype=torch.float32))
        return tensor.to(device=device, dtype=torch.complex64).contiguous()

    def _require_open(self) -> None:
        if not self._is_open or self._tm_array is None:
            raise RuntimeError("TorchTMBackend is closed; call open() first")

    @staticmethod
    def _positive_int(name: str, value: int) -> int:
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError("{} must be a positive integer".format(name))
        return int(value)

    def _multiply_intensity(self, fields: np.ndarray) -> torch.Tensor:
        self._require_open()
        field_array = np.asarray(fields)
        if field_array.ndim != 3:
            raise ValueError("fields must have shape [B,H,W], got {}".format(field_array.shape))
        if tuple(field_array.shape[1:]) != self.input_hw:
            raise ValueError(
                "field shape must be [B,{},{}], got {}".format(
                    self.input_hw[0], self.input_hw[1], tuple(field_array.shape)
                )
            )
        if field_array.shape[0] <= 0:
            raise ValueError("fields batch must not be empty")
        if not np.iscomplexobj(field_array):
            raise TypeError("fields must contain complex-valued optical fields")
        if not np.all(np.isfinite(field_array)):
            raise ValueError("fields contain NaN or infinite values")

        field_array = np.ascontiguousarray(field_array, dtype=np.complex64)
        field_tensor = torch.from_numpy(field_array).to(
            device=self.device, dtype=torch.complex64
        )
        field_vectors_t = field_tensor.reshape(field_tensor.shape[0], -1).transpose(0, 1)
        output_dim = self.output_hw[0] * self.output_hw[1]
        step = output_dim if self.chunk_rows is None else min(self.chunk_rows, output_dim)
        intensity = torch.empty(
            (field_tensor.shape[0], output_dim), device=self.device, dtype=torch.float32
        )

        with torch.inference_mode():
            for start in range(0, output_dim, step):
                end = min(start + step, output_dim)
                if self._tm_device is not None:
                    matrix_chunk = self._tm_device[start:end]
                else:
                    # Only the active rows are copied to the compute device.
                    matrix_chunk = self._array_to_complex_tensor(
                        self._tm_array[start:end], self.device
                    )
                detector = torch.matmul(matrix_chunk, field_vectors_t).transpose(0, 1)
                # real**2 + imag**2 avoids Torch's CUDA complex-abs JIT kernel,
                # which otherwise creates an unnecessary runtime NVRTC
                # dependency on some Windows installations.
                intensity[:, start:end] = detector.real.square() + detector.imag.square()
                del detector
                if self._tm_device is None:
                    del matrix_chunk
        intensity_4d = intensity.reshape(
            field_tensor.shape[0], 1, self.output_hw[0], self.output_hw[1]
        )
        return self.detector_psf(intensity_4d).reshape(field_tensor.shape[0], output_dim)

    def _sample_sensor(self, ideal: torch.Tensor, capture_repeats: int) -> torch.Tensor:
        repeats = self._positive_int("capture_repeats", capture_repeats)

        # The ideal path is deliberately allocation-light and exactly matches
        # abs(H @ field)^2 in float32.
        if self.dark_level == 0.0 and self.noise_std == 0.0 and not self.quantize8:
            return ideal

        accumulated = torch.zeros_like(ideal)
        for _ in range(repeats):
            capture = ideal + self.dark_level
            if self.noise_std > 0.0:
                noise = torch.randn(
                    capture.shape,
                    dtype=capture.dtype,
                    device=capture.device,
                    generator=self._generator,
                )
                capture = capture + noise * self.noise_std
            capture = torch.clamp(capture, min=0.0)
            if self.quantize8:
                capture = torch.round(torch.clamp(capture, max=255.0))
            accumulated.add_(capture)
        return accumulated / float(repeats)

    def project_and_capture_fields(
        self, fields: np.ndarray, capture_repeats: int = 1
    ) -> np.ndarray:
        """Propagate fields and return camera intensities as ``float32 [B,H,W]``."""
        repeats = self._positive_int("capture_repeats", capture_repeats)
        ideal = self._multiply_intensity(fields)
        captured = self._sample_sensor(ideal, repeats)
        return (
            captured.reshape(-1, self.output_hw[0], self.output_hw[1])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def capture_dark(
        self, batch_size: int = 1, capture_repeats: int = 1
    ) -> np.ndarray:
        """Return dark camera captures without performing a TM multiplication."""
        self._require_open()
        batch = self._positive_int("batch_size", batch_size)
        repeats = self._positive_int("capture_repeats", capture_repeats)
        ideal = torch.zeros(
            (batch, self.output_hw[0] * self.output_hw[1]),
            dtype=torch.float32,
            device=self.device,
        )
        captured = self._sample_sensor(ideal, repeats)
        return (
            captured.reshape(batch, self.output_hw[0], self.output_hw[1])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def close(self) -> None:
        """Release device and memory-mapped matrix references."""
        self._tm_device = None
        self._tm_array = None
        self._generator = None
        self._is_open = False

    def __enter__(self) -> "TorchTMBackend":
        return self.open()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = ["TorchTMBackend"]

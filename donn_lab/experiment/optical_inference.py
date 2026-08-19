"""Model-faithful closed-loop inference for a measured-TM optical network.

This module deliberately has no dependency on a particular DMD or camera SDK.
The hardware-specific code only has to implement :class:`OpticalBatchBackend`:
accept a batch of complex fields and return the corresponding camera intensity
frames.  All preprocessing and feedback between optical layers lives here so
that simulated and physical runs use exactly the same network semantics.

Coordinates in this module are always ``(row, column)`` / ``(y, x)``.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F


PathLike = Union[str, os.PathLike]


@runtime_checkable
class OpticalBatchBackend(Protocol):
    """Minimum contract shared by simulated and physical optical backends.

    ``fields`` has shape ``[B, H_mode, W_mode]`` and dtype ``complex64``.  A
    backend must encode every field it receives, including an all-zero field;
    the core intentionally does not invent a special DMD "black" pattern.
    Returned frames must be camera intensities with shape ``[B, H_out, W_out]``.

    Implementations may additionally expose ``open()``, ``close()``,
    ``metadata`` (a mapping or a no-argument method returning one), and
    ``capture_dark(batch_size=1, capture_repeats=1)``.  They are optional and
    are discovered at runtime.
    """

    def project_and_capture_fields(
        self,
        fields: np.ndarray,
        capture_repeats: int = 1,
    ) -> np.ndarray:
        """Project complex fields and return averaged float32 intensities."""


@dataclass(frozen=True)
class CheckpointPhases:
    """Validated phase masks and the inference-relevant checkpoint metadata."""

    phases: np.ndarray
    checkpoint_path: str
    input_hw: Tuple[int, int]
    mode_hw: Tuple[int, int]
    output_hw: Tuple[int, int]
    normalize_input: bool
    sqrt_amplitude: bool
    input_amplitude_normalization: str = "minmax"
    tmatrix_normalization: str = "none"
    detector_psf_sigma: float = 0.0
    epoch: Optional[int] = None
    global_step: Optional[int] = None
    model_name: Optional[str] = None
    config_dump: Mapping[str, Any] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        return int(self.phases.shape[0])


@dataclass(frozen=True)
class OpticalInferenceSettings:
    """Acquisition and quality-control settings for closed-loop inference."""

    capture_repeats: int = 1
    auto_capture_dark: bool = True
    dark_capture_repeats: int = 8
    saturation_level: Optional[float] = None
    max_saturated_fraction: float = 0.001
    fail_on_saturation: bool = False
    min_dynamic_range: Optional[float] = None
    fail_on_low_dynamic_range: bool = False
    fail_on_nonfinite: bool = True
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if int(self.capture_repeats) <= 0:
            raise ValueError("capture_repeats must be a positive integer")
        if int(self.dark_capture_repeats) <= 0:
            raise ValueError("dark_capture_repeats must be a positive integer")
        if self.saturation_level is not None:
            if not math.isfinite(float(self.saturation_level)) or float(self.saturation_level) <= 0.0:
                raise ValueError("saturation_level must be a finite positive number")
        if not 0.0 <= float(self.max_saturated_fraction) <= 1.0:
            raise ValueError("max_saturated_fraction must be in [0, 1]")
        if self.min_dynamic_range is not None:
            if not math.isfinite(float(self.min_dynamic_range)) or float(self.min_dynamic_range) < 0.0:
                raise ValueError("min_dynamic_range must be finite and non-negative")
        if not math.isfinite(float(self.eps)) or float(self.eps) <= 0.0:
            raise ValueError("eps must be a finite positive number")


@dataclass(frozen=True)
class CameraQuality:
    """Per-sample camera quality statistics for one optical layer."""

    layer_index: int
    minimum: np.ndarray
    maximum: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    nonfinite_count: np.ndarray
    saturated_count: np.ndarray
    saturated_fraction: np.ndarray
    below_dark_count: np.ndarray
    below_dark_fraction: np.ndarray
    saturation_level: Optional[float]
    saturation_warning: bool

    def summary(self) -> Dict[str, Any]:
        """Return a compact JSON-friendly summary for logs/manifests."""

        return {
            "layer_index": int(self.layer_index),
            "minimum": float(np.min(self.minimum)),
            "maximum": float(np.max(self.maximum)),
            "mean": float(np.mean(self.mean)),
            "std_mean": float(np.mean(self.std)),
            "nonfinite_count": int(np.sum(self.nonfinite_count)),
            "saturated_count": int(np.sum(self.saturated_count)),
            "max_saturated_fraction": float(np.max(self.saturated_fraction)),
            "below_dark_count": int(np.sum(self.below_dark_count)),
            "max_below_dark_fraction": float(np.max(self.below_dark_fraction)),
            "saturation_level": (
                None if self.saturation_level is None else float(self.saturation_level)
            ),
            "saturation_warning": bool(self.saturation_warning),
        }


@dataclass(frozen=True)
class LayerTrace:
    """Optional full trace of one projected/captured optical layer."""

    layer_index: int
    projected_amplitude: np.ndarray
    phase_radians: np.ndarray
    projected_field: np.ndarray
    raw_intensity: np.ndarray
    dark_corrected_intensity: np.ndarray
    quality: CameraQuality


@dataclass(frozen=True)
class OpticalBatchMetrics:
    """Per-sample localization and peak-to-background metrics."""

    predicted_rc: np.ndarray
    peak_intensity: np.ndarray
    peak_background_mean: np.ndarray
    peak_pbr: np.ndarray
    target_rc: Optional[np.ndarray] = None
    coord_mse_argmax: Optional[np.ndarray] = None
    pixel_distance: Optional[np.ndarray] = None
    target_intensity: Optional[np.ndarray] = None
    target_background_mean: Optional[np.ndarray] = None
    target_pbr: Optional[np.ndarray] = None

    def summary(self) -> Dict[str, Any]:
        """Aggregate this batch using the same means as model evaluation."""

        result = {
            "num_samples": int(self.predicted_rc.shape[0]),
            "mean_peak_intensity": float(np.mean(self.peak_intensity)),
            "mean_peak_pbr": float(np.mean(self.peak_pbr)),
        }
        if self.pixel_distance is not None:
            result.update(
                {
                    "coord_mse_argmax": float(np.mean(self.coord_mse_argmax)),
                    "mean_pixel_distance": float(np.mean(self.pixel_distance)),
                    "within_5px": float(np.mean(self.pixel_distance <= 5.0)),
                    "within_10px": float(np.mean(self.pixel_distance <= 10.0)),
                    "mean_target_intensity": float(np.mean(self.target_intensity)),
                    "mean_target_pbr": float(np.mean(self.target_pbr)),
                }
            )
        return result


@dataclass(frozen=True)
class OpticalInferenceResult:
    """Final detector intensity plus metrics and per-layer diagnostics."""

    final_intensity: np.ndarray
    metrics: OpticalBatchMetrics
    layer_quality: Tuple[CameraQuality, ...]
    traces: Tuple[LayerTrace, ...]
    dark_frame: np.ndarray
    backend_metadata: Mapping[str, Any]


_PHASE_KEY = re.compile(r"^phases\.(\d+)$")


def _torch_load_checkpoint(path: PathLike, map_location: Any) -> Any:
    # ``weights_only`` is unavailable on older torch versions used by some
    # camera-control environments.  Passing it explicitly where supported
    # also makes the trust boundary clear: experiment checkpoints are trusted.
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def _checkpoint_state(raw: Any) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(
            "Checkpoint must be a mapping, got {}".format(type(raw).__name__)
        )
    state = raw.get("model_state_dict", raw.get("state_dict", raw))
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint does not contain a mapping model state")
    return state


def _dict_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def load_checkpoint_phases(
    checkpoint_path: PathLike,
    expected_num_layers: Optional[int] = None,
    expected_hw: Optional[Sequence[int]] = None,
    map_location: Any = "cpu",
    strict: bool = True,
) -> CheckpointPhases:
    """Load and strictly validate ``phases.0 ... phases.N`` from a checkpoint.

    In strict mode, unexpected state keys are rejected and, when ``config_dump``
    is present, the model type, activation, layer count, and spatial dimensions
    are cross-checked. Returned phases preserve the checkpoint's finite
    float32 values and are stored as read-only ``[L, H_mode,W_mode]`` arrays.
    The complex exponential applies their physical ``2*pi`` periodicity; a
    separate remainder would add avoidable numerical drift from the trained
    forward pass.
    """

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(path))

    raw = _torch_load_checkpoint(path, map_location)
    state = _checkpoint_state(raw)
    phases_by_index = {}  # type: Dict[int, torch.Tensor]
    unexpected = []  # type: List[str]

    for original_key, value in state.items():
        if not isinstance(original_key, str):
            unexpected.append(repr(original_key))
            continue
        key = original_key[7:] if original_key.startswith("module.") else original_key
        match = _PHASE_KEY.fullmatch(key)
        if match is None:
            unexpected.append(original_key)
            continue
        index = int(match.group(1))
        if index in phases_by_index:
            raise ValueError("Duplicate phase key for layer {}".format(index))
        if not torch.is_tensor(value):
            raise ValueError("{} is not a tensor".format(original_key))
        phases_by_index[index] = value

    if strict and unexpected:
        raise ValueError(
            "Unexpected checkpoint state keys for measured_tm_scatter: {}".format(
                ", ".join(unexpected)
            )
        )
    if not phases_by_index:
        raise ValueError("No phase tensors matching phases.<index> were found")

    indices = sorted(phases_by_index)
    wanted_indices = list(range(len(indices)))
    if indices != wanted_indices:
        raise ValueError(
            "Phase indices must be contiguous from zero; found {}".format(indices)
        )
    if expected_num_layers is not None and len(indices) != int(expected_num_layers):
        raise ValueError(
            "Checkpoint has {} phase layers, expected {}".format(
                len(indices), int(expected_num_layers)
            )
        )

    tensors = []  # type: List[torch.Tensor]
    phase_hw = None  # type: Optional[Tuple[int, int]]
    for index in indices:
        tensor = phases_by_index[index]
        if tensor.ndim != 4 or tuple(tensor.shape[:2]) != (1, 1):
            raise ValueError(
                "phases.{} must have shape [1,1,H,W], got {}".format(
                    index, tuple(tensor.shape)
                )
            )
        current_hw = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        if current_hw[0] <= 0 or current_hw[1] <= 0:
            raise ValueError("phases.{} has an empty spatial dimension".format(index))
        if phase_hw is None:
            phase_hw = current_hw
        elif current_hw != phase_hw:
            raise ValueError(
                "All phase masks must share one shape; phases.{} is {}, expected {}".format(
                    index, current_hw, phase_hw
                )
            )
        if not torch.is_floating_point(tensor):
            raise ValueError(
                "phases.{} must be floating point, got {}".format(index, tensor.dtype)
            )
        detached = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(detached).all().item()):
            raise ValueError("phases.{} contains NaN or infinity".format(index))
        tensors.append(detached[0, 0])

    if phase_hw is None:
        raise AssertionError("phase_hw was not initialized")
    if expected_hw is not None:
        if len(expected_hw) != 2:
            raise ValueError("expected_hw must contain (height, width)")
        wanted_hw = (int(expected_hw[0]), int(expected_hw[1]))
        if phase_hw != wanted_hw:
            raise ValueError(
                "Checkpoint phase grid is {}, expected {}".format(phase_hw, wanted_hw)
            )

    config_dump = _dict_or_empty(raw.get("config_dump"))
    model_config = _dict_or_empty(config_dump.get("model_cfg"))
    data_config = _dict_or_empty(config_dump.get("data"))
    model_name_value = model_config.get("name", config_dump.get("model"))
    model_name = None if model_name_value is None else str(model_name_value)

    if strict and model_name is not None and model_name != "measured_tm_scatter":
        raise ValueError(
            "Checkpoint model is {!r}, expected 'measured_tm_scatter'".format(
                model_name
            )
        )
    configured_layers = model_config.get("num_layers")
    if configured_layers is not None and int(configured_layers) != len(indices):
        raise ValueError(
            "config_dump declares {} layers but checkpoint contains {}".format(
                int(configured_layers), len(indices)
            )
        )
    activation = model_config.get("activation")
    if strict and activation is not None and str(activation).lower() != "abs":
        raise ValueError(
            "Physical intensity feedback requires activation='abs'; checkpoint has {!r}".format(
                activation
            )
        )
    tmatrix_normalization = str(
        model_config.get("tmatrix_normalization", "none") or "none"
    ).lower()
    if strict and tmatrix_normalization not in {"none", "off", "false"}:
        raise ValueError(
            "Replay currently requires tmatrix_normalization='none'; "
            "checkpoint has {!r}".format(tmatrix_normalization)
        )
    input_amplitude_normalization = str(
        model_config.get("input_amplitude_normalization", "minmax") or "minmax"
    ).lower()
    if input_amplitude_normalization not in {"minmax", "max"}:
        raise ValueError(
            "input_amplitude_normalization must be one of {'minmax','max'}; "
            "checkpoint has {!r}".format(input_amplitude_normalization)
        )
    detector_psf_sigma = float(model_config.get("detector_psf_sigma", 0.0) or 0.0)
    if not math.isfinite(detector_psf_sigma) or detector_psf_sigma < 0.0:
        raise ValueError(
            "detector_psf_sigma must be finite and non-negative; checkpoint has {!r}".format(
                detector_psf_sigma
            )
        )

    input_hw = (
        int(data_config.get("h_in", phase_hw[0])),
        int(data_config.get("w_in", phase_hw[1])),
    )
    output_hw = (
        int(data_config.get("h_out", phase_hw[0])),
        int(data_config.get("w_out", phase_hw[1])),
    )
    mode_h = int(model_config.get("tmatrix_input_h", 0) or phase_hw[0])
    mode_w = int(model_config.get("tmatrix_input_w", 0) or phase_hw[1])
    mode_hw = (mode_h, mode_w)
    if mode_hw != phase_hw:
        raise ValueError(
            "config_dump TM input grid {} disagrees with phase grid {}".format(
                mode_hw, phase_hw
            )
        )
    if min(input_hw + output_hw + mode_hw) <= 0:
        raise ValueError("Checkpoint spatial dimensions must all be positive")

    stacked = torch.stack(tensors, dim=0).numpy().astype(np.float32, copy=False)
    stacked.setflags(write=False)

    return CheckpointPhases(
        phases=stacked,
        checkpoint_path=str(path.resolve()),
        input_hw=input_hw,
        mode_hw=mode_hw,
        output_hw=output_hw,
        normalize_input=bool(model_config.get("normalize_input", True)),
        sqrt_amplitude=bool(model_config.get("sqrt_amplitude", True)),
        input_amplitude_normalization=input_amplitude_normalization,
        tmatrix_normalization=tmatrix_normalization,
        detector_psf_sigma=detector_psf_sigma,
        epoch=_optional_int(raw.get("epoch")),
        global_step=_optional_int(raw.get("global_step")),
        model_name=model_name,
        config_dump=dict(config_dump),
    )


def normalize_spatial_per_sample(
    amplitude: torch.Tensor,
    eps: float = 1e-8,
    mode: str = "minmax",
) -> torch.Tensor:
    """Apply per-sample spatial min-max or max-only normalization."""

    if amplitude.ndim != 4:
        raise ValueError("amplitude must have shape [B,C,H,W]")
    mode_key = str(mode or "minmax").lower()
    if mode_key not in {"minmax", "max"}:
        raise ValueError("normalization mode must be one of {'minmax','max'}")
    amax = amplitude.amax(dim=(2, 3), keepdim=True)
    if mode_key == "max":
        return amplitude / (amax + float(eps))
    amin = amplitude.amin(dim=(2, 3), keepdim=True)
    return (amplitude - amin) / (amax - amin + float(eps))


def initial_amplitude(
    images: Union[torch.Tensor, np.ndarray],
    input_hw: Sequence[int],
    mode_hw: Optional[Sequence[int]] = None,
    normalize_input: bool = True,
    input_amplitude_normalization: str = "minmax",
    sqrt_amplitude: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reproduce ``MeasuredTMScatterNetwork._image_to_amplitude`` exactly.

    The returned tensor has shape ``[B,1,H_mode,W_mode]``.  A one-channel gray
    BCHW tensor is the trained experiment's normal input.  Three-channel and
    other channel counts retain the source model's RGB-weighted/mean fallback
    behavior, which is useful for parity tests and legacy datasets.
    """

    if len(input_hw) != 2:
        raise ValueError("input_hw must contain (height, width)")
    wanted_input_hw = (int(input_hw[0]), int(input_hw[1]))
    wanted_mode_hw = wanted_input_hw
    if mode_hw is not None:
        if len(mode_hw) != 2:
            raise ValueError("mode_hw must contain (height, width)")
        wanted_mode_hw = (int(mode_hw[0]), int(mode_hw[1]))

    x = images if torch.is_tensor(images) else torch.as_tensor(images)
    if x.ndim != 4:
        raise ValueError("images must be a BCHW tensor")
    if int(x.shape[0]) <= 0 or int(x.shape[1]) <= 0:
        raise ValueError("images must contain at least one sample and channel")
    actual_hw = (int(x.shape[-2]), int(x.shape[-1]))
    if actual_hw != wanted_input_hw:
        raise ValueError(
            "Input spatial size must be {}, got {}".format(
                wanted_input_hw, actual_hw
            )
        )

    x = x.float()
    if int(x.shape[1]) == 3:
        weights = torch.tensor(
            [0.299, 0.587, 0.114], dtype=x.dtype, device=x.device
        ).view(1, 3, 1, 1)
        amplitude = (x * weights).sum(dim=1, keepdim=True)
    elif int(x.shape[1]) == 1:
        amplitude = x
    else:
        amplitude = x.mean(dim=1, keepdim=True)

    if normalize_input:
        amplitude = normalize_spatial_per_sample(
            amplitude,
            eps=eps,
            mode=input_amplitude_normalization,
        )
    amplitude = torch.clamp(amplitude, min=0.0)
    if sqrt_amplitude:
        amplitude = torch.sqrt(amplitude + float(eps))
    if wanted_mode_hw != wanted_input_hw:
        amplitude = F.interpolate(amplitude, size=wanted_mode_hw, mode="area")
    return amplitude


def targets_to_rc(
    targets: Union[torch.Tensor, np.ndarray],
    batch_size: int,
    expected_hw: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Convert ``[B,2]`` coordinates or target maps to integer ``(row,col)``."""

    target = targets.detach().cpu().numpy() if torch.is_tensor(targets) else np.asarray(targets)
    if target.ndim == 2 and int(target.shape[1]) == 2:
        if int(target.shape[0]) != int(batch_size):
            raise ValueError(
                "Target batch size is {}, expected {}".format(
                    int(target.shape[0]), int(batch_size)
                )
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("Target coordinates contain NaN or infinity")
        rounded = np.rint(target)
        if not np.allclose(target, rounded, rtol=0.0, atol=1e-6):
            raise ValueError("Target coordinates must be integer-valued")
        return rounded.astype(np.int64, copy=False)

    target_map = target
    if target_map.ndim == 4 and int(target_map.shape[1]) == 1:
        target_map = target_map[:, 0]
    if target_map.ndim != 3:
        raise ValueError(
            "targets must have shape [B,2], [B,H,W], or [B,1,H,W]; got {}".format(
                tuple(target.shape)
            )
        )
    if int(target_map.shape[0]) != int(batch_size):
        raise ValueError(
            "Target batch size is {}, expected {}".format(
                int(target_map.shape[0]), int(batch_size)
            )
        )
    if not np.all(np.isfinite(target_map)):
        raise ValueError("Target maps contain NaN or infinity")
    height, width = int(target_map.shape[1]), int(target_map.shape[2])
    if expected_hw is not None:
        if len(expected_hw) != 2:
            raise ValueError("expected_hw must contain (height, width)")
        wanted_hw = (int(expected_hw[0]), int(expected_hw[1]))
        if (height, width) != wanted_hw:
            raise ValueError(
                "Target map shape {} does not match detector shape {}".format(
                    (height, width), wanted_hw
                )
            )
    flat = target_map.reshape(int(batch_size), -1).argmax(axis=1)
    return np.stack((flat // width, flat % width), axis=1).astype(np.int64)


def compute_optical_metrics(
    intensity: np.ndarray,
    target_rc: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> OpticalBatchMetrics:
    """Compute argmax localization and target/peak PBR per sample.

    PBR matches ``scripts/evaluate_measured_tm.py``: selected-pixel intensity
    divided by the mean of every other detector pixel.
    """

    frames = np.asarray(intensity, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError("intensity must have shape [B,H,W]")
    if int(frames.shape[0]) <= 0 or min(frames.shape[1:]) <= 0:
        raise ValueError("intensity must have non-empty batch and spatial dimensions")
    if not np.all(np.isfinite(frames)):
        raise ValueError("intensity contains NaN or infinity")

    batch, height, width = [int(value) for value in frames.shape]
    flat_frames = frames.reshape(batch, -1)
    flat_peak = flat_frames.argmax(axis=1)
    rows = np.arange(batch, dtype=np.int64)
    peak_intensity = flat_frames[rows, flat_peak].astype(np.float64)
    pixel_count = int(height * width)
    denom_count = max(pixel_count - 1, 1)
    totals = flat_frames.astype(np.float64).sum(axis=1)
    peak_background = (totals - peak_intensity) / float(denom_count)
    peak_pbr = np.maximum(peak_intensity, eps) / np.maximum(peak_background, eps)
    predicted_rc = np.stack((flat_peak // width, flat_peak % width), axis=1).astype(np.int64)

    if target_rc is None:
        return OpticalBatchMetrics(
            predicted_rc=predicted_rc,
            peak_intensity=peak_intensity,
            peak_background_mean=peak_background,
            peak_pbr=peak_pbr,
        )

    target = np.asarray(target_rc)
    if target.shape != (batch, 2):
        raise ValueError(
            "target_rc must have shape ({}, 2), got {}".format(batch, target.shape)
        )
    if not np.all(np.isfinite(target)):
        raise ValueError("target_rc contains NaN or infinity")
    rounded_target = np.rint(target)
    if not np.allclose(target, rounded_target, rtol=0.0, atol=1e-6):
        raise ValueError("target_rc must be integer-valued")
    target = rounded_target.astype(np.int64, copy=False)
    if (
        np.any(target[:, 0] < 0)
        or np.any(target[:, 0] >= height)
        or np.any(target[:, 1] < 0)
        or np.any(target[:, 1] >= width)
    ):
        raise ValueError(
            "target_rc is outside detector bounds (height={}, width={})".format(
                height, width
            )
        )

    target_flat = target[:, 0] * width + target[:, 1]
    target_intensity = flat_frames[rows, target_flat].astype(np.float64)
    target_background = (totals - target_intensity) / float(denom_count)
    target_pbr = np.maximum(target_intensity, eps) / np.maximum(target_background, eps)
    delta = predicted_rc.astype(np.float64) - target.astype(np.float64)
    coord_mse = np.mean(np.square(delta), axis=1)
    distance = np.sqrt(np.sum(np.square(delta), axis=1))

    return OpticalBatchMetrics(
        predicted_rc=predicted_rc,
        peak_intensity=peak_intensity,
        peak_background_mean=peak_background,
        peak_pbr=peak_pbr,
        target_rc=target,
        coord_mse_argmax=coord_mse,
        pixel_distance=distance,
        target_intensity=target_intensity,
        target_background_mean=target_background,
        target_pbr=target_pbr,
    )


def _backend_metadata(backend: OpticalBatchBackend) -> Mapping[str, Any]:
    value = getattr(backend, "metadata", {})
    if callable(value):
        value = value()
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("backend.metadata must be a mapping or return a mapping")
    return dict(value)


def _as_camera_batch(
    frames: Any,
    expected_batch: Optional[int],
    expected_hw: Tuple[int, int],
    source: str,
) -> np.ndarray:
    array = np.asarray(frames)
    if np.iscomplexobj(array):
        raise TypeError("{} returned complex data; camera frames must be intensity".format(source))
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(
            "{} must return [B,H,W] (or [H,W] for one dark frame), got {}".format(
                source, tuple(array.shape)
            )
        )
    if expected_batch is not None and int(array.shape[0]) != int(expected_batch):
        raise ValueError(
            "{} returned batch {}, expected {}".format(
                source, int(array.shape[0]), int(expected_batch)
            )
        )
    actual_hw = (int(array.shape[1]), int(array.shape[2]))
    if actual_hw != expected_hw:
        raise ValueError(
            "{} returned detector shape {}, expected {}".format(
                source, actual_hw, expected_hw
            )
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def _validate_dark_frame(
    dark_frame: Any,
    batch_size: int,
    output_hw: Tuple[int, int],
) -> np.ndarray:
    dark = _as_camera_batch(dark_frame, None, output_hw, "dark frame")
    if int(dark.shape[0]) not in (1, int(batch_size)):
        raise ValueError(
            "dark frame batch must be 1 or {}, got {}".format(
                int(batch_size), int(dark.shape[0])
            )
        )
    if not np.all(np.isfinite(dark)):
        raise ValueError("dark frame contains NaN or infinity")
    return dark


def _camera_quality(
    raw: np.ndarray,
    dark: np.ndarray,
    layer_index: int,
    settings: OpticalInferenceSettings,
) -> CameraQuality:
    spatial_axes = (1, 2)
    finite = np.isfinite(raw)
    nonfinite_count = np.sum(~finite, axis=spatial_axes, dtype=np.int64)
    safe_raw = np.where(finite, raw, np.float32(0.0))
    if settings.saturation_level is None:
        saturated = np.zeros(raw.shape, dtype=bool)
    else:
        saturated = finite & (raw >= float(settings.saturation_level))
    saturated_count = np.sum(saturated, axis=spatial_axes, dtype=np.int64)
    pixels_per_frame = int(raw.shape[1] * raw.shape[2])
    saturated_fraction = saturated_count.astype(np.float64) / float(pixels_per_frame)
    below_dark = finite & (raw < dark)
    below_dark_count = np.sum(below_dark, axis=spatial_axes, dtype=np.int64)
    below_dark_fraction = below_dark_count.astype(np.float64) / float(pixels_per_frame)
    warning = bool(np.any(saturated_fraction > settings.max_saturated_fraction))
    return CameraQuality(
        layer_index=int(layer_index),
        minimum=np.min(safe_raw, axis=spatial_axes).astype(np.float64),
        maximum=np.max(safe_raw, axis=spatial_axes).astype(np.float64),
        mean=np.mean(safe_raw, axis=spatial_axes, dtype=np.float64),
        std=np.std(safe_raw, axis=spatial_axes, dtype=np.float64),
        nonfinite_count=nonfinite_count,
        saturated_count=saturated_count,
        saturated_fraction=saturated_fraction,
        below_dark_count=below_dark_count,
        below_dark_fraction=below_dark_fraction,
        saturation_level=settings.saturation_level,
        saturation_warning=warning,
    )


class OpticalInferenceRunner:
    """Run trained phase masks as a camera-in-the-loop optical network."""

    def __init__(
        self,
        checkpoint: CheckpointPhases,
        backend: OpticalBatchBackend,
        settings: Optional[OpticalInferenceSettings] = None,
        dark_frame: Optional[np.ndarray] = None,
    ) -> None:
        if not isinstance(checkpoint, CheckpointPhases):
            raise TypeError("checkpoint must be a CheckpointPhases instance")
        if not callable(getattr(backend, "project_and_capture_fields", None)):
            raise TypeError("backend must implement project_and_capture_fields")
        self.checkpoint = checkpoint
        self.backend = backend
        self.settings = settings or OpticalInferenceSettings()
        self._dark_frame = None if dark_frame is None else np.asarray(dark_frame)
        self._opened = False

        backend_input_hw = getattr(backend, "input_hw", None)
        if backend_input_hw is not None:
            if len(backend_input_hw) != 2:
                raise ValueError("backend.input_hw must contain (height, width)")
            backend_hw = (int(backend_input_hw[0]), int(backend_input_hw[1]))
        else:
            input_height = getattr(backend, "input_height", checkpoint.mode_hw[0])
            input_width = getattr(backend, "input_width", checkpoint.mode_hw[1])
            backend_hw = (int(input_height), int(input_width))
        if backend_hw != checkpoint.mode_hw:
            raise ValueError(
                "Backend input grid {} does not match checkpoint mode grid {}".format(
                    backend_hw, checkpoint.mode_hw
                )
            )
        backend_output_hw = getattr(backend, "output_hw", None)
        if backend_output_hw is not None:
            if len(backend_output_hw) != 2:
                raise ValueError("backend.output_hw must contain (height, width)")
            detector_hw = (int(backend_output_hw[0]), int(backend_output_hw[1]))
            if detector_hw != checkpoint.output_hw:
                raise ValueError(
                    "Backend output grid {} does not match checkpoint detector grid {}".format(
                        detector_hw, checkpoint.output_hw
                    )
                )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: PathLike,
        backend: OpticalBatchBackend,
        settings: Optional[OpticalInferenceSettings] = None,
        dark_frame: Optional[np.ndarray] = None,
        expected_num_layers: Optional[int] = None,
        expected_hw: Optional[Sequence[int]] = None,
        strict: bool = True,
    ) -> "OpticalInferenceRunner":
        checkpoint = load_checkpoint_phases(
            checkpoint_path,
            expected_num_layers=expected_num_layers,
            expected_hw=expected_hw,
            map_location="cpu",
            strict=strict,
        )
        return cls(
            checkpoint=checkpoint,
            backend=backend,
            settings=settings,
            dark_frame=dark_frame,
        )

    @property
    def dark_frame(self) -> Optional[np.ndarray]:
        if self._dark_frame is None:
            return None
        return np.array(self._dark_frame, dtype=np.float32, copy=True)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return _backend_metadata(self.backend)

    def open(self) -> None:
        if self._opened:
            return
        method = getattr(self.backend, "open", None)
        if callable(method):
            method()
        self._opened = True

    def close(self) -> None:
        if not self._opened:
            return
        method = getattr(self.backend, "close", None)
        try:
            if callable(method):
                method()
        finally:
            self._opened = False

    def __enter__(self) -> "OpticalInferenceRunner":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def calibrate_dark(
        self,
        capture_repeats: Optional[int] = None,
    ) -> np.ndarray:
        """Capture and cache a dark frame, delegating zero-field handling."""

        repeats = (
            int(self.settings.dark_capture_repeats)
            if capture_repeats is None
            else int(capture_repeats)
        )
        if repeats <= 0:
            raise ValueError("capture_repeats must be positive")

        capture_method = getattr(self.backend, "capture_dark", None)
        if callable(capture_method):
            captured = capture_method(batch_size=1, capture_repeats=repeats)
            source = "backend.capture_dark"
        else:
            zero_field = np.zeros(
                (1, self.checkpoint.mode_hw[0], self.checkpoint.mode_hw[1]),
                dtype=np.complex64,
            )
            captured = self.backend.project_and_capture_fields(
                zero_field, capture_repeats=repeats
            )
            source = "backend.project_and_capture_fields(dark)"
        dark = _as_camera_batch(
            captured,
            expected_batch=None,
            expected_hw=self.checkpoint.output_hw,
            source=source,
        )
        if not np.all(np.isfinite(dark)):
            raise RuntimeError("Captured dark frame contains NaN or infinity")
        # A backend normally returns one already-averaged frame.  Averaging here
        # also makes a backend that returns several dark samples unambiguous.
        self._dark_frame = np.mean(dark, axis=0, keepdims=True, dtype=np.float32)
        return np.array(self._dark_frame, copy=True)

    def infer(
        self,
        images: Union[torch.Tensor, np.ndarray],
        targets: Optional[Union[torch.Tensor, np.ndarray]] = None,
        return_traces: bool = False,
        dark_frame: Optional[np.ndarray] = None,
    ) -> OpticalInferenceResult:
        """Run all trained layers, physically feeding each capture into the next.

        The camera output is intensity.  After dark subtraction and clamping it
        is used *directly* as the hidden-layer field amplitude, exactly as in
        :class:`MeasuredTMScatterNetwork`; no hidden-layer square root is taken.
        """

        if not self._opened:
            self.open()

        propagation_amplitude = initial_amplitude(
            images,
            input_hw=self.checkpoint.input_hw,
            mode_hw=self.checkpoint.mode_hw,
            normalize_input=self.checkpoint.normalize_input,
            input_amplitude_normalization=(
                self.checkpoint.input_amplitude_normalization
            ),
            sqrt_amplitude=self.checkpoint.sqrt_amplitude,
            eps=self.settings.eps,
        ).detach().to(device="cpu", dtype=torch.float32)
        batch_size = int(propagation_amplitude.shape[0])

        selected_dark = dark_frame
        if selected_dark is None:
            selected_dark = self._dark_frame
        if selected_dark is None and self.settings.auto_capture_dark:
            selected_dark = self.calibrate_dark()
        if selected_dark is None:
            selected_dark = np.zeros(
                (1, self.checkpoint.output_hw[0], self.checkpoint.output_hw[1]),
                dtype=np.float32,
            )
        dark = _validate_dark_frame(
            selected_dark,
            batch_size=batch_size,
            output_hw=self.checkpoint.output_hw,
        )

        quality_items = []  # type: List[CameraQuality]
        trace_items = []  # type: List[LayerTrace]
        detector_intensity = None  # type: Optional[torch.Tensor]

        for layer_index in range(self.checkpoint.num_layers):
            normalization_mode = (
                self.checkpoint.input_amplitude_normalization
                if layer_index == 0 and self.checkpoint.normalize_input
                else "minmax"
            )
            projected_amplitude = normalize_spatial_per_sample(
                propagation_amplitude,
                eps=self.settings.eps,
                mode=normalization_mode,
            )
            phase = torch.from_numpy(
                np.array(self.checkpoint.phases[layer_index], copy=True)
            ).view(1, 1, self.checkpoint.mode_hw[0], self.checkpoint.mode_hw[1])
            field_tensor = torch.polar(projected_amplitude.float(), phase)
            fields = np.ascontiguousarray(
                field_tensor[:, 0].numpy(), dtype=np.complex64
            )

            raw_capture = self.backend.project_and_capture_fields(
                fields,
                capture_repeats=int(self.settings.capture_repeats),
            )
            raw = _as_camera_batch(
                raw_capture,
                expected_batch=batch_size,
                expected_hw=self.checkpoint.output_hw,
                source="backend.project_and_capture_fields",
            )
            raw_for_trace = raw
            if return_traces:
                raw_for_trace = np.array(raw, copy=True)
            quality = _camera_quality(raw, dark, layer_index, self.settings)
            if np.any(quality.nonfinite_count > 0):
                if self.settings.fail_on_nonfinite:
                    raise RuntimeError(
                        "Camera returned non-finite pixels at optical layer {}".format(
                            layer_index
                        )
                    )
                raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            if quality.saturation_warning and self.settings.fail_on_saturation:
                raise RuntimeError(
                    "Camera saturation fraction exceeded {:.6f} at optical layer {}".format(
                        float(self.settings.max_saturated_fraction), layer_index
                    )
                )
            if self.settings.min_dynamic_range is not None:
                dynamic_range = quality.maximum - quality.minimum
                if (
                    np.any(dynamic_range < float(self.settings.min_dynamic_range))
                    and self.settings.fail_on_low_dynamic_range
                ):
                    raise RuntimeError(
                        "Camera dynamic range fell below {:.6f} at optical layer {}".format(
                            float(self.settings.min_dynamic_range), layer_index
                        )
                    )

            corrected = np.maximum(raw - dark, np.float32(0.0)).astype(
                np.float32, copy=False
            )
            detector_intensity = torch.from_numpy(
                np.ascontiguousarray(corrected)
            ).unsqueeze(1)
            quality_items.append(quality)

            if return_traces:
                trace_items.append(
                    LayerTrace(
                        layer_index=layer_index,
                        projected_amplitude=np.array(
                            projected_amplitude[:, 0].numpy(), copy=True
                        ),
                        phase_radians=np.array(
                            self.checkpoint.phases[layer_index], copy=True
                        ),
                        projected_field=np.array(fields, copy=True),
                        raw_intensity=np.array(raw_for_trace, copy=True),
                        dark_corrected_intensity=np.array(corrected, copy=True),
                        quality=quality,
                    )
                )

            if layer_index < self.checkpoint.num_layers - 1:
                if self.checkpoint.output_hw != self.checkpoint.mode_hw:
                    propagation_amplitude = F.interpolate(
                        detector_intensity,
                        size=self.checkpoint.mode_hw,
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    # Critical model behavior: captured intensity, not
                    # sqrt(captured intensity), is the next field amplitude.
                    propagation_amplitude = detector_intensity

        if detector_intensity is None:
            raise AssertionError("No optical layers were executed")
        final_intensity = np.ascontiguousarray(
            detector_intensity[:, 0].numpy(), dtype=np.float32
        )
        target_rc = None
        if targets is not None:
            target_rc = targets_to_rc(
                targets,
                batch_size=batch_size,
                expected_hw=self.checkpoint.output_hw,
            )
        metrics = compute_optical_metrics(final_intensity, target_rc=target_rc)

        return OpticalInferenceResult(
            final_intensity=final_intensity,
            metrics=metrics,
            layer_quality=tuple(quality_items),
            traces=tuple(trace_items),
            dark_frame=np.array(dark, dtype=np.float32, copy=True),
            backend_metadata=_backend_metadata(self.backend),
        )


__all__ = [
    "CameraQuality",
    "CheckpointPhases",
    "LayerTrace",
    "OpticalBatchBackend",
    "OpticalBatchMetrics",
    "OpticalInferenceResult",
    "OpticalInferenceRunner",
    "OpticalInferenceSettings",
    "compute_optical_metrics",
    "initial_amplitude",
    "load_checkpoint_phases",
    "normalize_spatial_per_sample",
    "targets_to_rc",
]

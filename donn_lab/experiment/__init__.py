"""Camera-in-the-loop experiment building blocks."""

from .optical_inference import (
    CameraQuality,
    CheckpointPhases,
    LayerTrace,
    OpticalBatchBackend,
    OpticalBatchMetrics,
    OpticalInferenceResult,
    OpticalInferenceRunner,
    OpticalInferenceSettings,
    compute_optical_metrics,
    initial_amplitude,
    load_checkpoint_phases,
    normalize_spatial_per_sample,
    targets_to_rc,
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

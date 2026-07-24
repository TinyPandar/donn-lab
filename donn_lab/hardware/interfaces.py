from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Roi:
    """Camera region of interest in pixel coordinates."""

    y: int
    x: int
    height: int
    width: int


@dataclass(frozen=True)
class HardwareConfig:
    """Small hardware-facing config that can live outside the training config."""

    input_height: int = 128
    input_width: int = 128
    camera_roi: Roi = Roi(y=0, x=0, height=128, width=128)
    exposure_ms: float = 10.0
    tm_output_dir: Path = Path("/data/donn/tm")


@runtime_checkable
class PatternProjector(Protocol):
    """Interface for an SLM/DMD/Lee-hologram display device."""

    input_height: int
    input_width: int

    def display_phase(self, phase_radians: np.ndarray) -> None:
        """Display one phase-only pattern in radians."""

    def display_complex_field(self, amplitude: np.ndarray, phase_radians: np.ndarray) -> None:
        """Display a complex input field through the device's encoding method."""


@runtime_checkable
class Camera(Protocol):
    """Interface for a camera after exposure, trigger, and ROI are configured."""

    roi: Roi

    def capture_intensity(self) -> np.ndarray:
        """Capture one intensity frame from the configured ROI."""


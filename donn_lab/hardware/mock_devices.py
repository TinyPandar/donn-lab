from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interfaces import Roi


@dataclass
class MemoryProjector:
    """In-memory projector used for local tests before real SDK integration."""

    input_height: int = 128
    input_width: int = 128

    def __post_init__(self) -> None:
        self._amplitude = np.ones((self.input_height, self.input_width), dtype=np.float32)
        self._phase = np.zeros((self.input_height, self.input_width), dtype=np.float32)

    @property
    def field_vector(self) -> np.ndarray:
        field = self._amplitude * np.exp(1j * self._phase)
        return field.reshape(-1).astype(np.complex64)

    def display_phase(self, phase_radians: np.ndarray) -> None:
        self._phase = np.asarray(phase_radians, dtype=np.float32).reshape(self.input_height, self.input_width)
        self._amplitude = np.ones_like(self._phase, dtype=np.float32)

    def display_complex_field(self, amplitude: np.ndarray, phase_radians: np.ndarray) -> None:
        self._amplitude = np.asarray(amplitude, dtype=np.float32).reshape(self.input_height, self.input_width)
        self._phase = np.asarray(phase_radians, dtype=np.float32).reshape(self.input_height, self.input_width)


@dataclass
class LinearMockCamera:
    """Camera simulator: intensity = abs(H @ displayed_field)^2."""

    projector: MemoryProjector
    transmission_matrix: np.ndarray
    output_height: int = 128
    output_width: int = 128
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        self.roi = Roi(y=0, x=0, height=self.output_height, width=self.output_width)
        expected_shape = (self.output_height * self.output_width, self.projector.input_height * self.projector.input_width)
        if tuple(self.transmission_matrix.shape) != expected_shape:
            raise ValueError(f"transmission_matrix shape must be {expected_shape}, got {self.transmission_matrix.shape}")

    def capture_intensity(self) -> np.ndarray:
        field = self.transmission_matrix @ self.projector.field_vector
        intensity = np.abs(field) ** 2
        frame = intensity.reshape(self.output_height, self.output_width).astype(np.float32)
        if self.noise_std > 0.0:
            frame = frame + np.random.normal(0.0, self.noise_std, size=frame.shape).astype(np.float32)
        return np.clip(frame, 0.0, None)


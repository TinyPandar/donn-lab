from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


def checkpoint_model_state(raw_checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return the model state dict from either modern or legacy checkpoints."""
    state = raw_checkpoint.get("model_state_dict", raw_checkpoint.get("state_dict", raw_checkpoint))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a dict-like model state")
    return state


def find_phase_keys(state: dict[str, Any]) -> list[str]:
    """Find ParameterList phase tensors saved as phases.0, phases.1, ..."""
    phase_keys = []
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        stripped = key[7:] if key.startswith("module.") else key
        if re.fullmatch(r"phases\.\d+", stripped):
            phase_keys.append(key)
    return sorted(phase_keys, key=lambda key: int(key.split(".")[-1]))


def _phase_grid_from_tensor(tensor: torch.Tensor) -> tuple[int, int]:
    if tensor.ndim < 2:
        side = int(math.isqrt(int(tensor.numel())))
        if side * side != int(tensor.numel()):
            raise ValueError(f"Cannot infer phase grid from tensor shape {tuple(tensor.shape)}")
        return side, side
    return int(tensor.shape[-2]), int(tensor.shape[-1])


def _reshape_column_delta(column_phase_delta: np.ndarray, phase: torch.Tensor) -> torch.Tensor:
    if phase.ndim < 2:
        if int(phase.numel()) != int(column_phase_delta.shape[0]):
            raise ValueError(
                f"Phase tensor shape {tuple(phase.shape)} has {phase.numel()} values, "
                f"but correction has {column_phase_delta.shape[0]} entries"
            )
        return torch.as_tensor(column_phase_delta, dtype=phase.dtype, device=phase.device).reshape_as(phase)

    height, width = _phase_grid_from_tensor(phase)
    if height * width != int(column_phase_delta.shape[0]):
        raise ValueError(
            f"Phase tensor shape {tuple(phase.shape)} has {height * width} pixels, "
            f"but correction has {column_phase_delta.shape[0]} entries"
        )

    correction = torch.as_tensor(column_phase_delta.reshape(height, width), dtype=phase.dtype, device=phase.device)
    while correction.ndim < phase.ndim:
        correction = correction.unsqueeze(0)
    return correction


def apply_phase_correction_to_state(
    state: dict[str, Any],
    column_phase_delta: np.ndarray,
    *,
    phase_sign: str = "subtract",
    wrap: bool = True,
) -> list[dict[str, Any]]:
    """Apply a column phase correction to every phase mask in a checkpoint.

    `phase_sign="subtract"` corresponds to compensating H1 ~= H0 * exp(i*delta)
    with phase <- phase - delta.
    """
    if phase_sign not in {"subtract", "add"}:
        raise ValueError("phase_sign must be 'subtract' or 'add'")

    phase_keys = find_phase_keys(state)
    if not phase_keys:
        raise ValueError("No phase tensors matching phases.<idx> were found in the checkpoint")

    applied: list[dict[str, Any]] = []
    for key in phase_keys:
        phase = state[key]
        if not torch.is_floating_point(phase):
            raise ValueError(f"{key} must be a floating point phase tensor, got {phase.dtype}")

        correction = _reshape_column_delta(column_phase_delta, phase)
        corrected = phase - correction if phase_sign == "subtract" else phase + correction
        if wrap:
            corrected = torch.remainder(corrected, 2.0 * math.pi)
        state[key] = corrected
        applied.append(
            {
                "key": key,
                "shape": list(phase.shape),
                "sign": phase_sign,
                "wrapped_to_0_2pi": bool(wrap),
            }
        )
    return applied


def write_column_phase_csv(column_phase_delta: np.ndarray, column_corr: np.ndarray, path: Path) -> None:
    """Write per-input-mode phase correction diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["column", "phase_delta_rad", "phase_corr"])
        writer.writeheader()
        for idx, (phase, corr) in enumerate(zip(column_phase_delta, column_corr, strict=True)):
            writer.writerow(
                {
                    "column": idx,
                    "phase_delta_rad": float(phase),
                    "phase_corr": float(corr),
                }
            )


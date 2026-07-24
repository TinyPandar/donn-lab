from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


ShapeLike = str | Sequence[int] | None


def parse_shape(shape: ShapeLike) -> tuple[int, int] | None:
    """Parse a matrix shape such as "16384,16384" or "16384x16384"."""
    if shape is None:
        return None
    if isinstance(shape, str):
        parts = [p for p in re.split(r"[xX,\s*]+", shape.strip()) if p]
        if len(parts) != 2:
            raise ValueError(f"shape must have two dimensions, got {shape!r}")
        return int(parts[0]), int(parts[1])
    if len(shape) != 2:
        raise ValueError(f"shape must have two dimensions, got {shape!r}")
    return int(shape[0]), int(shape[1])


def dtype_from_name(dtype_name: str) -> np.dtype:
    """Return the NumPy dtype used for raw matrix files."""
    choices = {
        "complex64": np.complex64,
        "complex128": np.complex128,
        "float32": np.float32,
        "float64": np.float64,
    }
    name = str(dtype_name).lower()
    if name not in choices:
        raise ValueError(f"Unsupported TM dtype {dtype_name!r}; expected one of {sorted(choices)}")
    return np.dtype(choices[name])


def load_tmatrix_numpy(
    path: str | Path,
    *,
    shape: ShapeLike = None,
    dtype: str = "complex64",
    layout: str = "out_in",
    mmap: bool = True,
) -> np.ndarray:
    """Load a measured transmission matrix as a NumPy array.

    Supported input formats:
    - `.npy`
    - `.npz`, using key `H` when present, otherwise the first array
    - raw binary/memmap files when `shape` is provided

    The returned matrix is always interpreted as `[N_out, N_in]`: rows are
    camera pixels, columns are input SLM/DMD modes.
    """
    tm_path = Path(path)
    if not tm_path.is_file():
        raise FileNotFoundError(f"TM file not found: {tm_path}")

    parsed_shape = parse_shape(shape)
    try:
        loaded = np.load(tm_path, mmap_mode=("r" if mmap else None), allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            key = "H" if "H" in loaded.files else loaded.files[0]
            arr = loaded[key]
        else:
            arr = loaded
    except Exception:
        if parsed_shape is None:
            raise ValueError(f"{tm_path} is not .npy/.npz; provide shape for raw memmap input")
        arr = np.memmap(tm_path, dtype=dtype_from_name(dtype), mode="r", shape=parsed_shape)

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 2 and not np.iscomplexobj(arr):
        arr = arr[..., 0] + 1j * arr[..., 1]
    if arr.ndim != 2:
        raise ValueError(f"TM must be 2D [N_out,N_in], got shape {arr.shape}")
    if parsed_shape is not None and tuple(arr.shape) != parsed_shape:
        raise ValueError(f"Shape mismatch for {tm_path}: got {arr.shape}, expected {parsed_shape}")

    layout_key = str(layout).lower()
    if layout_key in {"out_in", "nout_nin", "rows_out"}:
        return arr
    if layout_key in {"in_out", "nin_nout", "cols_out"}:
        return arr.T
    raise ValueError("layout must be 'out_in' or 'in_out'")


def load_tmatrix_torch(
    path: str | Path,
    *,
    shape: ShapeLike = None,
    dtype: str = "complex64",
    layout: str = "out_in",
    mmap: bool = True,
) -> torch.Tensor:
    """Load a measured matrix as a complex64 torch tensor.

    The matrix can be large. This function avoids an extra NumPy copy before
    handing the array to PyTorch; the final `.contiguous()` may still copy when
    the array is a transpose or a read-only memmap.
    """
    arr = load_tmatrix_numpy(path, shape=shape, dtype=dtype, layout=layout, mmap=mmap)
    # Measured H is a fixed buffer. Avoid copying a 2 GiB read-only memmap only
    # to satisfy PyTorch's writability warning.
    with warnings.catch_warnings():
        if not arr.flags.writeable:
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        tensor = torch.as_tensor(arr)
    if not torch.is_complex(tensor):
        tensor = torch.complex(tensor.float(), torch.zeros_like(tensor, dtype=torch.float32))
    return tensor.to(torch.complex64).contiguous()


def write_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a list of homogeneous dictionaries as a CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

from __future__ import annotations

import os
import sys
from pathlib import Path


# The CUDA 12.4 NVRTC package in this Windows conda environment installs its
# builtins DLL under ``<env>/bin`` rather than ``<env>/Library/bin``. Explicit
# Python launches do not add that directory automatically, so complex CUDA
# autograd kernels otherwise fail only when the first backward pass is compiled.
_DLL_DIRECTORY_HANDLES = []
_cuda_dll_dir = Path(sys.prefix) / "bin"
if os.name == "nt" and _cuda_dll_dir.is_dir():
    os.environ["PATH"] = str(_cuda_dll_dir) + os.pathsep + os.environ.get("PATH", "")
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if callable(add_dll_directory):
        _DLL_DIRECTORY_HANDLES.append(add_dll_directory(str(_cuda_dll_dir)))


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import main


def _with_measured_tm_model(argv: list[str]) -> list[str]:
    has_model = any(arg == "--model" or arg.startswith("--model=") for arg in argv)
    if has_model:
        return argv
    return ["--model", "measured_tm_scatter", *argv]


if __name__ == "__main__":
    main(_with_measured_tm_model(sys.argv[1:]))

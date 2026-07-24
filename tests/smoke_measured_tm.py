from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.schema import ExperimentConfig
from donn_lab.hardware import LinearMockCamera, MemoryProjector
from models.factory import DistContext, create_registered_model, register_builtin_models


def build_config(h_path: Path) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model = "measured_tm_scatter"
    cfg.model_cfg.name = "measured_tm_scatter"
    cfg.model_cfg.tmatrix_path = str(h_path)
    cfg.model_cfg.num_layers = 2
    cfg.model_cfg.phase_init = "zeros"
    cfg.data.h_in = 4
    cfg.data.w_in = 4
    cfg.data.h_out = 4
    cfg.data.w_out = 4
    cfg.data.input_mode = "gray"
    return cfg


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        rng = np.random.default_rng(7)
        h0 = (rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))).astype(np.complex64)
        column_delta = rng.uniform(-0.5, 0.5, size=16).astype(np.float32)
        h1 = h0 * np.exp(1j * column_delta)[None, :]
        np.save(tmp / "H0.npy", h0)
        np.save(tmp / "H1.npy", h1.astype(np.complex64))

        register_builtin_models()
        cfg = build_config(tmp / "H0.npy")
        model = create_registered_model(
            cfg.model,
            cfg,
            torch.device("cpu"),
            DistContext(False, 0, 1),
        )
        output = model(torch.rand(2, 1, 4, 4))
        assert tuple(output.shape) == (2, 4, 4)
        assert sorted(model.state_dict().keys()) == ["phases.0", "phases.1"]

        projector = MemoryProjector(input_height=4, input_width=4)
        camera = LinearMockCamera(projector=projector, transmission_matrix=h0, output_height=4, output_width=4)
        projector.display_phase(np.zeros((4, 4), dtype=np.float32))
        assert camera.capture_intensity().shape == (4, 4)

        subprocess.run(
            [
                sys.executable,
                "scripts/compare_tm_drift.py",
                "--h0",
                str(tmp / "H0.npy"),
                "--h1",
                str(tmp / "H1.npy"),
                "--out-dir",
                str(tmp / "compare"),
                "--chunk-rows",
                "4",
            ],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )

        phase = torch.full((1, 1, 4, 4), 1.25, dtype=torch.float32)
        torch.save({"model_state_dict": {"phases.0": phase.clone()}}, tmp / "epoch_1.pth")
        subprocess.run(
            [
                sys.executable,
                "scripts/apply_tm_correction.py",
                "--checkpoint",
                str(tmp / "epoch_1.pth"),
                "--h0",
                str(tmp / "H0.npy"),
                "--h1",
                str(tmp / "H1.npy"),
                "--out-dir",
                str(tmp / "corr"),
                "--chunk-rows",
                "4",
                "--no-row-align",
            ],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        corrected = torch.load(tmp / "corr" / "epoch_1_tm_corrected.pth", map_location="cpu")["model_state_dict"]["phases.0"]
        expected = torch.remainder(phase - torch.as_tensor(column_delta.reshape(1, 1, 4, 4)), 2.0 * math.pi)
        assert float((corrected - expected).abs().max()) < 1e-5

    print("smoke_measured_tm: ok")


if __name__ == "__main__":
    main()

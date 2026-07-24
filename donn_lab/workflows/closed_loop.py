from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkflowStep:
    """One shell command in the measured-TM closed loop."""

    name: str
    command: list[str]
    note: str


@dataclass(frozen=True)
class ClosedLoopPaths:
    """File paths shared by measurement, training, and evaluation."""

    config: Path
    h0: Path
    h1: Path
    run_dir: Path
    report_dir: Path
    checkpoint: Path | None = None


def _latest_checkpoint_arg(paths: ClosedLoopPaths) -> list[str]:
    if paths.checkpoint is None:
        return []
    return ["--checkpoint", str(paths.checkpoint)]


def build_closed_loop_steps(
    paths: ClosedLoopPaths,
    *,
    python_bin: str = "/home/limingfei/miniforge3/envs/speckle/bin/python",
    num_layers: int = 5,
) -> list[WorkflowStep]:
    """Build the stage-0 command sequence without executing it.

    Hardware measurement is still external at this stage. The commands below
    start after H0 exists and continue through H1 testing and phase correction.
    """
    h0_metrics = paths.report_dir / "eval_h0.csv"
    h1_metrics = paths.report_dir / "eval_h1.csv"
    corrected_dir = paths.report_dir / "phase_correction"
    corrected_checkpoint = corrected_dir / "corrected_from_h1.pth"

    return [
        WorkflowStep(
            name="train_from_h0",
            command=[
                python_bin,
                "scripts/train_measured_tm.py",
                "--config",
                str(paths.config),
                "--tmatrix_path",
                str(paths.h0),
                "--num_layers",
                str(num_layers),
            ],
            note="Train phase masks using the first measured transmission matrix H0.",
        ),
        WorkflowStep(
            name="evaluate_h0",
            command=[
                python_bin,
                "scripts/evaluate_measured_tm.py",
                "--run-dir",
                str(paths.run_dir),
                "--tmatrix-path",
                str(paths.h0),
                "--out",
                str(h0_metrics),
                *_latest_checkpoint_arg(paths),
            ],
            note="Evaluate the trained checkpoint on the same matrix used for training.",
        ),
        WorkflowStep(
            name="evaluate_h1",
            command=[
                python_bin,
                "scripts/evaluate_measured_tm.py",
                "--run-dir",
                str(paths.run_dir),
                "--tmatrix-path",
                str(paths.h1),
                "--out",
                str(h1_metrics),
                *_latest_checkpoint_arg(paths),
            ],
            note="Evaluate the same checkpoint on the remeasured matrix H1.",
        ),
        WorkflowStep(
            name="compare_h0_h1",
            command=[
                python_bin,
                "scripts/compare_tm_drift.py",
                "--h0",
                str(paths.h0),
                "--h1",
                str(paths.h1),
                "--out-dir",
                str(paths.report_dir / "tm_drift"),
            ],
            note="Summarize row and column drift between H0 and H1.",
        ),
        WorkflowStep(
            name="apply_phase_correction",
            command=[
                python_bin,
                "scripts/apply_tm_correction.py",
                "--checkpoint",
                str(paths.checkpoint or paths.run_dir / "epoch_300.pth"),
                "--h0",
                str(paths.h0),
                "--h1",
                str(paths.h1),
                "--out-dir",
                str(corrected_dir),
                "--out-checkpoint",
                str(corrected_checkpoint),
            ],
            note="Estimate input-column phase drift and write a corrected checkpoint.",
        ),
        WorkflowStep(
            name="evaluate_corrected_h1",
            command=[
                python_bin,
                "scripts/evaluate_measured_tm.py",
                "--run-dir",
                str(paths.run_dir),
                "--checkpoint",
                str(corrected_checkpoint),
                "--tmatrix-path",
                str(paths.h1),
                "--out",
                str(paths.report_dir / "eval_h1_corrected.csv"),
            ],
            note="Check whether the phase correction recovers H1 performance.",
        ),
    ]


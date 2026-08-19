from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG = REPO_ROOT / "configs/vehicle_dark_intersection_measured_tm_scatter_local.yaml"
DEFAULT_TMATRIX = REPO_ROOT / "tm.npy"
DEFAULT_TRAIN_PYTHON_CANDIDATE = Path(
    r"C:\Users\smart\miniconda3\envs\donn-lab\python.exe"
)
DEFAULT_HARDWARE_PYTHON = Path(
    r"C:\Users\smart\miniconda3\envs\py38\python.exe"
)


class WorkflowError(RuntimeError):
    pass


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_tmatrix(
    path: Path,
    *,
    full_finite_scan: bool,
    include_hash: bool,
) -> Dict[str, Any]:
    if not path.is_file():
        raise WorkflowError("TM file does not exist: {0}".format(path))
    try:
        matrix = np.load(str(path), mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise WorkflowError("Cannot open TM file {0}: {1}".format(path, exc)) from exc

    expected_shape = (128 * 128, 128 * 128)
    if tuple(matrix.shape) != expected_shape:
        raise WorkflowError(
            "TM shape is {0}, expected {1}".format(tuple(matrix.shape), expected_shape)
        )
    if matrix.dtype != np.dtype(np.complex64):
        raise WorkflowError(
            "TM dtype is {0}, expected complex64".format(matrix.dtype)
        )

    finite_count = int(matrix.size)
    if full_finite_scan:
        rows_per_chunk = 256
        finite_count = 0
        for start in range(0, matrix.shape[0], rows_per_chunk):
            stop = min(start + rows_per_chunk, matrix.shape[0])
            block = np.asarray(matrix[start:stop])
            block_finite = np.isfinite(block)
            finite_count += int(np.count_nonzero(block_finite))
            if not bool(np.all(block_finite)):
                bad = int(block.size - np.count_nonzero(block_finite))
                raise WorkflowError(
                    "TM contains {0} non-finite value(s) in rows {1}:{2}".format(
                        bad, start, stop
                    )
                )

    stat = path.stat()
    identity: Dict[str, Any] = {
        "path": str(path),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "finite_values": finite_count,
    }
    if include_hash:
        identity["sha256"] = _sha256(path)
    return identity


def _same_tm_identity(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    keys = ("path", "shape", "dtype", "size_bytes", "mtime_ns")
    if any(before.get(key) != after.get(key) for key in keys):
        return False
    if "sha256" in before or "sha256" in after:
        return before.get("sha256") == after.get("sha256")
    return True


def _format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def run_logged(
    command: Sequence[str],
    *,
    log_path: Path,
    dry_run: bool,
) -> None:
    rendered = _format_command(command)
    print("\n>>> {0}".format(rendered), flush=True)
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        log_handle.write(">>> {0}\n".format(rendered))
        log_handle.flush()
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise

    if return_code != 0:
        raise WorkflowError(
            "Command failed with exit code {0}; log: {1}".format(
                return_code, log_path
            )
        )


def _latest_checkpoint(checkpoint_root: Path, expected_epoch: int) -> Path:
    expected_name = "epoch_{0:03d}.pth".format(expected_epoch)
    exact = sorted(checkpoint_root.rglob(expected_name))
    if len(exact) == 1:
        return exact[0].resolve()
    if len(exact) > 1:
        raise WorkflowError(
            "Multiple final checkpoints were produced under {0}: {1}".format(
                checkpoint_root, [str(path) for path in exact]
            )
        )
    available = sorted(checkpoint_root.rglob("epoch_*.pth"))
    raise WorkflowError(
        "Training completed but {0} was not found under {1}; available={2}".format(
            expected_name,
            checkpoint_root,
            [str(path) for path in available],
        )
    )


def _read_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise WorkflowError("Expected summary was not written: {0}".format(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WorkflowError("Summary is not a JSON object: {0}".format(path))
    return payload


def _default_train_python() -> Path:
    if DEFAULT_TRAIN_PYTHON_CANDIDATE.is_file():
        return DEFAULT_TRAIN_PYTHON_CANDIDATE
    return Path(sys.executable)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train five measured-TM optical layers on the current tm.npy, "
            "validate them in simulation, then optionally run one guarded V4 hardware test."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tmatrix", type=Path, default=DEFAULT_TMATRIX)
    parser.add_argument("--train-python", type=Path, default=_default_train_python())
    parser.add_argument("--hardware-python", type=Path, default=DEFAULT_HARDWARE_PYTHON)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--tag", default="new-tm-linked")

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=300)
    training.add_argument("--train-batch-size", type=int, default=None)
    training.add_argument("--max-train-batches", type=int, default=None)
    training.add_argument("--max-test-batches", type=int, default=None)
    training.add_argument("--train-device", default="cuda:0")
    training.add_argument("--skip-training", action="store_true")
    training.add_argument("--checkpoint", type=Path, default=None)

    simulation = parser.add_argument_group("simulation gate")
    simulation.add_argument(
        "--inference-split",
        choices=("train", "val", "test"),
        default="val",
        help="Dataset split used by both simulation and the final hardware inference.",
    )
    simulation.add_argument("--sim-samples", type=int, default=32)
    simulation.add_argument("--sim-batch-size", type=int, default=16)
    simulation.add_argument("--max-sim-mean-distance", type=float, default=5.0)
    simulation.add_argument("--skip-simulation", action="store_true")
    simulation.add_argument("--allow-poor-simulation", action="store_true")

    hardware = parser.add_argument_group("guarded hardware test")
    hardware.add_argument("--arm-hardware", action="store_true")
    hardware.add_argument(
        "--yes",
        action="store_true",
        help="Skip the final interactive ARM confirmation (requires --arm-hardware).",
    )
    hardware.add_argument("--skip-hardware", action="store_true")
    hardware.add_argument("--hardware-samples", type=int, default=1)
    hardware.add_argument("--dark-frames", type=int, default=16)
    hardware.add_argument("--capture-repeats", type=int, default=1)
    hardware.add_argument("--min-dynamic-range", type=float, default=5.0)
    hardware.add_argument("--dmd-device", default=None)

    parser.add_argument("--skip-tm-hash", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.train_batch_size is not None and args.train_batch_size <= 0:
        parser.error("--train-batch-size must be positive")
    for name in ("max_train_batches", "max_test_batches"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error("--{0} must be non-negative".format(name.replace("_", "-")))
    if args.sim_samples <= 0 or args.sim_batch_size <= 0:
        parser.error("simulation sample and batch counts must be positive")
    if args.max_sim_mean_distance < 0:
        parser.error("--max-sim-mean-distance must be non-negative")
    if args.hardware_samples <= 0:
        parser.error("--hardware-samples must be positive")
    if args.dark_frames < 0 or args.capture_repeats <= 0:
        parser.error("--dark-frames must be non-negative and --capture-repeats positive")
    if args.min_dynamic_range < 0:
        parser.error("--min-dynamic-range must be non-negative")
    if args.skip_training and args.checkpoint is None:
        parser.error("--skip-training requires --checkpoint")
    if not args.skip_training and args.checkpoint is not None:
        parser.error("--checkpoint is only accepted together with --skip-training")
    if args.yes and not args.arm_hardware:
        parser.error("--yes is only meaningful together with --arm-hardware")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.tag):
        parser.error("--tag may contain only letters, digits, dot, underscore, and hyphen")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = _resolved(args.config)
    tmatrix = _resolved(args.tmatrix)
    train_python = _resolved(args.train_python)
    hardware_python = _resolved(args.hardware_python)

    required_files = {
        "config": config,
        "tmatrix": tmatrix,
        "train_python": train_python,
        "training_script": REPO_ROOT / "scripts/train_measured_tm.py",
        "experiment_script": REPO_ROOT / "scripts/run_optical_vehicle_experiment.py",
    }
    if not args.skip_hardware:
        required_files["hardware_python"] = hardware_python
    missing = ["{0}={1}".format(name, path) for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise WorkflowError("Required file(s) missing: " + "; ".join(missing))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    workflow_root = (
        _resolved(args.run_root)
        if args.run_root is not None
        else REPO_ROOT / "runs/linked_optical_vehicle" / (stamp + "-" + args.tag)
    )
    if workflow_root.exists() and not args.dry_run:
        raise WorkflowError("Workflow directory already exists: {0}".format(workflow_root))

    manifest_path = workflow_root / "workflow.json"
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "created_at_utc": _utc_now(),
        "repo_root": str(REPO_ROOT),
        "workflow_root": str(workflow_root),
        "config": str(config),
        "tmatrix": str(tmatrix),
        "train_python": str(train_python),
        "hardware_python": str(hardware_python),
        "commands": [],
    }

    def record(status: str, **updates: Any) -> None:
        manifest.update(updates)
        manifest["status"] = status
        manifest["updated_at_utc"] = _utc_now()
        if not args.dry_run:
            _atomic_write_json(manifest_path, manifest)

    try:
        print("Validating TM: {0}".format(tmatrix), flush=True)
        tm_identity = validate_tmatrix(
            tmatrix,
            full_finite_scan=not args.dry_run,
            include_hash=(not args.skip_tm_hash and not args.dry_run),
        )
        record("tm_validated", tm_identity_before=tm_identity)
        print(json.dumps(tm_identity, indent=2, ensure_ascii=False), flush=True)

        if args.skip_training:
            checkpoint = _resolved(args.checkpoint)
            if not checkpoint.is_file():
                raise WorkflowError("Checkpoint does not exist: {0}".format(checkpoint))
            record("checkpoint_selected", checkpoint=str(checkpoint), training_skipped=True)
        else:
            checkpoint_root = workflow_root / "training_checkpoints"
            train_log_root = workflow_root / "training_tensorboard"
            train_command: List[str] = [
                str(train_python),
                str(REPO_ROOT / "scripts/train_measured_tm.py"),
                "--config",
                str(config),
                "--tmatrix_path",
                str(tmatrix),
                "--epochs",
                str(args.epochs),
                "--device",
                str(args.train_device),
                "--ckpt_dir",
                str(checkpoint_root),
                "--log_dir",
                str(train_log_root),
                "--comment",
                args.tag,
            ]
            if args.train_batch_size is not None:
                train_command.extend(("--batch_size", str(args.train_batch_size)))
            if args.max_train_batches is not None:
                train_command.extend(("--max_train_batches", str(args.max_train_batches)))
            if args.max_test_batches is not None:
                train_command.extend(("--max_test_batches", str(args.max_test_batches)))
            manifest["commands"].append(_format_command(train_command))
            record("training")
            run_logged(
                train_command,
                log_path=workflow_root / "logs/train.log",
                dry_run=args.dry_run,
            )
            if args.dry_run:
                checkpoint = Path("<new-checkpoint>")
            else:
                checkpoint = _latest_checkpoint(checkpoint_root, args.epochs)
                from donn_lab.experiment.optical_inference import load_checkpoint_phases

                loaded = load_checkpoint_phases(
                    checkpoint,
                    expected_num_layers=5,
                    expected_hw=(128, 128),
                    strict=True,
                )
                record(
                    "trained",
                    checkpoint=str(checkpoint),
                    checkpoint_epoch=loaded.epoch,
                    checkpoint_global_step=loaded.global_step,
                )

        if not args.dry_run:
            tm_identity_after = validate_tmatrix(
                tmatrix,
                full_finite_scan=False,
                include_hash=not args.skip_tm_hash,
            )
            if not _same_tm_identity(tm_identity, tm_identity_after):
                raise WorkflowError(
                    "tm.npy changed after workflow start; refusing to mix training/test assets"
                )
            record("tm_reverified", tm_identity_after=tm_identity_after)

        simulation_summary: Optional[Dict[str, Any]] = None
        if not args.skip_simulation:
            simulation_output = workflow_root / "simulation"
            simulation_command = [
                str(train_python),
                str(REPO_ROOT / "scripts/run_optical_vehicle_experiment.py"),
                "--mode",
                "simulate",
                "--checkpoint",
                str(checkpoint),
                "--tmatrix",
                str(tmatrix),
                "--split",
                args.inference_split,
                "--max-samples",
                str(args.sim_samples),
                "--batch-size",
                str(min(args.sim_batch_size, args.sim_samples)),
                "--save-layers",
                "none",
                "--output-dir",
                str(simulation_output),
            ]
            manifest["commands"].append(_format_command(simulation_command))
            record("simulating")
            run_logged(
                simulation_command,
                log_path=workflow_root / "logs/simulation.log",
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                simulation_summary = _read_summary(simulation_output / "summary.json")
                if simulation_summary.get("status") != "complete":
                    raise WorkflowError(
                        "Simulation did not complete: {0}".format(simulation_summary)
                    )
                mean_distance = float(simulation_summary["mean_pixel_distance"])
                if (
                    not args.allow_poor_simulation
                    and mean_distance > args.max_sim_mean_distance
                ):
                    raise WorkflowError(
                        "Simulation mean distance {:.4f}px exceeds gate {:.4f}px; "
                        "hardware remains unarmed".format(
                            mean_distance, args.max_sim_mean_distance
                        )
                    )
                record("simulation_passed", simulation_summary=simulation_summary)

        if args.skip_hardware:
            record("complete", hardware_skipped=True)
            print("Workflow completed without hardware stage.")
            return 0

        preflight_command = [
            str(hardware_python),
            str(REPO_ROOT / "scripts/run_optical_vehicle_experiment.py"),
            "--mode",
            "hardware",
            "--checkpoint",
            str(checkpoint),
            "--preflight-only",
        ]
        if args.dmd_device:
            preflight_command.extend(("--dmd-device", args.dmd_device))
        manifest["commands"].append(_format_command(preflight_command))
        record("hardware_preflight")
        run_logged(
            preflight_command,
            log_path=workflow_root / "logs/hardware_preflight.log",
            dry_run=args.dry_run,
        )

        if args.dry_run:
            print("Dry-run complete; no training, simulation, or hardware command was executed.")
            return 0

        if not args.arm_hardware:
            continuation = [
                str(train_python),
                str(Path(__file__).resolve()),
                "--skip-training",
                "--checkpoint",
                str(checkpoint),
                "--tmatrix",
                str(tmatrix),
                "--hardware-python",
                str(hardware_python),
                "--skip-simulation",
                "--inference-split",
                args.inference_split,
                "--arm-hardware",
                "--hardware-samples",
                str(args.hardware_samples),
                "--dark-frames",
                str(args.dark_frames),
                "--capture-repeats",
                str(args.capture_repeats),
                "--min-dynamic-range",
                str(args.min_dynamic_range),
            ]
            if args.dmd_device:
                continuation.extend(("--dmd-device", args.dmd_device))
            record(
                "awaiting_hardware_arm",
                hardware_preflight_passed=True,
                continuation_command=_format_command(continuation),
            )
            print(
                "\nAll requested non-hardware stages passed or were explicitly skipped. "
                "Hardware was NOT armed."
            )
            print("To run the guarded one-sample test:\n{0}".format(_format_command(continuation)))
            return 0

        if not args.yes:
            response = input(
                "\n确认激光、光阑、散射介质、相机和触发线均已就绪。"
                "输入 ARM 才会启动 DMD 真机测试: "
            ).strip()
            if response != "ARM":
                record("hardware_cancelled", hardware_preflight_passed=True)
                print("Hardware test cancelled; no device was armed by this stage.")
                return 2

        hardware_output = workflow_root / "hardware"
        hardware_command = [
            str(hardware_python),
            str(REPO_ROOT / "scripts/run_optical_vehicle_experiment.py"),
            "--mode",
            "hardware",
            "--arm-hardware",
            "--checkpoint",
            str(checkpoint),
            "--split",
            args.inference_split,
            "--max-samples",
            str(args.hardware_samples),
            "--dark-frames",
            str(args.dark_frames),
            "--capture-repeats",
            str(args.capture_repeats),
            "--min-dynamic-range",
            str(args.min_dynamic_range),
            "--save-layers",
            "all",
            "--output-dir",
            str(hardware_output),
        ]
        if args.dmd_device:
            hardware_command.extend(("--dmd-device", args.dmd_device))
        manifest["commands"].append(_format_command(hardware_command))
        record("hardware_running", hardware_armed=True)
        run_logged(
            hardware_command,
            log_path=workflow_root / "logs/hardware.log",
            dry_run=False,
        )
        hardware_summary = _read_summary(hardware_output / "summary.json")
        if hardware_summary.get("status") != "complete":
            raise WorkflowError(
                "Hardware experiment did not complete: {0}".format(hardware_summary)
            )
        record(
            "complete",
            hardware_preflight_passed=True,
            hardware_summary=hardware_summary,
        )
        print("\nLinked training/simulation/hardware workflow completed.")
        print("Checkpoint: {0}".format(checkpoint))
        print("Artifacts: {0}".format(workflow_root))
        return 0
    except KeyboardInterrupt:
        record("interrupted", error="KeyboardInterrupt")
        print("Workflow interrupted; hardware subprocess was terminated.", file=sys.stderr)
        return 130
    except Exception as exc:
        record(
            "failed",
            error="{0}: {1}".format(type(exc).__name__, exc),
        )
        print("Workflow failed: {0}: {1}".format(type(exc).__name__, exc), file=sys.stderr)
        if not args.dry_run:
            print("Workflow record: {0}".format(manifest_path), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

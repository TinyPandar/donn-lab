from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def quick_file_identity(path: Path, include_hash: bool = False) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    result = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        result["sha256"] = sha256_file(resolved)
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(payload), stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def intensity_to_uint8(frame: np.ndarray) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError("Cannot render an intensity frame containing NaN or infinity")
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.rint((values - minimum) * (255.0 / (maximum - minimum))).astype(np.uint8)


class RunArtifactWriter:
    """Crash-safe run/sample writer with strict resume fingerprint checks."""

    def __init__(
        self,
        output_dir: Path,
        run_metadata: Mapping[str, Any],
        *,
        resume: bool = False,
    ) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.samples_dir = self.output_dir / "samples"
        self.manifest_path = self.output_dir / "manifest.csv"
        self.run_path = self.output_dir / "run.json"
        self.run_metadata = _jsonable(dict(run_metadata))
        self.resume = bool(resume)

    def prepare(self) -> None:
        if self.run_path.exists():
            if not self.resume:
                raise FileExistsError(
                    "Output run already exists; pass --resume or choose another directory: {0}".format(
                        self.output_dir
                    )
                )
            existing = json.loads(self.run_path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != self.run_metadata.get("fingerprint"):
                raise RuntimeError("Resume fingerprint differs from the existing run.json")
        else:
            if self.output_dir.exists() and any(self.output_dir.iterdir()):
                if self.resume:
                    raise RuntimeError("Cannot resume a non-empty directory without run.json")
                raise FileExistsError(
                    "Output directory is non-empty and has no run.json: {0}".format(
                        self.output_dir
                    )
                )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            metadata = dict(self.run_metadata)
            metadata.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
            _write_json_atomic(self.run_path, metadata)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        if self.resume:
            self._reconcile_completed_samples()

    def _reconcile_completed_samples(self) -> None:
        """Recover the manifest after a crash between sample rename and append."""

        manifest_ids = set()
        if self.manifest_path.is_file():
            try:
                with self.manifest_path.open("r", newline="", encoding="utf-8") as stream:
                    manifest_ids = {
                        str(row.get("sample_id"))
                        for row in csv.DictReader(stream)
                        if row.get("sample_id")
                    }
            except OSError:
                manifest_ids = set()
        for sample_dir in sorted(self.samples_dir.iterdir()):
            if not sample_dir.is_dir() or sample_dir.name.startswith("."):
                continue
            if sample_dir.name in manifest_ids:
                continue
            done_path = sample_dir / "DONE.json"
            prediction_path = sample_dir / "prediction.json"
            if not done_path.is_file() or not prediction_path.is_file():
                continue
            try:
                done = json.loads(done_path.read_text(encoding="utf-8"))
                prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                done.get("sample_id") != sample_dir.name
                or done.get("fingerprint") != self.run_metadata.get("fingerprint")
                or not isinstance(prediction, dict)
            ):
                continue
            self._append_manifest({"sample_id": sample_dir.name, **prediction})
            manifest_ids.add(sample_dir.name)

    def sample_done(self, sample_id: str) -> bool:
        sample_dir = self.samples_dir / sample_id
        done_path = sample_dir / "DONE.json"
        if not done_path.is_file():
            return False
        required = (
            sample_dir / "input_chw.npy",
            sample_dir / "final_intensity.npy",
            sample_dir / "prediction.json",
        )
        if not all(path.is_file() for path in required):
            return False
        try:
            done = json.loads(done_path.read_text(encoding="utf-8"))
            prediction = json.loads((sample_dir / "prediction.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            isinstance(prediction, dict)
            and done.get("sample_id") == sample_id
            and done.get("fingerprint") == self.run_metadata.get("fingerprint")
        )

    def update_run_metadata(self, updates: Mapping[str, Any]) -> None:
        """Merge runtime-discovered metadata without changing the resume fingerprint."""

        current = json.loads(self.run_path.read_text(encoding="utf-8"))
        for key, value in updates.items():
            if key == "fingerprint" and value != current.get("fingerprint"):
                raise ValueError("The run fingerprint is immutable")
            current[str(key)] = _jsonable(value)
        _write_json_atomic(self.run_path, current)

    def save_calibration_array(self, name: str, value: np.ndarray) -> Path:
        if not name or any(char in name for char in "\\/:"):
            raise ValueError("Calibration name must be a simple filename stem")
        calibration_dir = self.output_dir / "calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        path = calibration_dir / (name + ".npy")
        temporary = calibration_dir / (name + ".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, np.asarray(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        return path

    def write_sample(
        self,
        *,
        sample_id: str,
        input_chw: np.ndarray,
        final_intensity: np.ndarray,
        prediction: Mapping[str, Any],
        layer_frames: Optional[np.ndarray] = None,
        layer_raw_frames: Optional[np.ndarray] = None,
        layer_amplitudes: Optional[np.ndarray] = None,
        layer_fields: Optional[np.ndarray] = None,
    ) -> None:
        final_dir = self.samples_dir / sample_id
        if final_dir.exists():
            if self.sample_done(sample_id):
                return
            raise RuntimeError("Incomplete sample directory already exists: {0}".format(final_dir))

        temporary_dir = Path(
            tempfile.mkdtemp(prefix=".{0}.".format(sample_id), suffix=".tmp", dir=str(self.samples_dir))
        )
        try:
            np.save(str(temporary_dir / "input_chw.npy"), np.asarray(input_chw, dtype=np.float32))
            np.save(str(temporary_dir / "final_intensity.npy"), np.asarray(final_intensity, dtype=np.float32))
            if layer_frames is not None:
                np.savez_compressed(
                    str(temporary_dir / "layer_frames.npz"),
                    frames=np.asarray(layer_frames, dtype=np.float32),
                )
            if layer_raw_frames is not None:
                np.savez_compressed(
                    str(temporary_dir / "layer_raw_frames.npz"),
                    frames=np.asarray(layer_raw_frames, dtype=np.float32),
                )
            if layer_amplitudes is not None:
                np.savez_compressed(
                    str(temporary_dir / "layer_amplitudes.npz"),
                    amplitudes=np.asarray(layer_amplitudes, dtype=np.float32),
                )
            if layer_fields is not None:
                np.savez_compressed(
                    str(temporary_dir / "layer_fields.npz"),
                    fields=np.asarray(layer_fields, dtype=np.complex64),
                )

            preview = intensity_to_uint8(final_intensity)
            try:
                from PIL import Image

                Image.fromarray(preview, mode="L").save(str(temporary_dir / "final_preview.png"))
            except ImportError:
                pass
            _write_json_atomic(temporary_dir / "prediction.json", prediction)
            _write_json_atomic(
                temporary_dir / "DONE.json",
                {
                    "sample_id": sample_id,
                    "fingerprint": self.run_metadata.get("fingerprint"),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            os.replace(str(temporary_dir), str(final_dir))
        except Exception:
            shutil.rmtree(str(temporary_dir), ignore_errors=True)
            raise

        manifest_row = {"sample_id": sample_id, **_jsonable(dict(prediction))}
        self._append_manifest(manifest_row)

    def _append_manifest(self, row: Mapping[str, Any]) -> None:
        normalized = {str(key): value for key, value in row.items() if not isinstance(value, (dict, list, tuple))}
        existing_rows = []
        existing_fields = []
        if self.manifest_path.is_file():
            with self.manifest_path.open("r", newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                existing_fields = list(reader.fieldnames or [])
                existing_rows = list(reader)
        fields = existing_fields + [key for key in normalized if key not in existing_fields]
        row_by_id = {item.get("sample_id"): item for item in existing_rows}
        row_by_id[str(normalized.get("sample_id"))] = normalized
        temporary = self.manifest_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for item in row_by_id.values():
                writer.writerow(item)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(self.manifest_path))

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        _write_json_atomic(self.output_dir / "summary.json", summary)


__all__ = [
    "RunArtifactWriter",
    "intensity_to_uint8",
    "quick_file_identity",
    "sha256_file",
]

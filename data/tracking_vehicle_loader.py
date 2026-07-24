import csv
import os
import random
from collections import OrderedDict
from typing import Iterator, List, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - fallback is exercised only when cv2 is unavailable
    cv2 = None
    from PIL import Image


class TrackingVehicleDataset:
    """
    Iterable over the synthetic tracking dataset producing (frames_bgr, centers_seq).

    Data Format:
    - Labels are stored in root/keyframes_manifest.csv
    - Required columns: sequence_id, image, center_x, center_y, frame_idx
    - Optional split column: train / val / test

    Expected directory layout:
    - root/sequences/{split}/{sequence_id}/frame_*.png
    - root/keyframes_manifest.csv

    Output format:
    - frames_bgr: np.ndarray, shape (T, H, W, 3), uint8
    - centers_seq: np.ndarray, shape (T, 2), float32
    """

    def __init__(
        self,
        root: str,
        split: str = "Train",
        manifest_file: str = "keyframes_manifest.csv",
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.root = root
        self.split = split
        self.manifest_path = os.path.join(self.root, manifest_file)
        self.sequences_root = os.path.join(self.root, "sequences")

        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        if not os.path.isdir(self.sequences_root):
            raise FileNotFoundError(f"Sequences dir not found: {self.sequences_root}")

        requested_split = self._normalize_requested_split(split)
        self.samples = self._load_sequences(requested_split=requested_split, train_ratio=train_ratio, seed=seed)
        if not self.samples:
            raise RuntimeError(f"No tracking sequences found for split={split!r} in {self.root}")

    @staticmethod
    def _normalize_requested_split(split: str) -> str:
        split_lower = split.strip().lower()
        if split_lower == "train":
            return "train"
        if split_lower in {"test", "val", "valid", "validation"}:
            return "val"
        return split_lower

    @staticmethod
    def _normalize_row_split(split_value: str) -> str:
        split_lower = split_value.strip().lower()
        if split_lower in {"valid", "validation", "test"}:
            return "val"
        return split_lower

    @staticmethod
    def _read_image_bgr(img_path: str) -> np.ndarray | None:
        if cv2 is not None:
            return cv2.imread(img_path, cv2.IMREAD_COLOR)

        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
        return arr[:, :, ::-1].copy()

    def _load_sequences(
        self,
        requested_split: str,
        train_ratio: float,
        seed: int,
    ) -> List[Tuple[str, List[str], np.ndarray]]:
        grouped: "OrderedDict[str, List[Tuple[int, str, float, float]]]" = OrderedDict()
        split_by_sequence: dict[str, str] = {}

        with open(self.manifest_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_columns = {"sequence_id", "image", "center_x", "center_y", "frame_idx"}
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise RuntimeError(
                    f"Manifest file must contain columns {sorted(required_columns)}: {self.manifest_path}"
                )

            for row in reader:
                sequence_id = (row.get("sequence_id") or "").strip()
                rel_path = (row.get("image") or "").strip()
                if not sequence_id or not rel_path:
                    continue

                img_path = os.path.normpath(os.path.join(self.root, rel_path.replace("/", os.sep)))
                if not os.path.isfile(img_path):
                    continue

                try:
                    frame_idx = int(row["frame_idx"])
                    cx = float(row["center_x"])
                    cy = float(row["center_y"])
                except (TypeError, ValueError):
                    continue

                grouped.setdefault(sequence_id, []).append((frame_idx, img_path, cx, cy))
                if row.get("split"):
                    split_by_sequence[sequence_id] = self._normalize_row_split(row["split"])

        items = list(grouped.items())
        if not items:
            return []

        has_explicit_split = any(seq_id in split_by_sequence for seq_id, _ in items)
        if has_explicit_split:
            filtered_items = [
                (seq_id, frames)
                for seq_id, frames in items
                if split_by_sequence.get(seq_id, "train") == requested_split
            ]
        else:
            ordered = sorted(items, key=lambda item: item[0])
            rng = random.Random(seed)
            rng.shuffle(ordered)
            split_idx = int(len(ordered) * train_ratio)
            if requested_split == "train":
                filtered_items = ordered[:split_idx]
            else:
                filtered_items = ordered[split_idx:]

        samples: List[Tuple[str, List[str], np.ndarray]] = []
        for sequence_id, frames in filtered_items:
            frames_sorted = sorted(frames, key=lambda item: item[0])
            frame_paths = [img_path for _, img_path, _, _ in frames_sorted]
            centers = np.asarray([(cx, cy) for _, _, cx, cy in frames_sorted], dtype=np.float32).reshape(-1, 2)
            samples.append((sequence_id, frame_paths, centers))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for _, frame_paths, centers in self.samples:
            frames = []
            failed = False
            for frame_path in frame_paths:
                img = self._read_image_bgr(frame_path)
                if img is None:
                    failed = True
                    break
                frames.append(img)
            if failed or not frames:
                continue
            yield np.stack(frames, axis=0), centers.copy()


SynTrackDataset = TrackingVehicleDataset

__all__ = ["TrackingVehicleDataset", "SynTrackDataset"]

from __future__ import annotations

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


class VehicleCenterDataset:
    """
    Iterable over the synthetic vehicle dataset producing (image_bgr, centers).

    Data Format:
    - Labels are stored in root/annotations.csv
    - Required columns: image, center_x, center_y
    - Optional split column: train / val / test

    Expected directory layout:
    - root/random/images/{split}/*.png
    - root/random/annotations.csv
    Or pass the random directory directly:
    - root/images/{split}/*.png
    - root/annotations.csv
    Or flat structure (split by ratio):
    - root/images/*.png
    - root/annotations.csv

    Output format matches synped_loader.py:
    - image_bgr: np.ndarray, shape (H, W, 3), uint8
    - centers: np.ndarray, shape (N, 2), float32
    """

    def __init__(
        self,
        root: str,
        split: str = "Train",
        annotation_file: str = "annotations.csv",
        train_ratio: float = 0.8,
        seed: int = 42,
        default_variant: str = "random",
    ) -> None:
        self.root = self._resolve_root(root, annotation_file=annotation_file, default_variant=default_variant)
        self.split = split
        self.annotation_path = os.path.join(self.root, annotation_file)
        self.img_dir = os.path.join(self.root, "images")

        if not os.path.isfile(self.annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Images dir not found: {self.img_dir}")

        requested_split = self._normalize_requested_split(split)
        self.samples = self._load_samples(requested_split=requested_split, train_ratio=train_ratio, seed=seed)

        if not self.samples:
            raise RuntimeError(f"No samples found for split={split!r} in {self.root}")

    @staticmethod
    def _resolve_root(root: str, annotation_file: str, default_variant: str) -> str:
        root = os.path.abspath(os.path.expanduser(root))
        if os.path.isfile(os.path.join(root, annotation_file)):
            return root

        variant = (default_variant or "").strip()
        if variant:
            candidate = os.path.join(root, variant)
            if os.path.isfile(os.path.join(candidate, annotation_file)):
                return candidate

        return root

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

    def _load_samples(self, requested_split: str, train_ratio: float, seed: int) -> List[Tuple[str, np.ndarray]]:
        grouped: "OrderedDict[str, List[Tuple[float, float]]]" = OrderedDict()
        split_by_image: dict[str, str] = {}

        with open(self.annotation_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_columns = {"image", "center_x", "center_y"}
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise RuntimeError(
                    f"Annotation file must contain columns {sorted(required_columns)}: {self.annotation_path}"
                )

            for row in reader:
                rel_path = (row.get("image") or "").strip()
                if not rel_path:
                    continue

                img_path = os.path.normpath(os.path.join(self.root, rel_path.replace("/", os.sep)))
                if not os.path.isfile(img_path):
                    continue

                try:
                    cx = float(row["center_x"])
                    cy = float(row["center_y"])
                except (TypeError, ValueError):
                    continue

                grouped.setdefault(img_path, []).append((cx, cy))
                if row.get("split"):
                    split_by_image[img_path] = self._normalize_row_split(row["split"])

        items = list(grouped.items())
        if not items:
            return []

        has_explicit_split = any(path in split_by_image for path, _ in items)
        if has_explicit_split:
            filtered = [
                (img_path, centers)
                for img_path, centers in items
                if split_by_image.get(img_path, "train") == requested_split
            ]
        else:
            ordered = sorted(items, key=lambda item: item[0])
            rng = random.Random(seed)
            rng.shuffle(ordered)
            split_idx = int(len(ordered) * train_ratio)
            if requested_split == "train":
                filtered = ordered[:split_idx]
            else:
                filtered = ordered[split_idx:]

        return [
            (img_path, np.asarray(centers, dtype=np.float32).reshape(-1, 2))
            for img_path, centers in filtered
        ]

    @staticmethod
    def _read_image_bgr(img_path: str) -> np.ndarray | None:
        if cv2 is not None:
            return cv2.imread(img_path, cv2.IMREAD_COLOR)

        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
        return arr[:, :, ::-1].copy()

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for img_path, centers in self.samples:
            img = self._read_image_bgr(img_path)
            if img is None:
                continue
            yield img, centers.copy()


SynCarDataset = VehicleCenterDataset

__all__ = ["VehicleCenterDataset", "SynCarDataset"]

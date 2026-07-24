import os
from typing import Iterator, List, Tuple

import cv2
import numpy as np


class LocalizationDataset:
    """
    Iterable over synthetic dataset producing (image_bgr, centers) where centers are
    person instance centers derived from the instance mask per image.

    Expected directory layout:
    - root/PNGImages/*.png
    - root/PedMasks/*_mask.png
    """

    def __init__(self, root: str, split: str = "Train") -> None:
        # Penn-Fudan doesn't have explicit Train/Test in the original release.
        # We accept split for API symmetry but simply read all images under PNGImages.
        self.root = root
        self.img_dir = os.path.join(self.root, "images")
        self.label_dir = os.path.join(self.root, "labels")
        if split == "Train":
            self.img_dir = os.path.join(self.img_dir, "train")
            self.label_dir = os.path.join(self.label_dir, "train")
        elif split == "Test":
            self.img_dir = os.path.join(self.img_dir, "val")
            self.label_dir = os.path.join(self.label_dir, "val")
        self.split = split


        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"images dir not found: {self.img_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"label dir not found: {self.label_dir}")

        self.images: List[str] = []
        for fn in sorted(os.listdir(self.img_dir)):
            lf = fn.lower()
            if lf.endswith(".png") or lf.endswith(".jpg") or lf.endswith(".jpeg") or lf.endswith(".bmp"):
                self.images.append(os.path.join(self.img_dir, fn))
        if not self.images:
            raise RuntimeError("No images found under PNGImages")

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for img_path in self.images:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.label_dir, f"{stem}.txt")
            centers: List[Tuple[float, float]] = []
            if os.path.isfile(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 2:
                            continue
                        try:
                            cx, cy = map(float, parts)
                        except ValueError:
                            continue
                        centers.append((cx*256, cy*256))
            centers_arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
            yield img, centers_arr


__all__ = ["LocalizationDataset"]




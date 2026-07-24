import os
from typing import Iterator, List, Tuple

import cv2
import numpy as np


def _box_to_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x_min, y_min, x_max, y_max = box
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def _find_mask_path(masks_dir: str, stem: str) -> str | None:
    """Locate the mask file for a given image stem in Penn-Fudan.

    Common layouts:
    - PedMasks/<stem>_mask.png (official)
    - PedMasks/<stem>.png (fallback)
    - Any file in PedMasks that starts with <stem>
    """
    cand1 = os.path.join(masks_dir, f"{stem}_mask.png")
    if os.path.isfile(cand1):
        return cand1
    cand2 = os.path.join(masks_dir, f"{stem}.png")
    if os.path.isfile(cand2):
        return cand2
    if os.path.isdir(masks_dir):
        for fn in os.listdir(masks_dir):
            if os.path.splitext(fn)[0].startswith(stem):
                p = os.path.join(masks_dir, fn)
                if os.path.isfile(p):
                    return p
    return None


class PennFudanDataset:
    """
    Iterable over Penn-Fudan dataset producing (image_bgr, centers) where centers are
    person instance centers derived from the instance mask per image.

    Expected directory layout:
    - root/PNGImages/*.png
    - root/PedMasks/*_mask.png
    """

    def __init__(self, root: str, split: str = "Train") -> None:
        # Penn-Fudan doesn't have explicit Train/Test in the original release.
        # We accept split for API symmetry but simply read all images under PNGImages.
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, "PNGImages")
        self.mask_dir = os.path.join(root, "PedMasks")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"PNGImages dir not found: {self.img_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"PedMasks dir not found: {self.mask_dir}")

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
            mask_path = _find_mask_path(self.mask_dir, stem)
            centers: List[Tuple[float, float]] = []
            if mask_path is not None and os.path.isfile(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    if mask.ndim == 3:
                        # In case mask is RGB, convert to single channel
                        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    else:
                        mask_gray = mask
                    # Unique instance ids > 0
                    ids = np.unique(mask_gray)
                    ids = ids[ids > 0]
                    for inst_id in ids.tolist():
                        ys, xs = np.where(mask_gray == inst_id)
                        if ys.size == 0 or xs.size == 0:
                            continue
                        y_min, y_max = int(ys.min()), int(ys.max())
                        x_min, x_max = int(xs.min()), int(xs.max())
                        # discard degenerate boxes
                        if x_max <= x_min or y_max <= y_min:
                            continue
                        cx, cy = _box_to_center((x_min, y_min, x_max, y_max))
                        centers.append((cx, cy))
            centers_arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
            yield img, centers_arr


__all__ = ["PennFudanDataset"]



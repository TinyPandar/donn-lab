import os
import math
import csv
import cv2
import torch
import numpy as np
import torch.nn.functional as F

try:
    from inria_loader import InriaPersonDataset
except Exception:
    InriaPersonDataset = None

try:
    from data.penn_fudan_loader import PennFudanDataset
except Exception:
    PennFudanDataset = None

try:
    from data.synthetic_loader import LocalizationDataset
except Exception:
    LocalizationDataset = None

try:
    from data.synped_loader import SynPedDataset
except Exception:
    SynPedDataset = None

try:
    from data.tracking_vehicle_loader import TrackingVehicleDataset
except Exception:
    TrackingVehicleDataset = None

try:
    from vehicle_center_loader import VehicleCenterDataset
except Exception:
    VehicleCenterDataset = None

# Default settings (can be updated via update_config)
H_in, W_in = 128, 128
H_out, W_out = 128, 128
INPUT_MODE = "rgb"
INPUT_MODE_EXPLICIT = False
AUG_ENABLE = True
AUG_HFLIP_P = 0.5
AUG_COLOR_P = 0.8
AUG_BLUR_P = 0.2
AUG_NOISE_P = 0.2
ZOOM_CROP_MODE = "none"
ZOOM_CROP_FACTOR = 1.0
VEHICLE_CHANNEL_MODE = "rgb"
VEHICLE_CHANNEL = "r"
VEHICLE_CHANNEL_INVERT = False
VEHICLE_CHANNEL_P_LOW = 2.0
VEHICLE_CHANNEL_P_HIGH = 98.0
VEHICLE_TARGET_MODE = "center"
VEHICLE_TARGET_COLUMN = "box_mask"
MNIST_TARGET_MODE = "coord"

_TRACKING_VEHICLE_TOTAL_FRAMES_CACHE: dict[tuple[str, str], int] = {}

def update_config(h_in, w_in, h_out, w_out, aug_params=None):
    global H_in, W_in, H_out, W_out, INPUT_MODE, INPUT_MODE_EXPLICIT
    global AUG_ENABLE, AUG_HFLIP_P, AUG_COLOR_P, AUG_BLUR_P, AUG_NOISE_P
    global ZOOM_CROP_MODE, ZOOM_CROP_FACTOR
    global VEHICLE_CHANNEL_MODE, VEHICLE_CHANNEL, VEHICLE_CHANNEL_INVERT, VEHICLE_CHANNEL_P_LOW, VEHICLE_CHANNEL_P_HIGH
    global VEHICLE_TARGET_MODE, VEHICLE_TARGET_COLUMN, MNIST_TARGET_MODE
    H_in, W_in = h_in, w_in
    H_out, W_out = h_out, w_out
    if aug_params:
        INPUT_MODE = str(aug_params.get('input_mode', INPUT_MODE)).lower()
        INPUT_MODE_EXPLICIT = bool(aug_params.get('input_mode_explicit', INPUT_MODE_EXPLICIT))
        AUG_ENABLE = aug_params.get('enable_aug', AUG_ENABLE)
        AUG_HFLIP_P = aug_params.get('aug_hflip_p', AUG_HFLIP_P)
        AUG_COLOR_P = aug_params.get('aug_color_p', AUG_COLOR_P)
        AUG_BLUR_P = aug_params.get('aug_blur_p', AUG_BLUR_P)
        AUG_NOISE_P = aug_params.get('aug_noise_p', AUG_NOISE_P)
        ZOOM_CROP_MODE = aug_params.get('zoom_crop_mode', ZOOM_CROP_MODE)
        ZOOM_CROP_FACTOR = aug_params.get('zoom_crop_factor', ZOOM_CROP_FACTOR)
        VEHICLE_CHANNEL_MODE = str(aug_params.get('vehicle_channel_mode', VEHICLE_CHANNEL_MODE)).lower()
        VEHICLE_CHANNEL = str(aug_params.get('vehicle_channel', VEHICLE_CHANNEL)).lower()
        VEHICLE_CHANNEL_INVERT = bool(aug_params.get('vehicle_channel_invert', VEHICLE_CHANNEL_INVERT))
        VEHICLE_CHANNEL_P_LOW = float(aug_params.get('vehicle_channel_p_low', VEHICLE_CHANNEL_P_LOW))
        VEHICLE_CHANNEL_P_HIGH = float(aug_params.get('vehicle_channel_p_high', VEHICLE_CHANNEL_P_HIGH))
        VEHICLE_TARGET_MODE = str(aug_params.get('vehicle_target_mode', VEHICLE_TARGET_MODE)).lower()
        VEHICLE_TARGET_COLUMN = str(aug_params.get('vehicle_target_column', VEHICLE_TARGET_COLUMN))
        MNIST_TARGET_MODE = str(aug_params.get('mnist_target_mode', MNIST_TARGET_MODE)).lower()

def _ceil_div(n: int, d: int) -> int:
    n = int(n)
    d = int(d)
    if d <= 0:
        return 0
    return int((n + d - 1) // d)

def _count_valid_label_files(label_dir: str, multiple_objects: bool) -> int:
    if not label_dir or (not os.path.isdir(label_dir)):
        return 0
    total = 0
    for fn in os.listdir(label_dir):
        if not fn.lower().endswith(".txt"):
            continue
        fp = os.path.join(label_dir, fn)
        if not os.path.isfile(fp):
            continue
        valid_lines = 0
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    try:
                        float(parts[0])
                        float(parts[1])
                    except Exception:
                        continue
                    valid_lines += 1
                    if multiple_objects and valid_lines >= 1:
                        break
                    if not multiple_objects and valid_lines > 1:
                        break
        except Exception:
            continue

        if (multiple_objects and valid_lines >= 1) or (not multiple_objects and valid_lines == 1):
            total += 1
    return total

def estimate_total_batches(
    dataset_name: str,
    data_root: str,
    split: str,
    batch_size: int,
    max_batches: int,
    label_filter: str,
    multiple_objects: bool,
) -> int | None:
    if int(max_batches) > 0:
        return int(max_batches)
    if int(batch_size) <= 0:
        return None

    if dataset_name == "synped":
        if SynPedDataset is None:
            return None
        ds = SynPedDataset(data_root, split=split)
        n = _count_valid_label_files(getattr(ds, "label_dir", ""), multiple_objects=multiple_objects)
        return _ceil_div(n, batch_size) if n > 0 else None

    if dataset_name == "syn":
        if LocalizationDataset is None:
            return None
        ds = LocalizationDataset(data_root, split=split)
        n = _count_valid_label_files(getattr(ds, "label_dir", ""), multiple_objects=multiple_objects)
        return _ceil_div(n, batch_size) if n > 0 else None

    if dataset_name == "mnist":
        import torchvision
        tfm = _mnist_letterbox_transform((int(H_in), int(W_in)))
        train_flag = (split.lower() == "train")
        ds = torchvision.datasets.MNIST(root=data_root, train=train_flag, transform=tfm, download=True)
        n = int(len(ds))
        return _ceil_div(n, batch_size) if n > 0 else None

    if dataset_name == "inria":
        if InriaPersonDataset is None:
            return None
        ds = InriaPersonDataset(data_root, split=split, include_negatives=False)
        n = int(len(ds))
        return _ceil_div(n, batch_size) if n > 0 else None

    if dataset_name == "pennfudan":
        if PennFudanDataset is None:
            return None
        ds = PennFudanDataset(data_root, split=split)
        n = int(len(ds))
        return _ceil_div(n, batch_size) if n > 0 else None

    if dataset_name in ("tracking_vehicle", "syntrack", "tracking"):
        if TrackingVehicleDataset is None:
            return None
        cache_key = (data_root, split)
        cached = _TRACKING_VEHICLE_TOTAL_FRAMES_CACHE.get(cache_key)
        if cached is None:
            try:
                ds = TrackingVehicleDataset(data_root, split=split)
            except Exception:
                return None
            total_frames = 0
            for _seq_id, frame_paths, centers in getattr(ds, "samples", []):
                n_frames = int(len(frame_paths))
                n_centers = int(centers.shape[0]) if getattr(centers, "ndim", 0) == 2 else 0
                total_frames += min(n_frames, n_centers)
            cached = int(total_frames)
            _TRACKING_VEHICLE_TOTAL_FRAMES_CACHE[cache_key] = cached
        return _ceil_div(cached, batch_size) if cached > 0 else None

    if dataset_name == "vehicle":
        if VehicleCenterDataset is None:
            return None
        try:
            ds = VehicleCenterDataset(data_root, split=split)
            n = len(ds.samples)
            return _ceil_div(n, batch_size) if n > 0 else None
        except Exception:
            return None

    return None

def _rgb_to_model_array(img_rgb: np.ndarray) -> np.ndarray:
    mode = str(INPUT_MODE or "rgb").lower()
    img_rgb = img_rgb.astype(np.float32) / 255.0

    if mode in {"auto", "default", "rgb", "bgr", "none", "off"}:
        return img_rgb
    if mode in {"gray", "grey", "grayscale", "luminance", "lum"}:
        weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float32).reshape(1, 1, 3)
        return (img_rgb * weights).sum(axis=2, keepdims=True)
    if mode == "mean":
        return img_rgb.mean(axis=2, keepdims=True)

    channel_map = {"r": 0, "red": 0, "0": 0, "g": 1, "green": 1, "1": 1, "b": 2, "blue": 2, "2": 2}
    if mode in channel_map:
        return img_rgb[:, :, channel_map[mode] : channel_map[mode] + 1]

    raise ValueError("input_mode must be one of {'auto','rgb','gray','mean','r','g','b'}")


def prepare_image_bgr_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """将 OpenCV BGR 图像预处理为 [1, C, H_in, W_in] 的 torch.float32 张量，范围 [0,1]。"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (W_in, H_in), interpolation=cv2.INTER_AREA)
    img_arr = _rgb_to_model_array(img_rgb)
    x = torch.from_numpy(np.ascontiguousarray(img_arr)).permute(2, 0, 1).unsqueeze(0)
    return x

def _augment_image_and_centers(img_bgr: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aug_img = img_bgr
    aug_centers = centers.copy() if centers is not None else None

    H, W = aug_img.shape[:2]

    if AUG_HFLIP_P > 0 and np.random.rand() < AUG_HFLIP_P:
        aug_img = cv2.flip(aug_img, 1)
        if aug_centers is not None and aug_centers.size > 0:
            aug_centers[:, 0] = (W - 1) - aug_centers[:, 0]

    if AUG_COLOR_P > 0 and np.random.rand() < AUG_COLOR_P:
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV)
        h_shift = np.random.randint(-10, 11)
        s_scale = np.random.uniform(0.8, 1.2)
        v_scale = np.random.uniform(0.8, 1.2)
        h = hsv[:, :, 0].astype(np.int32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)
        h = (h + h_shift) % 180
        s = np.clip(s * s_scale, 0, 255)
        v = np.clip(v * v_scale, 0, 255)
        hsv[:, :, 0] = h.astype(np.uint8)
        hsv[:, :, 1] = s.astype(np.uint8)
        hsv[:, :, 2] = v.astype(np.uint8)
        aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if AUG_BLUR_P > 0 and np.random.rand() < AUG_BLUR_P:
        k = np.random.choice([3, 5])
        aug_img = cv2.GaussianBlur(aug_img, (k, k), 0)

    if AUG_NOISE_P > 0 and np.random.rand() < AUG_NOISE_P:
        sigma = np.random.uniform(5.0, 12.0)
        noise = np.random.normal(0.0, sigma, size=aug_img.shape).astype(np.float32)
        tmp = aug_img.astype(np.float32) + noise
        aug_img = np.clip(tmp, 0, 255).astype(np.uint8)

    return aug_img, (aug_centers if aug_centers is not None else centers)

def _apply_label_agnostic_zoom_crop(
    img_bgr: np.ndarray,
    centers: np.ndarray,
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(ZOOM_CROP_MODE or "none").lower()
    factor = float(ZOOM_CROP_FACTOR or 1.0)
    if mode in ("none", "off", "false") or factor <= 1.0:
        return img_bgr, centers
    if mode not in ("center", "random"):
        raise ValueError("zoom_crop_mode must be one of {'none', 'center', 'random'}")

    h, w = img_bgr.shape[:2]
    new_h = max(int(round(h * factor)), h)
    new_w = max(int(round(w * factor)), w)
    zoomed = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    max_left = max(new_w - w, 0)
    max_top = max(new_h - h, 0)
    if mode == "random" and split == "Train":
        left = int(np.random.randint(0, max_left + 1)) if max_left > 0 else 0
        top = int(np.random.randint(0, max_top + 1)) if max_top > 0 else 0
    else:
        left = max_left // 2
        top = max_top // 2

    cropped = zoomed[top:top + h, left:left + w]
    if cropped.shape[0] != h or cropped.shape[1] != w:
        cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    if centers is None or getattr(centers, "size", 0) == 0:
        return cropped, centers
    new_centers = centers.astype(np.float32, copy=True)
    new_centers[:, 0] = new_centers[:, 0] * (new_w / max(w, 1)) - left
    new_centers[:, 1] = new_centers[:, 1] * (new_h / max(h, 1)) - top
    keep = (
        (new_centers[:, 0] >= 0)
        & (new_centers[:, 0] < w)
        & (new_centers[:, 1] >= 0)
        & (new_centers[:, 1] < h)
    )
    return cropped, new_centers[keep]

def _pick_and_scale_center(centers: np.ndarray, orig_h: int, orig_w: int) -> tuple | None:
    if centers is None or centers.size == 0:
        return None
    cx = centers[:, 0]
    cy = centers[:, 1]
    rows = np.clip(np.round(cy / orig_h * H_out).astype(np.int64), 0, H_out - 1)
    cols = np.clip(np.round(cx / orig_w * W_out).astype(np.int64), 0, W_out - 1)
    return int(rows[0]), int(cols[0])

def mnist_label_to_coord(label: int, h: int, w: int) -> tuple[int, int]:
    num = 10
    idx = int(label) % num
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    r = 0.35 * min(h, w)
    theta = 2.0 * np.pi * (idx / num)
    y = cy + r * np.sin(theta)
    x = cx + r * np.cos(theta)
    rr = int(np.clip(np.round(y), 0, h - 1))
    cc = int(np.clip(np.round(x), 0, w - 1))
    return rr, cc


def _mnist_letterbox_transform(image_hw: tuple[int, int]):
    """Resize MNIST without distortion, then zero-pad to the optical input grid."""
    from torchvision import transforms

    h_in, w_in = int(image_hw[0]), int(image_hw[1])
    side = min(h_in, w_in)
    pad_h, pad_w = h_in - side, w_in - side
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return transforms.Compose(
        [
            transforms.Resize((side, side)),
            transforms.Pad((left, top, right, bottom), fill=0),
            transforms.ToTensor(),
        ]
    )

def _batch_iter_inria(root: str, split: str, batch_size: int, multiple_objects: bool = True):
    if InriaPersonDataset is None: raise RuntimeError("InriaPersonDataset not available")
    ds = InriaPersonDataset(root, split=split, include_negatives=False)
    imgs, coords = [], []
    for img_bgr, centers in ds:
        if split == "Train" and AUG_ENABLE:
            img_bgr, centers = _augment_image_and_centers(img_bgr, centers)
        img_bgr, centers = _apply_label_agnostic_zoom_crop(img_bgr, centers, split=split)
        if not multiple_objects:
            if centers is None or centers.size == 0 or getattr(centers, "shape", None) is None or len(centers.shape) != 2 or centers.shape[0] != 1:
                continue
        orig_h, orig_w = img_bgr.shape[:2]
        rc = _pick_and_scale_center(centers, orig_h, orig_w)
        if rc is None: continue
        x1 = prepare_image_bgr_to_tensor(img_bgr)
        imgs.append(x1)
        coords.append(torch.tensor([rc[0], rc[1]]).unsqueeze(0))
        if len(imgs) == batch_size:
            yield torch.cat(imgs, dim=0), torch.cat(coords, dim=0).long()
            imgs, coords = [], []
    if imgs:
        yield torch.cat(imgs, dim=0), torch.cat(coords, dim=0).long()

def _batch_iter_pennfudan(root: str, split: str, batch_size: int, multiple_objects: bool = False):
    if PennFudanDataset is None: raise RuntimeError("PennFudanDataset not available")
    ds = PennFudanDataset(root, split=split)
    imgs, coords = [], []
    for img_bgr, centers in ds:
        if split == "Train" and AUG_ENABLE:
            img_bgr, centers = _augment_image_and_centers(img_bgr, centers)
        img_bgr, centers = _apply_label_agnostic_zoom_crop(img_bgr, centers, split=split)
        if not multiple_objects:
            if centers is None or centers.size == 0 or getattr(centers, "shape", None) is None or len(centers.shape) != 2 or centers.shape[0] != 1:
                continue
        orig_h, orig_w = img_bgr.shape[:2]
        rc = _pick_and_scale_center(centers, orig_h, orig_w)
        if rc is None: continue
        x1 = prepare_image_bgr_to_tensor(img_bgr)
        imgs.append(x1)
        coords.append(torch.tensor([rc[0], rc[1]]).unsqueeze(0))
        if len(imgs) == batch_size:
            yield torch.cat(imgs, dim=0), torch.cat(coords, dim=0).long()
            imgs, coords = [], []
    if imgs:
        yield torch.cat(imgs, dim=0), torch.cat(coords, dim=0).long()

def _batch_iter_mnist(root: str, split: str, batch_size: int, image_hw: tuple[int, int], out_hw: tuple[int, int]):
    import torchvision
    h_in, w_in = int(image_hw[0]), int(image_hw[1])
    h_out, w_out = int(out_hw[0]), int(out_hw[1])
    tfm = _mnist_letterbox_transform((h_in, w_in))
    train_flag = (split.lower() == "train")
    ds = torchvision.datasets.MNIST(root=root, train=train_flag, transform=tfm, download=True)
    imgs, targets = [], []
    for img_tensor, label in ds:
        if str(INPUT_MODE or "rgb").lower() in {"auto", "default", "rgb", "bgr", "none", "off"}:
            img_in = img_tensor.repeat(3, 1, 1) if img_tensor.ndim == 3 and img_tensor.shape[0] == 1 else img_tensor
        else:
            img_in = img_tensor[:1] if img_tensor.ndim == 3 else img_tensor.unsqueeze(0)
        imgs.append(img_in.unsqueeze(0))
        if MNIST_TARGET_MODE == "class":
            targets.append(torch.tensor([int(label)], dtype=torch.long))
        elif MNIST_TARGET_MODE == "coord":
            rr, cc = mnist_label_to_coord(int(label), h_out, w_out)
            targets.append(torch.tensor([[rr, cc]], dtype=torch.long))
        else:
            raise ValueError("mnist_target_mode must be one of {'coord', 'class'}")
        if len(imgs) == batch_size:
            yield torch.cat(imgs, dim=0), torch.cat(targets, dim=0).long()
            imgs, targets = [], []
    if imgs:
        yield torch.cat(imgs, dim=0), torch.cat(targets, dim=0).long()

def _batch_iter_syn(root: str, split: str, batch_size: int, multiple_objects: bool = True):
    if LocalizationDataset is None: raise RuntimeError("LocalizationDataset not available")
    ds = LocalizationDataset(root, split=split)
    imgs, coords = [], []
    for img_bgr, centers in ds:
        if split == "Train" and AUG_ENABLE:
            img_bgr, centers = _augment_image_and_centers(img_bgr, centers)
        img_bgr, centers = _apply_label_agnostic_zoom_crop(img_bgr, centers, split=split)
        if not multiple_objects:
            if centers is None or centers.size == 0 or getattr(centers, "shape", None) is None or len(centers.shape) != 2 or centers.shape[0] != 1:
                continue
        orig_h, orig_w = img_bgr.shape[:2]
        rc = _pick_and_scale_center(centers, orig_h, orig_w)
        if rc is None: continue
        x1 = prepare_image_bgr_to_tensor(img_bgr)
        imgs.append(x1)
        coords.append(torch.tensor([rc[0], rc[1]]).unsqueeze(0))
        if len(imgs) == batch_size:
            yield torch.cat(imgs, dim=0), torch.cat(coords, dim=0).long()
            imgs, coords = [], []
    if imgs:
        yield torch.cat(imgs, dim=0), torch.cat(coords, dim=0).long()

def _batch_iter_synped(root: str, split: str, batch_size: int, multiple_objects: bool = True):
    if SynPedDataset is None: raise RuntimeError("SynPedDataset not available")
    ds = SynPedDataset(root, split=split)
    imgs, coords = [], []
    for img_bgr, centers in ds:
        if split == "Train" and AUG_ENABLE:
            img_bgr, centers = _augment_image_and_centers(img_bgr, centers)
        img_bgr, centers = _apply_label_agnostic_zoom_crop(img_bgr, centers, split=split)
        if not multiple_objects:
            if centers is None or centers.size == 0 or getattr(centers, "shape", None) is None or len(centers.shape) != 2 or centers.shape[0] != 1:
                continue
        orig_h, orig_w = img_bgr.shape[:2]
        rc = _pick_and_scale_center(centers, orig_h, orig_w)
        if rc is None: continue
        x1 = prepare_image_bgr_to_tensor(img_bgr)
        imgs.append(x1)
        coords.append(torch.tensor([rc[0], rc[1]]).unsqueeze(0))
        if len(imgs) == batch_size:
            yield torch.cat(imgs, dim=0), torch.cat(coords, dim=0).long()
            imgs, coords = [], []
    if imgs:
        yield torch.cat(imgs, dim=0), torch.cat(coords, dim=0).long()


class _TrackingVehicleFrameDataset:
    """Frame-level view over TrackingVehicleDataset to match legacy (image_bgr, centers) API."""

    def __init__(self, root: str, split: str = "Train") -> None:
        if TrackingVehicleDataset is None:
            raise RuntimeError("TrackingVehicleDataset not available")
        self.seq_ds = TrackingVehicleDataset(root, split=split)

    def __len__(self) -> int:
        total = 0
        for _seq_id, frame_paths, centers in getattr(self.seq_ds, "samples", []):
            n_frames = int(len(frame_paths))
            n_centers = int(centers.shape[0]) if getattr(centers, "ndim", 0) == 2 else 0
            total += min(n_frames, n_centers)
        return int(total)

    def __iter__(self):
        for _seq_id, frame_paths, centers in getattr(self.seq_ds, "samples", []):
            if getattr(centers, "ndim", 0) != 2:
                continue
            n = min(int(len(frame_paths)), int(centers.shape[0]))
            for i in range(n):
                img = TrackingVehicleDataset._read_image_bgr(frame_paths[i])
                if img is None:
                    continue
                c = np.asarray([centers[i]], dtype=np.float32).reshape(1, 2)
                yield img, c


def _apply_vehicle_channel_mode(img_bgr: np.ndarray) -> np.ndarray:
    mode = str(VEHICLE_CHANNEL_MODE).lower()
    if mode in {"rgb", "bgr", "none", "off"}:
        return img_bgr

    channel_map = {"b": 0, "blue": 0, "0": 0, "g": 1, "green": 1, "1": 1, "r": 2, "red": 2, "2": 2}
    if mode in {"gray", "grey", "grayscale", "rgb_gray", "rgb_gray_grid"}:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    elif mode in {"random_channel", "random_rgb_channel", "random_bgr_channel", "random_single", "random_single_bright"}:
        ch_idx = int(np.random.randint(0, 3))
        gray = img_bgr[:, :, ch_idx].astype(np.float32)
    else:
        ch_idx = channel_map.get(str(VEHICLE_CHANNEL).lower(), 2)
        gray = img_bgr[:, :, ch_idx].astype(np.float32)

    if VEHICLE_CHANNEL_INVERT or mode in {"single_dark", "dark_bright"}:
        gray = 255.0 - gray

    if mode in {"single_bright", "bright", "car_bright", "foreground_bright", "single_dark", "dark_bright", "random_single_bright"}:
        lo = np.percentile(gray, VEHICLE_CHANNEL_P_LOW)
        hi = np.percentile(gray, VEHICLE_CHANNEL_P_HIGH)
        if hi > lo + 1e-6:
            gray = (gray - lo) * (255.0 / (hi - lo))
        gray = np.clip(gray, 0.0, 255.0)

    gray_u8 = gray.astype(np.uint8)
    return cv2.merge([gray_u8, gray_u8, gray_u8])


def _is_vehicle_mask_target() -> bool:
    return str(VEHICLE_TARGET_MODE).lower() in {"box_mask", "mask", "target_mask", "box"}


def _use_legacy_vehicle_channel_mode() -> bool:
    return not INPUT_MODE_EXPLICIT


def _load_vehicle_target_paths(root: str, annotation_path: str, target_column: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not os.path.isfile(annotation_path):
        return mapping
    with open(annotation_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "image" not in reader.fieldnames or target_column not in reader.fieldnames:
            return mapping
        for row in reader:
            rel_image = (row.get("image") or "").strip()
            rel_target = (row.get(target_column) or "").strip()
            if not rel_image or not rel_target:
                continue
            image_path = os.path.normpath(os.path.join(root, rel_image.replace("/", os.sep)))
            if os.path.isabs(rel_target):
                target_path = os.path.normpath(rel_target)
            else:
                target_path = os.path.normpath(os.path.join(root, rel_target.replace("/", os.sep)))
            mapping[image_path] = target_path
    return mapping


def _prepare_vehicle_mask_target(mask_path: str) -> torch.Tensor | None:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[:2] != (H_out, W_out):
        mask = cv2.resize(mask, (W_out, H_out), interpolation=cv2.INTER_AREA)
    target = mask.astype(np.float32) / 255.0
    return torch.from_numpy(target).unsqueeze(0)


def _batch_iter_vehicle(root: str, split: str, batch_size: int, multiple_objects: bool = True):
    if VehicleCenterDataset is None:
        raise RuntimeError("VehicleCenterDataset not available")
    ds = VehicleCenterDataset(root, split=split)
    use_mask_target = _is_vehicle_mask_target()
    target_paths = _load_vehicle_target_paths(
        root=getattr(ds, "root", root),
        annotation_path=getattr(ds, "annotation_path", os.path.join(root, "annotations.csv")),
        target_column=VEHICLE_TARGET_COLUMN,
    ) if use_mask_target else {}
    imgs, targets = [], []

    if use_mask_target:
        for img_path, centers in getattr(ds, "samples", []):
            img_bgr = VehicleCenterDataset._read_image_bgr(img_path)
            if img_bgr is None:
                continue
            target_path = target_paths.get(os.path.normpath(img_path))
            if not target_path or not os.path.isfile(target_path):
                continue
            mask_t = _prepare_vehicle_mask_target(target_path)
            if mask_t is None:
                continue
            if _use_legacy_vehicle_channel_mode():
                img_bgr = _apply_vehicle_channel_mode(img_bgr)
            x1 = prepare_image_bgr_to_tensor(img_bgr)
            imgs.append(x1)
            targets.append(mask_t)
            if len(imgs) == batch_size:
                yield torch.cat(imgs, dim=0), torch.cat(targets, dim=0).float()
                imgs, targets = [], []
        if imgs:
            yield torch.cat(imgs, dim=0), torch.cat(targets, dim=0).float()
        return

    for img_bgr, centers in ds:
        if split == "Train" and AUG_ENABLE:
            img_bgr, centers = _augment_image_and_centers(img_bgr, centers)
        if not multiple_objects:
            if centers is None or centers.size == 0 or getattr(centers, "shape", None) is None or len(centers.shape) != 2 or centers.shape[0] != 1:
                continue
        orig_h, orig_w = img_bgr.shape[:2]
        rc = _pick_and_scale_center(centers, orig_h, orig_w)
        if rc is None:
            continue
        if _use_legacy_vehicle_channel_mode():
            img_bgr = _apply_vehicle_channel_mode(img_bgr)
        x1 = prepare_image_bgr_to_tensor(img_bgr)
        imgs.append(x1)
        targets.append(torch.tensor([rc[0], rc[1]]).unsqueeze(0))
        if len(imgs) == batch_size:
            yield torch.cat(imgs, dim=0), torch.cat(targets, dim=0).long()
            imgs, targets = [], []
    if imgs:
        yield torch.cat(imgs, dim=0), torch.cat(targets, dim=0).long()

def _batch_iter_tracking_vehicle(root: str, split: str, batch_size: int, multiple_objects: bool = False):
    frame_ds = _TrackingVehicleFrameDataset(root, split=split)
    imgs, coords = [], []
    for img_bgr, centers in frame_ds:
        if split == "Train" and AUG_ENABLE:
            img_bgr, centers = _augment_image_and_centers(img_bgr, centers)
        if not multiple_objects:
            if centers is None or centers.size == 0 or getattr(centers, "shape", None) is None or len(centers.shape) != 2 or centers.shape[0] != 1:
                continue
        orig_h, orig_w = img_bgr.shape[:2]
        rc = _pick_and_scale_center(centers, orig_h, orig_w)
        if rc is None: continue
        x1 = prepare_image_bgr_to_tensor(img_bgr)
        imgs.append(x1)
        coords.append(torch.tensor([rc[0], rc[1]]).unsqueeze(0))
        if len(imgs) == batch_size:
            yield torch.cat(imgs, dim=0), torch.cat(coords, dim=0).long()
            imgs, coords = [], []
    if imgs:
        yield torch.cat(imgs, dim=0), torch.cat(coords, dim=0).long()


def get_batch_iter(dataset_name, data_root, split, batch_size, label_filter="person", multiple_objects=True):
    if dataset_name == "inria":
        return _batch_iter_inria(data_root, split, batch_size, multiple_objects)
    elif dataset_name == "pennfudan":
        return _batch_iter_pennfudan(data_root, split, batch_size, multiple_objects)
    elif dataset_name == "syn":
        return _batch_iter_syn(data_root, split, batch_size, multiple_objects)
    elif dataset_name == "synped":
        return _batch_iter_synped(data_root, split, batch_size, multiple_objects)
    elif dataset_name in ("tracking_vehicle", "syntrack", "tracking"):
        return _batch_iter_tracking_vehicle(data_root, split, batch_size, multiple_objects)
    elif dataset_name == "vehicle":
        return _batch_iter_vehicle(data_root, split, batch_size, multiple_objects)
    elif dataset_name == "mnist":
        return _batch_iter_mnist(data_root, split, batch_size, (H_in, W_in), (H_out, W_out))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_dataset(dataset_name, data_root, split, label_filter="person"):
    if dataset_name == "inria":
        if InriaPersonDataset is None: return None
        return InriaPersonDataset(data_root, split=split, include_negatives=False)
    elif dataset_name == "pennfudan":
        if PennFudanDataset is None: return None
        return PennFudanDataset(data_root, split=split)
    elif dataset_name == "syn":
        if LocalizationDataset is None: return None
        return LocalizationDataset(data_root, split=split)
    elif dataset_name == "synped":
        if SynPedDataset is None: return None
        return SynPedDataset(data_root, split=split)
    elif dataset_name in ("tracking_vehicle", "syntrack", "tracking"):
        if TrackingVehicleDataset is None: return None
        return _TrackingVehicleFrameDataset(data_root, split=split)
    elif dataset_name == "vehicle":
        if VehicleCenterDataset is None: return None
        return VehicleCenterDataset(data_root, split=split)
    elif dataset_name == "mnist":
        import torchvision
        is_train = (split == "Train")
        tfm = _mnist_letterbox_transform((H_in, W_in))
        return torchvision.datasets.MNIST(root=data_root, train=is_train, transform=tfm, download=True)
    return None

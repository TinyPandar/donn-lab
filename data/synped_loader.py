import os
import random
from typing import Iterator, List, Tuple

import cv2
import numpy as np


class SynPedDataset:
    """
    Iterable over dataset producing (image_bgr, centers).
    
    Data Format:
    - Labels are text files with integer coordinates: "Center_X Center_Y"
    - No header, no score column.
    
    Expected directory layout:
    - root/images/{split}/*.png
    - root/labels/{split}/*.txt
    Or flat structure (split by ratio):
    - root/images/*.png
    - root/labels/*.txt
    """

    def __init__(self, root: str, split: str = "Train", label_folder_name: str = "labels", train_ratio: float = 0.8, seed: int = 42) -> None:
        self.root = root
        self.split = split
        
        # 路径构建
        self.img_dir = os.path.join(self.root, "images")
        self.label_dir = os.path.join(self.root, label_folder_name)

        # 自动适配 split 子目录 (兼容 train/val 结构或扁平结构)
        has_split_dir = False
        if split == "Train":
            if os.path.isdir(os.path.join(self.img_dir, "train")):
                self.img_dir = os.path.join(self.img_dir, "train")
                self.label_dir = os.path.join(self.label_dir, "train")
                has_split_dir = True
        elif split == "Test":
            if os.path.isdir(os.path.join(self.img_dir, "val")):
                self.img_dir = os.path.join(self.img_dir, "val")
                self.label_dir = os.path.join(self.label_dir, "val")
                has_split_dir = True

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Images dir not found: {self.img_dir}")
        
        # 预加载图片列表
        self.images: List[str] = []
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
        
        # 排序确保多卡训练或调试时顺序一致
        all_files = sorted(os.listdir(self.img_dir))
        for fn in all_files:
            if os.path.splitext(fn.lower())[1] in valid_exts:
                self.images.append(os.path.join(self.img_dir, fn))
        
        if not self.images:
            raise RuntimeError(f"No images found in {self.img_dir}")

        # 如果没有物理分割子目录，则执行按比例分割
        if not has_split_dir:
            # 使用固定种子打乱，确保 Train/Test 看到的总集合一致且分割互斥
            # 注意：必须先排序再打乱 (上面已经 sorted)
            rng = random.Random(seed)
            rng.shuffle(self.images)
            
            split_idx = int(len(self.images) * train_ratio)
            
            if split == "Train":
                self.images = self.images[:split_idx]
            elif split == "Test":
                self.images = self.images[split_idx:]

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for img_path in self.images:
            # 1. 读取图像
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
                
            # 2. 构造标签路径
            # 假设文件名一一对应: /images/train/001.png -> /labels/train/001.txt
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.label_dir, f"{stem}.txt")
            
            centers: List[Tuple[float, float]] = []
            
            # 3. 解析坐标文件 (Integer X Y -> Float Tensor Ready)
            if os.path.isfile(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        # 严格只取前两列
                        if len(parts) < 2:
                            continue
                            
                        try:
                            # 即使文本是 "227"，float("227") -> 227.0
                            # 保持 float32 对回归任务的 Loss 计算更友好
                            cx = float(parts[0])
                            cy = float(parts[1])
                            centers.append((cx, cy))
                        except ValueError:
                            continue
            
            # 4. 格式化输出
            # 如果没有找到标签，这里会返回空数组 (0, 2)
            # 输出 dtype=float32 以适配 PyTorch/TensorFlow 默认类型
            centers_arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
            
            yield img, centers_arr

__all__ = ["SynPedDataset"]
"""
SPECTRA — PyTorch Datasets for Landsat 9 Training
Two dataset classes for the two-stage pipeline:
  - LandsatSRDataset:    200m TIR → 100m TIR (Model A: Super-Resolution)
  - LandsatColorDataset: 100m TIR → 100m RGB (Model B: Colorization)
"""

import os
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import config


class LandsatSRDataset(Dataset):
    """
    Super-Resolution dataset: 200m TIR → 100m TIR.
    
    Expected directory structure:
        sr/
        ├── train/
        │   ├── A/  (200m TIR patches — low-res input)
        │   └── B/  (100m TIR patches — high-res ground truth)
        └── val/
            ├── A/
            └── B/
    """

    def __init__(self, root: str, split: str = "train", img_size: int = 256, jitter_size: int = 286):
        super().__init__()
        self.split = split
        self.img_size = img_size
        self.jitter_size = jitter_size

        self.dir_low = Path(root) / split / "A"   # 200m TIR
        self.dir_high = Path(root) / split / "B"   # 100m TIR

        # Get sorted filenames present in both directories
        if self.dir_low.exists() and self.dir_high.exists():
            low_files = set(os.listdir(self.dir_low))
            high_files = set(os.listdir(self.dir_high))
            self.filenames = sorted(low_files & high_files)
        else:
            self.filenames = []

        if len(self.filenames) == 0:
            raise ValueError(
                f"No matching files found in {self.dir_low} and {self.dir_high}. "
                "Run preprocess_landsat.py first to generate training patches."
            )

        print(f"[SR {split.upper()}] Loaded {len(self.filenames)} paired patches (200m→100m)")

    def __len__(self) -> int:
        return len(self.filenames)

    def _apply_transforms(self, low_img: Image.Image, high_img: Image.Image):
        """Apply synchronized transforms to both images."""
        if self.split == "train" and config.RANDOM_JITTER:
            low_img = TF.resize(low_img, [self.jitter_size, self.jitter_size], interpolation=T.InterpolationMode.BICUBIC)
            high_img = TF.resize(high_img, [self.jitter_size, self.jitter_size], interpolation=T.InterpolationMode.BICUBIC)

            i, j, h, w = T.RandomCrop.get_params(low_img, output_size=(self.img_size, self.img_size))
            low_img = TF.crop(low_img, i, j, h, w)
            high_img = TF.crop(high_img, i, j, h, w)

            if config.RANDOM_FLIP and random.random() > 0.5:
                low_img = TF.hflip(low_img)
                high_img = TF.hflip(high_img)
        else:
            low_img = TF.resize(low_img, [self.img_size, self.img_size], interpolation=T.InterpolationMode.BICUBIC)
            high_img = TF.resize(high_img, [self.img_size, self.img_size], interpolation=T.InterpolationMode.BICUBIC)

        low_tensor = TF.to_tensor(low_img)    # (1, H, W)
        high_tensor = TF.to_tensor(high_img)   # (1, H, W)

        # Ensure single-channel
        if low_tensor.shape[0] == 3:
            low_tensor = low_tensor.mean(dim=0, keepdim=True)
        if high_tensor.shape[0] == 3:
            high_tensor = high_tensor.mean(dim=0, keepdim=True)

        # Normalize [0,1] → [-1,1]
        low_tensor = low_tensor * 2.0 - 1.0
        high_tensor = high_tensor * 2.0 - 1.0

        return low_tensor, high_tensor

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        low_img = Image.open(self.dir_low / fname).convert("L")
        high_img = Image.open(self.dir_high / fname).convert("L")
        return self._apply_transforms(low_img, high_img)


class LandsatColorDataset(Dataset):
    """
    Colorization dataset: 100m TIR → 100m RGB.
    
    Expected directory structure:
        color/
        ├── train/
        │   ├── A/  (100m TIR patches — grayscale input)
        │   └── B/  (100m RGB patches — color ground truth)
        └── val/
            ├── A/
            └── B/
    """

    def __init__(self, root: str, split: str = "train", img_size: int = 256, jitter_size: int = 286):
        super().__init__()
        self.split = split
        self.img_size = img_size
        self.jitter_size = jitter_size

        self.dir_ir = Path(root) / split / "A"    # 100m TIR
        self.dir_rgb = Path(root) / split / "B"    # 100m RGB

        if self.dir_ir.exists() and self.dir_rgb.exists():
            ir_files = set(os.listdir(self.dir_ir))
            rgb_files = set(os.listdir(self.dir_rgb))
            self.filenames = sorted(ir_files & rgb_files)
        else:
            self.filenames = []

        if len(self.filenames) == 0:
            raise ValueError(
                f"No matching files found in {self.dir_ir} and {self.dir_rgb}. "
                "Run preprocess_landsat.py first to generate training patches."
            )

        print(f"[COLOR {split.upper()}] Loaded {len(self.filenames)} paired patches (TIR→RGB)")

    def __len__(self) -> int:
        return len(self.filenames)

    def _apply_transforms(self, ir_img: Image.Image, rgb_img: Image.Image):
        """Apply synchronized transforms to both images."""
        if self.split == "train" and config.RANDOM_JITTER:
            ir_img = TF.resize(ir_img, [self.jitter_size, self.jitter_size], interpolation=T.InterpolationMode.BICUBIC)
            rgb_img = TF.resize(rgb_img, [self.jitter_size, self.jitter_size], interpolation=T.InterpolationMode.BICUBIC)

            i, j, h, w = T.RandomCrop.get_params(ir_img, output_size=(self.img_size, self.img_size))
            ir_img = TF.crop(ir_img, i, j, h, w)
            rgb_img = TF.crop(rgb_img, i, j, h, w)

            if config.RANDOM_FLIP and random.random() > 0.5:
                ir_img = TF.hflip(ir_img)
                rgb_img = TF.hflip(rgb_img)
        else:
            ir_img = TF.resize(ir_img, [self.img_size, self.img_size], interpolation=T.InterpolationMode.BICUBIC)
            rgb_img = TF.resize(rgb_img, [self.img_size, self.img_size], interpolation=T.InterpolationMode.BICUBIC)

        ir_tensor = TF.to_tensor(ir_img)     # (1, H, W) grayscale
        rgb_tensor = TF.to_tensor(rgb_img)   # (3, H, W) color

        if ir_tensor.shape[0] == 3:
            ir_tensor = ir_tensor.mean(dim=0, keepdim=True)

        # Normalize [0,1] → [-1,1]
        ir_tensor = ir_tensor * 2.0 - 1.0
        rgb_tensor = rgb_tensor * 2.0 - 1.0

        return ir_tensor, rgb_tensor

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        ir_img = Image.open(self.dir_ir / fname).convert("L")
        rgb_img = Image.open(self.dir_rgb / fname).convert("RGB")
        return self._apply_transforms(ir_img, rgb_img)


if __name__ == "__main__":
    # Quick test
    print("Testing LandsatSRDataset...")
    try:
        sr_ds = LandsatSRDataset(root=config.SR_DATA_ROOT, split="train")
        low, high = sr_ds[0]
        print(f"  Low-res (200m):  shape={low.shape}, range=[{low.min():.2f}, {low.max():.2f}]")
        print(f"  High-res (100m): shape={high.shape}, range=[{high.min():.2f}, {high.max():.2f}]")
    except Exception as e:
        print(f"  Skipped: {e}")

    print("\nTesting LandsatColorDataset...")
    try:
        color_ds = LandsatColorDataset(root=config.COLOR_DATA_ROOT, split="train")
        ir, rgb = color_ds[0]
        print(f"  IR (100m TIR):  shape={ir.shape}, range=[{ir.min():.2f}, {ir.max():.2f}]")
        print(f"  RGB (100m):     shape={rgb.shape}, range=[{rgb.min():.2f}, {rgb.max():.2f}]")
    except Exception as e:
        print(f"  Skipped: {e}")

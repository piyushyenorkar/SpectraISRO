"""
SPECTRA — PyTorch Dataset for Paired IR/RGB Images
Loads paired infrared and RGB images for pix2pix training.
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


class IRColorDataset(Dataset):
    """
    Paired IR/RGB dataset for pix2pix.

    Expected directory structure:
        data/
        ├── train/
        │   ├── A/  (IR images — grayscale)
        │   └── B/  (RGB images — color ground truth)
        └── val/
            ├── A/
            └── B/

    Images in A/ and B/ must have matching filenames.
    """

    def __init__(self, root: str, split: str = "train", img_size: int = 256, jitter_size: int = 286):
        super().__init__()
        self.split = split
        self.img_size = img_size
        self.jitter_size = jitter_size

        self.dir_ir = Path(root) / split / "A"
        self.dir_rgb = Path(root) / split / "B"

        # Get sorted filenames present in both directories
        ir_files = set(os.listdir(self.dir_ir))
        rgb_files = set(os.listdir(self.dir_rgb))
        self.filenames = sorted(ir_files & rgb_files)

        if len(self.filenames) == 0:
            raise ValueError(
                f"No matching files found in {self.dir_ir} and {self.dir_rgb}. "
                "Ensure both directories contain images with matching filenames."
            )

        print(f"[{split.upper()}] Loaded {len(self.filenames)} paired images")

    def __len__(self) -> int:
        return len(self.filenames)

    def _apply_transforms(self, ir_img: Image.Image, rgb_img: Image.Image):
        """Apply synchronized transforms to both images."""

        if self.split == "train" and config.RANDOM_JITTER:
            # Resize to slightly larger
            ir_img = TF.resize(ir_img, [self.jitter_size, self.jitter_size], interpolation=T.InterpolationMode.BICUBIC)
            rgb_img = TF.resize(rgb_img, [self.jitter_size, self.jitter_size], interpolation=T.InterpolationMode.BICUBIC)

            # Random crop to target size (synchronized)
            i, j, h, w = T.RandomCrop.get_params(ir_img, output_size=(self.img_size, self.img_size))
            ir_img = TF.crop(ir_img, i, j, h, w)
            rgb_img = TF.crop(rgb_img, i, j, h, w)

            # Random horizontal flip (synchronized)
            if config.RANDOM_FLIP and random.random() > 0.5:
                ir_img = TF.hflip(ir_img)
                rgb_img = TF.hflip(rgb_img)
        else:
            # Validation: just resize
            ir_img = TF.resize(ir_img, [self.img_size, self.img_size], interpolation=T.InterpolationMode.BICUBIC)
            rgb_img = TF.resize(rgb_img, [self.img_size, self.img_size], interpolation=T.InterpolationMode.BICUBIC)

        # Convert to tensors and normalize to [-1, 1]
        ir_tensor = TF.to_tensor(ir_img)    # (1, H, W) if grayscale
        rgb_tensor = TF.to_tensor(rgb_img)  # (3, H, W)

        # Ensure IR is single-channel
        if ir_tensor.shape[0] == 3:
            ir_tensor = ir_tensor.mean(dim=0, keepdim=True)

        # Normalize [0,1] → [-1,1]
        ir_tensor = ir_tensor * 2.0 - 1.0
        rgb_tensor = rgb_tensor * 2.0 - 1.0

        return ir_tensor, rgb_tensor

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]

        # Load images
        ir_img = Image.open(self.dir_ir / fname).convert("L")   # Force grayscale
        rgb_img = Image.open(self.dir_rgb / fname).convert("RGB")  # Force RGB

        ir_tensor, rgb_tensor = self._apply_transforms(ir_img, rgb_img)

        return ir_tensor, rgb_tensor


if __name__ == "__main__":
    # Quick test
    ds = IRColorDataset(root=config.DATA_ROOT, split="train")
    ir, rgb = ds[0]
    print(f"IR shape:  {ir.shape}, range: [{ir.min():.2f}, {ir.max():.2f}]")
    print(f"RGB shape: {rgb.shape}, range: [{rgb.min():.2f}, {rgb.max():.2f}]")

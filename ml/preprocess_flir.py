"""
SPECTRA — FLIR Dataset Preprocessing Script
Converts raw FLIR dataset into paired A/B format for pix2pix training.

Usage:
    python preprocess_flir.py --input C:/path/to/FLIR_Dataset --output data

Output structure:
    data/
    ├── train/
    │   ├── A/  (IR grayscale images)
    │   └── B/  (RGB color images)
    └── val/
        ├── A/
        └── B/
"""

import os
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import config


def find_image_pairs(flir_root: str) -> list[tuple[str, str]]:
    """
    Find paired thermal/RGB images in the FLIR dataset.

    FLIR dataset structure varies, but commonly:
        FLIR_Dataset/
        ├── train/
        │   ├── thermal_8_bit/ or PreviewData/
        │   └── RGB/
        └── val/
            ├── thermal_8_bit/ or PreviewData/
            └── RGB/
    """
    flir_root = Path(flir_root)
    pairs = []

    # Try multiple possible directory structures
    possible_structures = [
        # Structure 1: train/thermal_8_bit + train/RGB
        ("train/thermal_8_bit", "train/RGB"),
        ("val/thermal_8_bit", "val/RGB"),
        # Structure 2: train/PreviewData + train/RGB
        ("train/PreviewData", "train/RGB"),
        ("val/PreviewData", "val/RGB"),
        # Structure 3: Flat structure
        ("thermal_8_bit", "RGB"),
        ("thermal", "visible"),
        ("ir", "rgb"),
        ("IR", "RGB"),
        # Structure 4: FLIR ADAS
        ("video/thermal_8_bit", "video/RGB"),
    ]

    for ir_subdir, rgb_subdir in possible_structures:
        ir_dir = flir_root / ir_subdir
        rgb_dir = flir_root / rgb_subdir

        if ir_dir.exists() and rgb_dir.exists():
            print(f"  Found: {ir_subdir} ↔ {rgb_subdir}")

            ir_files = {f.stem: f for f in ir_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")}
            rgb_files = {f.stem: f for f in rgb_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")}

            # Match by filename stem
            common_stems = set(ir_files.keys()) & set(rgb_files.keys())
            for stem in common_stems:
                pairs.append((str(ir_files[stem]), str(rgb_files[stem])))

    if not pairs:
        # Fallback: search recursively for any thermal/RGB pair
        print("  Scanning recursively for image pairs...")
        all_files = list(flir_root.rglob("*"))
        image_files = [f for f in all_files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")]

        # Group by filename stem
        by_stem = {}
        for f in image_files:
            by_stem.setdefault(f.stem, []).append(f)

        for stem, files in by_stem.items():
            if len(files) >= 2:
                # Heuristic: smaller file is likely IR (grayscale), larger is RGB
                files.sort(key=lambda f: f.stat().st_size)
                pairs.append((str(files[0]), str(files[-1])))

    return pairs


def preprocess_and_save(pairs: list[tuple[str, str]], output_root: str, val_ratio: float = 0.1):
    """Preprocess image pairs and save in pix2pix format."""
    output_root = Path(output_root)

    # Create directories
    for split in ["train", "val"]:
        for domain in ["A", "B"]:
            (output_root / split / domain).mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    random.shuffle(pairs)
    val_count = max(1, int(len(pairs) * val_ratio))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    print(f"\n📊 Split: {len(train_pairs)} train | {len(val_pairs)} val")

    # Process each split
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        print(f"\n🔄 Processing {split_name}...")
        for idx, (ir_path, rgb_path) in enumerate(tqdm(split_pairs, desc=f"  {split_name}")):
            try:
                # Load images
                ir_img = Image.open(ir_path).convert("L")    # Force grayscale
                rgb_img = Image.open(rgb_path).convert("RGB")  # Force RGB

                # Resize to target size
                ir_img = ir_img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BICUBIC)
                rgb_img = rgb_img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BICUBIC)

                # Save with consistent naming
                filename = f"{idx:06d}.png"
                ir_img.save(output_root / split_name / "A" / filename)
                rgb_img.save(output_root / split_name / "B" / filename)

            except Exception as e:
                print(f"  ⚠️  Skipped {ir_path}: {e}")
                continue

    # Print summary
    train_a_count = len(list((output_root / "train" / "A").iterdir()))
    val_a_count = len(list((output_root / "val" / "A").iterdir()))
    print(f"\n✅ Dataset ready!")
    print(f"   Train: {train_a_count} pairs → {output_root / 'train'}")
    print(f"   Val:   {val_a_count} pairs → {output_root / 'val'}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess FLIR dataset for SPECTRA")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to downloaded FLIR dataset folder",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.DATA_ROOT,
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=config.VAL_SPLIT,
        help="Fraction of data to use for validation",
    )
    args = parser.parse_args()

    print("🔬 SPECTRA — FLIR Dataset Preprocessor")
    print(f"   Input:  {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Size:   {config.IMG_SIZE}×{config.IMG_SIZE}")
    print()

    # Find pairs
    print("🔍 Searching for paired IR/RGB images...")
    pairs = find_image_pairs(args.input)

    if not pairs:
        print("❌ No image pairs found! Check your dataset structure.")
        print("   Expected: thermal_8_bit/ + RGB/ directories")
        print("   Or any two directories with matching filenames")
        return

    print(f"✅ Found {len(pairs)} paired images")

    # Process
    preprocess_and_save(pairs, args.output, args.val_ratio)


if __name__ == "__main__":
    main()

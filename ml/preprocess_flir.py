"""
SPECTRA - FLIR ADAS v2 Dataset Preprocessing Script
Converts raw FLIR ADAS v2 dataset into paired A/B format for pix2pix training.

Usage:
    python preprocess_flir.py --input C:/Users/piyus/Downloads/FLIR_Dataset --output data

Output structure:
    data/
    +-- train/
    |   +-- A/  (IR grayscale images, 256x256)
    |   +-- B/  (RGB color images, 256x256)
    +-- val/
        +-- A/
        +-- B/
"""

import os
import json
import re
import argparse
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import config


def find_adas_v2_pairs(flir_root: str) -> dict:
    """
    Find paired thermal/RGB images in FLIR ADAS v2 dataset using index.json.

    The FLIR ADAS v2 dataset pairs thermal and RGB videos. The thermal video
    descriptions contain a JSON snippet like {"RGB": "videoId"} that maps
    each thermal video to its corresponding RGB video. Frames are matched
    by frameIndex within these paired videos.

    Filename pattern: video-{videoId}-frame-{frameIndex:06d}-{datasetFrameId}.jpg
    """
    flir_root = Path(flir_root)
    adas_root = flir_root / "FLIR_ADAS_v2"
    if not adas_root.exists():
        adas_root = flir_root

    result = {"train": [], "val": []}

    for split in ["train", "val"]:
        thermal_data = adas_root / f"images_thermal_{split}" / "data"
        rgb_data = adas_root / f"images_rgb_{split}" / "data"
        thermal_index_file = adas_root / f"images_thermal_{split}" / "index.json"
        rgb_index_file = adas_root / f"images_rgb_{split}" / "index.json"

        if not all(p.exists() for p in [thermal_data, rgb_data, thermal_index_file, rgb_index_file]):
            print(f"  [!] Missing {split} directories or index files, skipping...")
            continue

        print(f"\n  Loading {split} index files...")

        with open(thermal_index_file, "r", encoding="utf-8") as f:
            th_idx = json.load(f)
        with open(rgb_index_file, "r", encoding="utf-8") as f:
            rgb_idx = json.load(f)

        # Step 1: Extract thermal_videoId -> rgb_videoId mapping from descriptions
        th_to_rgb_vid = {}
        for v in th_idx.get("videos", []):
            desc = v.get("description", "")
            m = re.search(r'"RGB":\s*"(\w+)"', desc)
            if m:
                th_to_rgb_vid[v["id"]] = m.group(1)

        print(f"    Video mappings: {len(th_to_rgb_vid)}/{len(th_idx.get('videos', []))} thermal videos mapped to RGB")

        # Step 2: Build RGB frame lookup: (rgb_videoId, frameIndex) -> datasetFrameId
        rgb_by_vid_frame = {}
        for fr in rgb_idx.get("frames", []):
            vm = fr["videoMetadata"]
            key = (vm["videoId"], vm["frameIndex"])
            rgb_by_vid_frame[key] = fr["datasetFrameId"]

        # Step 3: For each thermal frame, find the matching RGB frame
        matched = 0
        for fr in th_idx.get("frames", []):
            vm = fr["videoMetadata"]
            th_vid_id = vm["videoId"]
            frame_idx = vm["frameIndex"]
            th_ds_frame_id = fr["datasetFrameId"]

            rgb_vid_id = th_to_rgb_vid.get(th_vid_id)
            if not rgb_vid_id:
                continue

            rgb_ds_frame_id = rgb_by_vid_frame.get((rgb_vid_id, frame_idx))
            if not rgb_ds_frame_id:
                continue

            # Construct filenames
            th_filename = f"video-{th_vid_id}-frame-{frame_idx:06d}-{th_ds_frame_id}.jpg"
            rgb_filename = f"video-{rgb_vid_id}-frame-{frame_idx:06d}-{rgb_ds_frame_id}.jpg"

            th_path = thermal_data / th_filename
            rgb_path = rgb_data / rgb_filename

            if th_path.exists() and rgb_path.exists():
                result[split].append((str(th_path), str(rgb_path)))
                matched += 1

        print(f"    Matched {matched} IR/RGB pairs for {split}")

    return result


def preprocess_and_save(split_pairs: dict, output_root: str):
    """Preprocess image pairs and save in pix2pix A/B format."""
    output_root = Path(output_root)

    # Create directories
    for split in ["train", "val"]:
        for domain in ["A", "B"]:
            (output_root / split / domain).mkdir(parents=True, exist_ok=True)

    total_saved = {"train": 0, "val": 0}

    for split_name, pairs in split_pairs.items():
        if not pairs:
            print(f"\n  [!] No pairs for {split_name}, skipping")
            continue

        random.shuffle(pairs)
        print(f"\n  Processing {split_name} ({len(pairs)} pairs)...")

        for idx, (ir_path, rgb_path) in enumerate(
            tqdm(pairs, desc=f"    {split_name}", unit="img")
        ):
            try:
                ir_img = Image.open(ir_path).convert("L")
                rgb_img = Image.open(rgb_path).convert("RGB")

                ir_img = ir_img.resize(
                    (config.IMG_SIZE, config.IMG_SIZE), Image.BICUBIC
                )
                rgb_img = rgb_img.resize(
                    (config.IMG_SIZE, config.IMG_SIZE), Image.BICUBIC
                )

                filename = f"{idx:06d}.png"
                ir_img.save(output_root / split_name / "A" / filename)
                rgb_img.save(output_root / split_name / "B" / filename)
                total_saved[split_name] += 1

            except Exception as e:
                print(f"\n  [!] Skipped {Path(ir_path).name}: {e}")
                continue

    print(f"\n{'='*50}")
    print(f"DONE! Dataset ready.")
    print(f"   Train: {total_saved['train']} pairs -> {output_root / 'train'}")
    print(f"   Val:   {total_saved['val']} pairs -> {output_root / 'val'}")
    print(f"   Size:  {config.IMG_SIZE}x{config.IMG_SIZE} PNG")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess FLIR ADAS v2 dataset for SPECTRA pix2pix training"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to downloaded FLIR dataset folder (containing FLIR_ADAS_v2/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.DATA_ROOT,
        help="Output directory for processed dataset (default: data)",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("SPECTRA - FLIR ADAS v2 Preprocessor")
    print("=" * 50)
    print(f"   Input:  {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Size:   {config.IMG_SIZE}x{config.IMG_SIZE}")
    print()

    print("Searching for paired IR/RGB images...")
    split_pairs = find_adas_v2_pairs(args.input)

    total = sum(len(v) for v in split_pairs.values())
    if total == 0:
        print("\nERROR: No image pairs found! Check your dataset structure.")
        print("   Expected: FLIR_Dataset/FLIR_ADAS_v2/images_thermal_train/data/")
        return

    print(f"\nTotal matched: {total} pairs")
    print(f"   Train: {len(split_pairs['train'])}")
    print(f"   Val:   {len(split_pairs['val'])}")

    preprocess_and_save(split_pairs, args.output)


if __name__ == "__main__":
    main()

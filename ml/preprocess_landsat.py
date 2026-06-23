"""
SPECTRA — Landsat 9 Preprocessing Pipeline

Follows the exact ISRO Workflow Summary (PS10 Explainer Session, Slide 6):
    1. Read raw GeoTIFF band files (B2, B3, B4, B10)
    2. Merge B2+B3+B4 into RGB composite
    3. Downscale RGB by 3.3x → 100m resolution
    4. Downscale B10 (TIR) by 3.3x → 100m ground truth TIR
    5. Downscale B10 (TIR) by 6.7x → 200m low-res input TIR
    6. Extract 256x256 patches from all three versions
    7. Save in paired A/B format for training

Usage:
    python preprocess_landsat.py --input data/raw --output data

Input structure (from Google Earth Engine or USGS EarthExplorer):
    data/raw/
    ├── mumbai_urban_B2.tif
    ├── mumbai_urban_B3.tif
    ├── mumbai_urban_B4.tif
    ├── mumbai_urban_B10.tif
    ├── delhi_urban_B2.tif
    ...

    OR (USGS EarthExplorer format):
    data/raw/
    ├── LC09_..._B2.TIF
    ├── LC09_..._B3.TIF
    ├── LC09_..._B4.TIF
    ├── LC09_..._B10.TIF
    ...

Output structure:
    data/
    ├── sr/                         # For Model A (Super-Resolution)
    │   ├── train/
    │   │   ├── A/  (200m TIR patches — low-res input)
    │   │   └── B/  (100m TIR patches — high-res ground truth)
    │   └── val/
    │       ├── A/
    │       └── B/
    └── color/                      # For Model B (Colorization)
        ├── train/
        │   ├── A/  (100m TIR patches — grayscale input)
        │   └── B/  (100m RGB patches — color ground truth)
        └── val/
            ├── A/
            └── B/
"""

import os
import re
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

import config


def read_geotiff(filepath):
    """
    Read a GeoTIFF file and return as a numpy array.
    Tries rasterio first, falls back to tifffile, then PIL.
    """
    filepath = str(filepath)
    
    if HAS_RASTERIO:
        with rasterio.open(filepath) as src:
            data = src.read(1).astype(np.float32)
            return data
    elif HAS_TIFFFILE:
        data = tifffile.imread(filepath).astype(np.float32)
        if data.ndim == 3:
            data = data[0]  # Take first band if multi-band
        return data
    else:
        # Fallback to PIL (won't handle all GeoTIFF features but works for basic ones)
        img = Image.open(filepath)
        return np.array(img, dtype=np.float32)


def normalize_band(data, percentile_clip=2):
    """
    Normalize band data to [0, 1] using percentile clipping.
    This removes outliers and produces visually clean images.
    """
    if data.size == 0:
        return data
    
    low = np.percentile(data, percentile_clip)
    high = np.percentile(data, 100 - percentile_clip)
    
    if high <= low:
        return np.zeros_like(data)
    
    normalized = (data - low) / (high - low)
    return np.clip(normalized, 0, 1)


def find_band_groups(input_dir):
    """
    Scan input directory and group files by region/scene.
    
    Supports two naming conventions:
    1. GEE format: {region}_B2.tif, {region}_B3.tif, etc.
    2. USGS format: LC09_..._B2.TIF, LC09_..._B3.TIF, etc.
    
    Returns dict: { region_name: { 'B2': path, 'B3': path, 'B4': path, 'B10': path } }
    """
    input_dir = Path(input_dir)
    groups = defaultdict(dict)
    
    for f in sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.TIF")):
        name = f.stem
        
        # Try GEE format: {region}_{band}
        gee_match = re.match(r'^(.+)_(B\d+)$', name, re.IGNORECASE)
        if gee_match:
            region = gee_match.group(1)
            band = gee_match.group(2).upper()
            groups[region][band] = f
            continue
        
        # Try USGS format: LC09_..._B{N}
        usgs_match = re.search(r'(B\d+)', name, re.IGNORECASE)
        if usgs_match:
            band = usgs_match.group(1).upper()
            # Use everything before the band as region name
            region = name[:usgs_match.start()].rstrip('_')
            if not region:
                region = "scene"
            groups[region][band] = f
            continue
    
    # Filter to only groups that have all 4 required bands
    complete_groups = {}
    for region, bands in groups.items():
        required = {'B2', 'B3', 'B4', 'B10'}
        if required.issubset(bands.keys()):
            complete_groups[region] = bands
        else:
            missing = required - bands.keys()
            print(f"  ⚠️  Region '{region}' missing bands: {missing}, skipping")
    
    return complete_groups


def merge_rgb(b2_data, b3_data, b4_data):
    """
    Merge Landsat 9 bands B2 (Blue), B3 (Green), B4 (Red) into an RGB image.
    
    Note: Landsat band order is B2=Blue, B3=Green, B4=Red
    So RGB = [B4, B3, B2] (Red, Green, Blue)
    """
    r = normalize_band(b4_data)  # Red
    g = normalize_band(b3_data)  # Green
    b = normalize_band(b2_data)  # Blue
    
    # Stack into (H, W, 3) RGB image
    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def downscale_image(image, factor, method='bicubic'):
    """
    Downscale an image by the given factor.
    
    Args:
        image: numpy array (H, W) or (H, W, C)
        factor: downscale factor (e.g., 3.3 means new_size = old_size / 3.3)
        method: interpolation method
    """
    h, w = image.shape[:2]
    new_h = max(1, int(h / factor))
    new_w = max(1, int(w / factor))
    
    if image.ndim == 2:
        pil_img = Image.fromarray(image, mode='F')
        resized = pil_img.resize((new_w, new_h), Image.BICUBIC)
        return np.array(resized, dtype=np.float32)
    else:
        # Multi-channel (RGB)
        pil_img = Image.fromarray((image * 255).astype(np.uint8), mode='RGB')
        resized = pil_img.resize((new_w, new_h), Image.BICUBIC)
        return np.array(resized, dtype=np.float32) / 255.0


def extract_patches(image, patch_size, stride):
    """
    Extract overlapping patches from an image.
    
    Args:
        image: numpy array (H, W) or (H, W, C)
        patch_size: size of each square patch
        stride: step between patches
    
    Returns:
        list of numpy patches
    """
    h, w = image.shape[:2]
    patches = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            if image.ndim == 2:
                patch = image[y:y+patch_size, x:x+patch_size]
            else:
                patch = image[y:y+patch_size, x:x+patch_size, :]
            
            # Skip patches that are mostly empty/black
            if np.mean(patch) < 0.01:
                continue
            
            patches.append(patch)
    
    return patches


def save_patch_as_png(patch, filepath):
    """Save a float32 patch [0,1] as PNG."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if patch.ndim == 2:
        # Grayscale
        img = Image.fromarray((patch * 255).astype(np.uint8), mode='L')
    else:
        # RGB
        img = Image.fromarray((patch * 255).astype(np.uint8), mode='RGB')
    
    img.save(str(filepath))


def process_region(region_name, band_paths, output_root, patch_idx_start=0):
    """
    Process a single region's band files into training patches.
    
    Following the exact ISRO workflow:
    1. Read B2, B3, B4, B10
    2. Merge B2+B3+B4 into RGB
    3. Downscale RGB by 3.3x → 100m RGB
    4. Downscale B10 by 3.3x → 100m TIR (ground truth for Model A)
    5. Downscale B10 by 6.7x → 200m TIR (input for Model A)
    6. Extract patches from all three
    7. Save as paired A/B images
    """
    print(f"\n  📂 Processing region: {region_name}")
    
    # Step 1: Read bands
    print(f"     Reading B2 (Blue)...")
    b2 = read_geotiff(band_paths['B2'])
    print(f"     Reading B3 (Green)...")
    b3 = read_geotiff(band_paths['B3'])
    print(f"     Reading B4 (Red)...")
    b4 = read_geotiff(band_paths['B4'])
    print(f"     Reading B10 (Thermal IR)...")
    b10 = read_geotiff(band_paths['B10'])
    
    print(f"     Band shapes: B2={b2.shape}, B3={b3.shape}, B4={b4.shape}, B10={b10.shape}")
    
    # Ensure all optical bands have the same shape
    min_h = min(b2.shape[0], b3.shape[0], b4.shape[0])
    min_w = min(b2.shape[1], b3.shape[1], b4.shape[1])
    b2 = b2[:min_h, :min_w]
    b3 = b3[:min_h, :min_w]
    b4 = b4[:min_h, :min_w]
    
    # Resize B10 to match optical bands (it may be different resolution in L2 products)
    if b10.shape != (min_h, min_w):
        b10_pil = Image.fromarray(b10, mode='F')
        b10_pil = b10_pil.resize((min_w, min_h), Image.BICUBIC)
        b10 = np.array(b10_pil, dtype=np.float32)
    
    # Step 2: Merge RGB
    print(f"     Merging B2+B3+B4 into RGB...")
    rgb_30m = merge_rgb(b2, b3, b4)
    
    # Normalize TIR
    tir_30m = normalize_band(b10)
    
    # Step 3: Downscale RGB by 3.3x → 100m
    print(f"     Downscaling RGB by {config.DOWNSCALE_100M}x → 100m...")
    rgb_100m = downscale_image(rgb_30m, config.DOWNSCALE_100M)
    
    # Step 4: Downscale TIR by 3.3x → 100m (ground truth)
    print(f"     Downscaling TIR by {config.DOWNSCALE_100M}x → 100m (ground truth)...")
    tir_100m = downscale_image(tir_30m, config.DOWNSCALE_100M)
    
    # Step 5: Downscale TIR by 6.7x → 200m (low-res input)
    print(f"     Downscaling TIR by {config.DOWNSCALE_200M}x → 200m (input)...")
    tir_200m = downscale_image(tir_30m, config.DOWNSCALE_200M)
    
    print(f"     Sizes: RGB@100m={rgb_100m.shape}, TIR@100m={tir_100m.shape}, TIR@200m={tir_200m.shape}")
    
    # Step 6: Extract patches
    patch_size = config.PATCH_SIZE
    stride = config.PATCH_STRIDE
    
    # For SR model: need 200m and 100m TIR patches
    # The 200m patches are smaller than 100m patches, so we need to resize them
    # Or we extract patches from the same spatial locations
    
    # Method: Extract patches from 100m versions, then for each patch location,
    # extract the corresponding region from 200m (which will be smaller, then we resize to match)
    
    print(f"     Extracting {patch_size}x{patch_size} patches (stride={stride})...")
    
    # Upscale 200m TIR back to 100m size so all images match spatial dimensions
    tir_200m_upscaled = Image.fromarray(tir_200m, mode='F')
    tir_200m_upscaled = tir_200m_upscaled.resize(
        (rgb_100m.shape[1], rgb_100m.shape[0]), Image.BICUBIC
    )
    tir_200m_upscaled = np.array(tir_200m_upscaled, dtype=np.float32)

    # Now all three arrays are precisely the same size. Extract patches from them.
    # We will loop explicitly so the patches are perfectly aligned.
    tir_100m_patches = []
    rgb_100m_patches = []
    tir_200m_patches = []

    h, w = tir_100m.shape[:2]
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            p_tir100 = tir_100m[y:y+patch_size, x:x+patch_size]
            p_rgb100 = rgb_100m[y:y+patch_size, x:x+patch_size, :]
            p_tir200 = tir_200m_upscaled[y:y+patch_size, x:x+patch_size]
            
            # Skip empty/black patches
            if np.mean(p_tir100) < 0.01:
                continue
                
            tir_100m_patches.append(p_tir100)
            rgb_100m_patches.append(p_rgb100)
            tir_200m_patches.append(p_tir200)
            
    n_patches = len(tir_100m_patches)
    
    print(f"     Extracted {n_patches} patch triplets (200m TIR, 100m TIR, 100m RGB)")
    
    if n_patches == 0:
        print(f"     ⚠️  No patches extracted! Image may be too small for patch_size={patch_size}")
        return patch_idx_start
    
    # Step 7: Split into train/val
    indices = list(range(n_patches))
    random.shuffle(indices)
    
    n_val = max(1, int(n_patches * config.VAL_SPLIT))
    n_train = n_patches - n_val
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    print(f"     Split: {n_train} train + {n_val} val")
    
    # Step 8: Save patches
    output_root = Path(output_root)
    idx = patch_idx_start
    
    for split_name, split_indices in [("train", train_indices), ("val", val_indices)]:
        for i in split_indices:
            fname = f"{idx:06d}.png"
            
            # Super-Resolution pairs: A=200m TIR, B=100m TIR
            save_patch_as_png(tir_200m_patches[i], output_root / "sr" / split_name / "A" / fname)
            save_patch_as_png(tir_100m_patches[i], output_root / "sr" / split_name / "B" / fname)
            
            # Colorization pairs: A=100m TIR, B=100m RGB
            save_patch_as_png(tir_100m_patches[i], output_root / "color" / split_name / "A" / fname)
            save_patch_as_png(rgb_100m_patches[i], output_root / "color" / split_name / "B" / fname)
            
            idx += 1
    
    print(f"     ✅ Saved {idx - patch_idx_start} patch sets")
    return idx


def main():
    parser = argparse.ArgumentParser(
        description="SPECTRA — Landsat 9 Preprocessing Pipeline (ISRO PS10)"
    )
    parser.add_argument(
        "--input", type=str, default=config.LANDSAT_RAW_DIR,
        help="Directory containing raw GeoTIFF band files"
    )
    parser.add_argument(
        "--output", type=str, default=config.DATA_ROOT,
        help="Output directory for processed patches"
    )
    parser.add_argument(
        "--patch-size", type=int, default=128,
        help="Patch size in pixels (default: 128)"
    )
    parser.add_argument(
        "--stride", type=int, default=64,
        help="Stride between patches (default: 64)"
    )
    args = parser.parse_args()
    
    # Override config with CLI args
    config.PATCH_SIZE = args.patch_size
    config.PATCH_STRIDE = args.stride
    
    print("=" * 60)
    print("SPECTRA — Landsat 9 Preprocessing Pipeline")
    print("Following ISRO PS10 Workflow Summary")
    print("=" * 60)
    print(f"   Input:      {args.input}")
    print(f"   Output:     {args.output}")
    print(f"   Patch size: {args.patch_size}x{args.patch_size}")
    print(f"   Stride:     {args.stride}")
    print(f"   Downscale:  {config.DOWNSCALE_100M}x (100m), {config.DOWNSCALE_200M}x (200m)")
    
    # Check input directory
    if not Path(args.input).exists():
        print(f"\n❌ ERROR: Input directory does not exist: {args.input}")
        print(f"   Download data first with: python download_landsat.py")
        return
    
    # Find band groups
    print(f"\n🔍 Scanning for GeoTIFF band files...")
    band_groups = find_band_groups(args.input)
    
    if not band_groups:
        print(f"\n❌ ERROR: No complete band groups found in {args.input}")
        print(f"   Each region needs: _B2.tif, _B3.tif, _B4.tif, _B10.tif")
        return
    
    print(f"   Found {len(band_groups)} complete region(s):")
    for name, bands in band_groups.items():
        print(f"     • {name}: {', '.join(sorted(bands.keys()))}")
    
    # Process each region
    total_patches = 0
    for region_name, band_paths in band_groups.items():
        total_patches = process_region(region_name, band_paths, args.output, total_patches)
    
    # Summary
    output = Path(args.output)
    print(f"\n{'='*60}")
    print(f"✅ PREPROCESSING COMPLETE!")
    print(f"   Total patches: {total_patches}")
    print(f"\n   Super-Resolution data (Model A):")
    print(f"     {output / 'sr' / 'train' / 'A'}/ → 200m TIR (input)")
    print(f"     {output / 'sr' / 'train' / 'B'}/ → 100m TIR (ground truth)")
    print(f"\n   Colorization data (Model B):")
    print(f"     {output / 'color' / 'train' / 'A'}/ → 100m TIR (input)")
    print(f"     {output / 'color' / 'train' / 'B'}/ → 100m RGB (ground truth)")
    print(f"\n   Next step: python train.py --mode sr      (train Model A)")
    print(f"              python train.py --mode colorize (train Model B)")
    print(f"              python train.py --mode e2e      (train end-to-end)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""
SPECTRA ML — Hyperparameter Configuration
Two-Stage Pipeline: Super-Resolution + Colorization
Dataset: Landsat 9 (USGS) — Bands B2, B3, B4, B10
"""
import os

# ─── Dataset & Paths ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")
LANDSAT_RAW_DIR = os.path.join(DATA_ROOT, "raw")
SR_DATA_ROOT = os.path.join(DATA_ROOT, "sr")
COLOR_DATA_ROOT = os.path.join(DATA_ROOT, "color")

# ─── Image Settings ─────────────────────────────────────────
IMG_SIZE = 256                         # All patches resized/cropped to 256x256
JITTER_SIZE = 286                      # Resize to this before random crop (augmentation)
PATCH_SIZE = 256                       # Patch extraction size from full scenes
PATCH_STRIDE = 128                     # Stride for overlapping patches (50% overlap)

# ─── Landsat 9 Downscaling (per ISRO Workflow) ──────────────
# Landsat 9 optical bands (B2,B3,B4) are 30m native resolution
# Landsat 9 TIR band (B10) is 100m native resolution (resampled to 30m in L2 products)
# ISRO workflow:
#   - Downscale RGB by 3.3x from 30m → ~100m
#   - Downscale TIR (B10) by 3.3x from 30m → ~100m (ground truth)
#   - Downscale TIR (B10) by 6.7x from 30m → ~200m (low-res input)
DOWNSCALE_100M = 3.3                   # Factor to simulate 100m from 30m
DOWNSCALE_200M = 6.7                   # Factor to simulate 200m from 30m

# ─── Google Earth Engine Download Regions ────────────────────
# Each region: (name, center_lat, center_lon, size_km)
# Size is set to 150km x 150km (~full Landsat scene size) for massive dataset generation
GEE_REGIONS = [
    ("mumbai_urban", 19.07, 72.87, 150),       # Urban + Coast
    ("delhi_urban", 28.61, 77.20, 150),         # Dense Urban
    ("gujarat_rural", 22.30, 72.63, 150),       # Rural / agriculture
    ("chilika_water", 19.72, 85.32, 150),       # Water body (Chilika Lake)
    ("bangalore_mixed", 12.97, 77.59, 150),     # Mixed urban/vegetation
    ("himalayas_snow", 32.24, 77.18, 150),      # Snow/Mountains (Manali region)
    ("thar_desert", 26.91, 70.90, 150),         # Desert (Jaisalmer region)
    ("sundarbans_delta", 21.94, 88.89, 150),    # Delta/Mangroves
]
GEE_DATE_START = "2024-01-01"
GEE_DATE_END = "2024-12-31"
GEE_CLOUD_COVER_MAX = 10              # Max cloud cover percentage
GEE_EXPORT_SCALE = 30                 # Export at 30m/pixel (native optical resolution)
GEE_DRIVE_FOLDER = "SPECTRA_Landsat9" # Google Drive export folder

# ─── Model Architecture ────────────────────────────────────
INPUT_CHANNELS = 1                    # Grayscale TIR input (single channel)
OUTPUT_CHANNELS = 3                   # RGB output
GENERATOR_FILTERS = 64                # Base number of filters in generator
DISCRIMINATOR_FILTERS = 64            # Base number of filters in discriminator
SR_UPSCALE_FACTOR = 2                 # Super-resolution upscale factor

# ─── Training ──────────────────────────────────────────────
EPOCHS = 200
BATCH_SIZE = 4                        # Batch size (increase for GPU with more VRAM)
LEARNING_RATE = 0.0002
BETA1 = 0.5                          # Adam beta1
BETA2 = 0.999                        # Adam beta2
LAMBDA_L1 = 100                      # Weight for L1 loss (pixel reconstruction)
LAMBDA_GAN = 1                       # Weight for adversarial loss
LAMBDA_SR = 10                       # Weight for SR L1 loss in end-to-end mode
LAMBDA_PHYSICS = 0.1                 # Weight for physics-informed loss (bonus)

# ─── Augmentation ──────────────────────────────────────────
RANDOM_JITTER = True                  # Resize + random crop augmentation
RANDOM_FLIP = True                    # Random horizontal flip

# ─── Training Checkpoints & Output ──────────────────────────
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
ONNX_OUTPUT = os.path.join(BASE_DIR, "spectrasatnet.onnx")
ONNX_OPSET = 14                     
SAVE_EVERY = 20                       # Save checkpoint every N epochs
SAMPLE_DIR = "samples"                # Directory for sample outputs during training

# ─── Evaluation ────────────────────────────────────────────
VAL_SPLIT = 0.1                       # 10% of data for validation
TEST_SPLIT = 0.1                      # 10% of data for testing

# ─── ONNX Export ───────────────────────────────────────────
ONNX_OPSET = 17                       # ONNX opset version
ONNX_OUTPUT = "spectra_generator.onnx"

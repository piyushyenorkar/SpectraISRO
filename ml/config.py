"""
SPECTRA ML — Hyperparameter Configuration
pix2pix GAN for Infrared Image Colorization
"""

# ─── Dataset ────────────────────────────────────────────────
DATA_ROOT = "data"                # Root folder for train/val data
IMG_SIZE = 256                    # All images resized to 256x256
JITTER_SIZE = 286                 # Resize to this before random crop (augmentation)

# ─── Model Architecture ────────────────────────────────────
INPUT_CHANNELS = 1                # Grayscale IR input (single channel)
OUTPUT_CHANNELS = 3               # RGB output
GENERATOR_FILTERS = 64            # Base number of filters in generator
DISCRIMINATOR_FILTERS = 64        # Base number of filters in discriminator

# ─── Training ──────────────────────────────────────────────
EPOCHS = 200
BATCH_SIZE = 1                    # pix2pix paper recommends batch_size=1
LEARNING_RATE = 0.0002
BETA1 = 0.5                      # Adam beta1
BETA2 = 0.999                    # Adam beta2
LAMBDA_L1 = 100                  # Weight for L1 loss (pixel reconstruction)
LAMBDA_GAN = 1                   # Weight for adversarial loss

# ─── Augmentation ──────────────────────────────────────────
RANDOM_JITTER = True              # Resize + random crop augmentation
RANDOM_FLIP = True                # Random horizontal flip

# ─── Checkpointing ─────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY = 20                   # Save checkpoint every N epochs
SAMPLE_DIR = "samples"            # Directory for sample outputs during training

# ─── Evaluation ────────────────────────────────────────────
VAL_SPLIT = 0.1                   # 10% of data for validation

# ─── ONNX Export ───────────────────────────────────────────
ONNX_OPSET = 17                   # ONNX opset version
ONNX_OUTPUT = "spectra_generator.onnx"

"""
SPECTRA — Image Preprocessing Utilities
"""

import io
from PIL import Image
import torch
import torchvision.transforms.functional as TF


IMG_SIZE = 256


def load_and_preprocess(file_bytes: bytes) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Load uploaded image and preprocess for model inference.

    Args:
        file_bytes: Raw image bytes from upload

    Returns:
        tensor: (1, 1, 256, 256) normalized to [-1, 1]
        original_size: (width, height) of original image
    """
    # Load image
    img = Image.open(io.BytesIO(file_bytes))
    original_size = img.size  # (width, height)

    # Convert to grayscale
    img = img.convert("L")

    # Resize to model input size
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)

    # Convert to tensor: (1, H, W) in [0, 1]
    tensor = TF.to_tensor(img)

    # Normalize to [-1, 1]
    tensor = tensor * 2.0 - 1.0

    # Add batch dimension: (1, 1, H, W)
    tensor = tensor.unsqueeze(0)

    return tensor, original_size


def validate_image(file_bytes: bytes, max_size_mb: float = 10.0) -> tuple[bool, str]:
    """Validate uploaded image file."""
    # Check size
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f}MB (max {max_size_mb}MB)"

    # Check if valid image
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
    except Exception:
        return False, "Invalid image file"

    return True, "OK"

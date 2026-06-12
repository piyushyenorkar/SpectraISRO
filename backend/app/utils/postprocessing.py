"""
SPECTRA — Image Postprocessing Utilities
"""

import io
import base64

import torch
from PIL import Image
import numpy as np


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert model output tensor to PIL Image.

    Args:
        tensor: (1, 3, H, W) in [-1, 1]

    Returns:
        PIL Image in RGB mode
    """
    # Remove batch dim and move to CPU
    img = tensor.squeeze(0).detach().cpu()

    # Denormalize: [-1, 1] → [0, 1]
    img = (img + 1.0) / 2.0
    img = img.clamp(0, 1)

    # Convert to numpy: (C, H, W) → (H, W, C)
    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)

    return Image.fromarray(img_np, mode="RGB")


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64-encoded data URI."""
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    mime = f"image/{format.lower()}"
    return f"data:{mime};base64,{encoded}"


def tensor_to_base64(tensor: torch.Tensor, format: str = "PNG") -> str:
    """Convert model output tensor directly to base64 string."""
    pil_img = tensor_to_pil(tensor)
    return pil_to_base64(pil_img, format)

"""
SPECTRA — Evaluation Metrics (PSNR & SSIM)
"""

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a [-1,1] tensor to [0,1] numpy array (H, W, C)."""
    img = tensor.detach().cpu()
    img = (img + 1.0) / 2.0  # [-1,1] → [0,1]
    img = img.clamp(0, 1)
    img = img.squeeze(0)  # Remove batch dim
    img = img.permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
    return img


def compute_psnr_ssim(generated: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """
    Compute PSNR and SSIM between generated and target images.

    Args:
        generated: (1, 3, H, W) tensor in [-1, 1]
        target:    (1, 3, H, W) tensor in [-1, 1]

    Returns:
        (psnr, ssim) as floats
    """
    gen_np = tensor_to_numpy(generated)
    tgt_np = tensor_to_numpy(target)

    psnr = peak_signal_noise_ratio(tgt_np, gen_np, data_range=1.0)
    ssim = structural_similarity(tgt_np, gen_np, data_range=1.0, channel_axis=2)

    return psnr, ssim


if __name__ == "__main__":
    # Quick test with random tensors
    fake = torch.randn(1, 3, 256, 256).clamp(-1, 1)
    real = torch.randn(1, 3, 256, 256).clamp(-1, 1)
    psnr, ssim = compute_psnr_ssim(fake, real)
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")

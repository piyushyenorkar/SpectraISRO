"""
SPECTRA — Image Quality Metrics Service
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(generated: np.ndarray, reference: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    return float(peak_signal_noise_ratio(reference, generated, data_range=1.0))


def compute_ssim(generated: np.ndarray, reference: np.ndarray) -> float:
    """Compute Structural Similarity Index."""
    return float(structural_similarity(reference, generated, data_range=1.0, channel_axis=2))


def compute_enhancement_metrics(ir_array: np.ndarray, colorized_array: np.ndarray) -> dict:
    """
    Compute enhancement metrics comparing IR input vs colorized output.
    Since we don't have ground truth RGB at inference, we compute:
    - Dynamic range expansion (how much richer the color output is)
    - Information content increase (entropy comparison)
    """
    from scipy.stats import entropy as scipy_entropy

    # Dynamic range
    ir_range = float(ir_array.max() - ir_array.min())
    color_range = float(colorized_array.max() - colorized_array.min())

    # Information entropy (higher = more information)
    ir_flat = (ir_array.flatten() * 255).astype(np.uint8)
    color_flat = (colorized_array.mean(axis=2).flatten() * 255).astype(np.uint8)

    ir_hist = np.histogram(ir_flat, bins=256, range=(0, 255))[0]
    color_hist = np.histogram(color_flat, bins=256, range=(0, 255))[0]

    ir_entropy = float(scipy_entropy(ir_hist + 1e-10))
    color_entropy = float(scipy_entropy(color_hist + 1e-10))

    return {
        "dynamic_range_ir": round(ir_range, 4),
        "dynamic_range_color": round(color_range, 4),
        "entropy_ir": round(ir_entropy, 4),
        "entropy_color": round(color_entropy, 4),
        "information_gain": round(color_entropy - ir_entropy, 4),
    }

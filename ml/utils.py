"""
SPECTRA — Training Utilities
"""

import os
from pathlib import Path

import torch
from torchvision.utils import save_image


def save_checkpoint(generator, discriminator, opt_g, opt_d, epoch, path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": opt_g.state_dict(),
        "optimizer_d_state_dict": opt_d.state_dict(),
    }, path)
    print(f"  💾 Checkpoint saved: {path}")


def load_checkpoint(path, generator, discriminator, opt_g, opt_d, device):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    opt_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    opt_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
    return checkpoint["epoch"]


def save_sample_images(ir, real_rgb, fake_rgb, epoch, output_dir="samples"):
    """Save side-by-side comparison: IR | Generated | Real."""
    os.makedirs(output_dir, exist_ok=True)

    # Denormalize from [-1,1] to [0,1]
    ir_vis = ir.repeat(1, 3, 1, 1)  # Convert single-channel to 3-channel for visualization
    ir_vis = (ir_vis + 1) / 2.0
    real_vis = (real_rgb + 1) / 2.0
    fake_vis = (fake_rgb + 1) / 2.0

    # Stack horizontally: IR | Generated | Real
    comparison = torch.cat([ir_vis, fake_vis, real_vis], dim=3)
    save_image(comparison, os.path.join(output_dir, f"epoch_{epoch:04d}.png"), nrow=1)


def denormalize(tensor):
    """Convert tensor from [-1,1] to [0,1]."""
    return (tensor + 1.0) / 2.0

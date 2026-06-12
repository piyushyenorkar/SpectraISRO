"""
SPECTRA — PatchGAN Discriminator for pix2pix GAN
Classifies 70×70 overlapping patches as real or fake.

Input:  concatenated [IR_image, RGB_image] = (batch, 4, 256, 256)
Output: (batch, 1, 30, 30) — each value = real/fake for that patch
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator with 70×70 receptive field.

    Architecture:
        c1: 4   → 64   (128×128) — no BatchNorm
        c2: 64  → 128  (64×64)
        c3: 128 → 256  (32×32)
        c4: 256 → 512  (31×31)  — stride=1 padding
        out: 512 → 1   (30×30)  — real/fake map
    """

    def __init__(self, in_channels: int = 4, base_filters: int = 64):
        super().__init__()
        f = base_filters

        self.model = nn.Sequential(
            # Block 1: no normalization (per pix2pix paper)
            nn.Conv2d(in_channels, f, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2
            nn.Conv2d(f, f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3
            nn.Conv2d(f * 2, f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: stride=1 for finer spatial output
            nn.Conv2d(f * 4, f * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output: 1-channel patch classification map
            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, ir_image: torch.Tensor, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ir_image:  (B, 1, 256, 256) grayscale IR input
            rgb_image: (B, 3, 256, 256) RGB image (real or generated)

        Returns:
            (B, 1, 30, 30) patch-level real/fake predictions
        """
        # Concatenate IR and RGB along channel dimension → (B, 4, 256, 256)
        x = torch.cat([ir_image, rgb_image], dim=1)
        return self.model(x)


if __name__ == "__main__":
    # Quick test
    disc = PatchGANDiscriminator(in_channels=4)
    ir = torch.randn(1, 1, 256, 256)
    rgb = torch.randn(1, 3, 256, 256)
    out = disc(ir, rgb)
    print(f"IR input:   {ir.shape}")
    print(f"RGB input:  {rgb.shape}")
    print(f"Disc output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in disc.parameters()):,}")

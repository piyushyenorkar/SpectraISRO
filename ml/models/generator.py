"""
SPECTRA — U-Net Generator for pix2pix GAN
Translates grayscale IR images → RGB colorized images.

Architecture: 8-block U-Net encoder-decoder with skip connections.
Input:  (batch, 1, 256, 256)  — grayscale IR
Output: (batch, 3, 256, 256)  — RGB colorized
"""

import torch
import torch.nn as nn


class UNetDownBlock(nn.Module):
    """Encoder block: Conv2d → [BatchNorm] → LeakyReLU"""

    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    """Decoder block: ConvTranspose2d → BatchNorm → [Dropout] → ReLU"""

    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """
    U-Net Generator for pix2pix.

    Encoder:
        e1: 1   → 64   (128×128) — no BatchNorm on first layer
        e2: 64  → 128  (64×64)
        e3: 128 → 256  (32×32)
        e4: 256 → 512  (16×16)
        e5: 512 → 512  (8×8)
        e6: 512 → 512  (4×4)
        e7: 512 → 512  (2×2)

    Bottleneck:
        b:  512 → 512  (1×1)

    Decoder (with skip connections):
        d7: 512       → 512  (2×2)   + skip e7 → 1024
        d6: 1024      → 512  (4×4)   + skip e6 → 1024
        d5: 1024      → 512  (8×8)   + skip e5 → 1024
        d4: 1024      → 512  (16×16) + skip e4 → 1024
        d3: 1024      → 256  (32×32) + skip e3 → 512
        d2: 512       → 128  (64×64) + skip e2 → 256
        d1: 256       → 64   (128×128)+ skip e1 → 128

    Output:
        out: 128 → 3   (256×256) + Tanh
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3, base_filters: int = 64):
        super().__init__()
        f = base_filters  # 64

        # ─── Encoder ───────────────────────────────────────
        self.e1 = UNetDownBlock(in_channels, f, use_norm=False)     # 1→64
        self.e2 = UNetDownBlock(f, f * 2)                           # 64→128
        self.e3 = UNetDownBlock(f * 2, f * 4)                       # 128→256
        self.e4 = UNetDownBlock(f * 4, f * 8)                       # 256→512
        self.e5 = UNetDownBlock(f * 8, f * 8)                       # 512→512
        self.e6 = UNetDownBlock(f * 8, f * 8)                       # 512→512
        self.e7 = UNetDownBlock(f * 8, f * 8)                       # 512→512

        # ─── Bottleneck ────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # ─── Decoder ──────────────────────────────────────
        self.d7 = UNetUpBlock(f * 8, f * 8, use_dropout=True)       # 512→512, +skip=1024
        self.d6 = UNetUpBlock(f * 16, f * 8, use_dropout=True)      # 1024→512, +skip=1024
        self.d5 = UNetUpBlock(f * 16, f * 8, use_dropout=True)      # 1024→512, +skip=1024
        self.d4 = UNetUpBlock(f * 16, f * 8)                        # 1024→512, +skip=1024
        self.d3 = UNetUpBlock(f * 16, f * 4)                        # 1024→256, +skip=512
        self.d2 = UNetUpBlock(f * 8, f * 2)                         # 512→128, +skip=256
        self.d1 = UNetUpBlock(f * 4, f)                              # 256→64, +skip=128

        # ─── Output ───────────────────────────────────────
        self.output = nn.Sequential(
            nn.ConvTranspose2d(f * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.e1(x)     # (B, 64, 128, 128)
        e2 = self.e2(e1)    # (B, 128, 64, 64)
        e3 = self.e3(e2)    # (B, 256, 32, 32)
        e4 = self.e4(e3)    # (B, 512, 16, 16)
        e5 = self.e5(e4)    # (B, 512, 8, 8)
        e6 = self.e6(e5)    # (B, 512, 4, 4)
        e7 = self.e7(e6)    # (B, 512, 2, 2)

        # Bottleneck
        b = self.bottleneck(e7)  # (B, 512, 1, 1)

        # Decoder with skip connections
        d7 = self.d7(b, e7)      # (B, 1024, 2, 2)
        d6 = self.d6(d7, e6)     # (B, 1024, 4, 4)
        d5 = self.d5(d6, e5)     # (B, 1024, 8, 8)
        d4 = self.d4(d5, e4)     # (B, 1024, 16, 16)
        d3 = self.d3(d4, e3)     # (B, 512, 32, 32)
        d2 = self.d2(d3, e2)     # (B, 256, 64, 64)
        d1 = self.d1(d2, e1)     # (B, 128, 128, 128)

        return self.output(d1)    # (B, 3, 256, 256)


def init_weights(model: nn.Module, mean: float = 0.0, std: float = 0.02):
    """Initialize weights from N(mean, std) for Conv and BatchNorm layers."""
    for m in model.modules():
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, mean, std)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, std)
            nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    # Quick test
    model = UNetGenerator(in_channels=1, out_channels=3)
    init_weights(model)
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

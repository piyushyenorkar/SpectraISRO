"""
SPECTRA — SpectraSatNet Unified Model for Backend Inference

Copied from ml/model.py for backend deployment.
Two-stage pipeline: Super-Resolution + Colorization in a single forward pass.
"""

import torch
import torch.nn as nn


class SubPixelConvBlock(nn.Module):
    """Super-Resolution block using Sub-Pixel Convolution (PixelShuffle)."""
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(SubPixelConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x


class SpectraSatNet(nn.Module):
    """
    Unified End-to-End Network for ISRO PS-10.
    Stage 1: Super Resolution (200m TIR -> 100m TIR)
    Stage 2: Colorization (100m TIR -> 100m RGB)
    """
    def __init__(self):
        super(SpectraSatNet, self).__init__()

        # Stage 1: Super Resolution
        self.sr_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.sr_upsample = SubPixelConvBlock(64, 64, upscale_factor=2)
        self.sr_output = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # Stage 2: Colorization
        self.color_encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.color_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.color_output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x_low_res_tir):
        # Stage 1: Super Resolution
        sr_features = self.sr_feature_extractor(x_low_res_tir)
        sr_features_high_res = self.sr_upsample(sr_features)
        out_high_res_tir = self.sr_output(sr_features_high_res)

        # Stage 2: Colorization
        encoded = self.color_encoder(sr_features_high_res)
        decoded = self.color_decoder(encoded)
        decoded = decoded + sr_features_high_res  # Skip connection
        out_color_rgb = self.color_output(decoded)

        return out_high_res_tir, out_color_rgb

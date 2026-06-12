"""
SPECTRA — U-Net Generator (Backend Copy)
Identical to ml/models/generator.py for inference.
"""

import torch
import torch.nn as nn


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
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

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3, base_filters: int = 64):
        super().__init__()
        f = base_filters

        self.e1 = UNetDownBlock(in_channels, f, use_norm=False)
        self.e2 = UNetDownBlock(f, f * 2)
        self.e3 = UNetDownBlock(f * 2, f * 4)
        self.e4 = UNetDownBlock(f * 4, f * 8)
        self.e5 = UNetDownBlock(f * 8, f * 8)
        self.e6 = UNetDownBlock(f * 8, f * 8)
        self.e7 = UNetDownBlock(f * 8, f * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.d7 = UNetUpBlock(f * 8, f * 8, use_dropout=True)
        self.d6 = UNetUpBlock(f * 16, f * 8, use_dropout=True)
        self.d5 = UNetUpBlock(f * 16, f * 8, use_dropout=True)
        self.d4 = UNetUpBlock(f * 16, f * 8)
        self.d3 = UNetUpBlock(f * 16, f * 4)
        self.d2 = UNetUpBlock(f * 8, f * 2)
        self.d1 = UNetUpBlock(f * 4, f)

        self.output = nn.Sequential(
            nn.ConvTranspose2d(f * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)

        b = self.bottleneck(e7)

        d7 = self.d7(b, e7)
        d6 = self.d6(d7, e6)
        d5 = self.d5(d6, e5)
        d4 = self.d4(d5, e4)
        d3 = self.d3(d4, e3)
        d2 = self.d2(d3, e2)
        d1 = self.d1(d2, e1)

        return self.output(d1)

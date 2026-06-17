import torch
import torch.nn as nn
import torch.nn.functional as F

class SubPixelConvBlock(nn.Module):
    """
    Super-Resolution block using Sub-Pixel Convolution (PixelShuffle)
    """
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(SubPixelConvBlock, self).__init__()
        # To upscale by factor f, we need f^2 channels before PixelShuffle
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
    Unified End-to-End Network for ISRO PS-10
    Stage 1: Super Resolution (200m TIR -> 100m TIR)
    Stage 2: Colorization (100m TIR -> 100m RGB)
    """
    def __init__(self):
        super(SpectraSatNet, self).__init__()
        
        # ─── Stage 1: Super Resolution Branch (TIR 200m -> TIR 100m) ───
        # Input: 1 channel (Grayscale TIR)
        self.sr_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        # Upscale 2x
        self.sr_upsample = SubPixelConvBlock(64, 64, upscale_factor=2)
        # Output 1: High-Res TIR
        self.sr_output = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
        # ─── Stage 2: Colorization Branch (TIR 100m -> RGB 100m) ───
        # Takes the rich 64-channel 100m features from the SR branch
        self.color_encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # Downsample
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # Downsample
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.color_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # Upsample
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # Upsample
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output 2: Colorized RGB
        self.color_output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh() # Output in range [-1, 1]
        )

    def forward(self, x_low_res_tir):
        # Stage 1: Super Resolution
        sr_features = self.sr_feature_extractor(x_low_res_tir)
        sr_features_high_res = self.sr_upsample(sr_features)
        out_high_res_tir = self.sr_output(sr_features_high_res)
        
        # Stage 2: Colorization
        # We pass the high-res features directly into the colorizer
        encoded = self.color_encoder(sr_features_high_res)
        decoded = self.color_decoder(encoded)
        
        # Skip connection from SR output features to colorizer decoder
        decoded = decoded + sr_features_high_res
        
        out_color_rgb = self.color_output(decoded)
        
        return out_high_res_tir, out_color_rgb

# ─── Bonus: Physics-Informed Loss Function Stub ───

def thermal_emissivity_loss(tir_100m, predicted_rgb_100m):
    """
    Physics-Informed Loss:
    Penalizes the network if the predicted RGB color contradicts the 
    thermodynamic properties of the TIR temperature.
    
    Example: 
    - High TIR value (Hot) -> Should be urban/bare soil (Grey/Brown)
    - Low TIR value (Cold) -> Should be vegetation/water (Green/Blue)
    """
    # Normalize TIR to [0, 1] for thresholding
    tir_normalized = (tir_100m + 1.0) / 2.0
    
    # Extract RGB channels (assuming format is NCHW)
    r = predicted_rgb_100m[:, 0, :, :]
    g = predicted_rgb_100m[:, 1, :, :]
    b = predicted_rgb_100m[:, 2, :, :]
    
    # Example heuristic constraint:
    # If pixel is very cold (e.g. TIR < 0.2), vegetation/water probability is high
    # We penalize high Red values in cold regions.
    cold_mask = (tir_normalized < 0.2).float()
    penalty_cold = cold_mask * r  # Penalize red
    
    # If pixel is very hot (e.g. TIR > 0.8), urban/concrete probability is high
    # We penalize high Green values in hot regions.
    hot_mask = (tir_normalized > 0.8).float()
    penalty_hot = hot_mask * g    # Penalize green
    
    loss = torch.mean(penalty_cold) + torch.mean(penalty_hot)
    
    return loss * 0.1 # Weight of the physics loss

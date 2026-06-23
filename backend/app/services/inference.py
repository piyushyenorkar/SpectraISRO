"""
SPECTRA — Model Inference Service
Loads SpectraSatNet (unified SR + Colorization) and runs two-stage inference.
Falls back to UNetGenerator if SpectraSatNet weights are unavailable.
"""

import os
import time
import torch
import numpy as np

from app.models.spectra_sat_net import SpectraSatNet
from app.models.generator import UNetGenerator
from app.utils.preprocessing import load_and_preprocess
from app.utils.postprocessing import tensor_to_pil, pil_to_base64


class InferenceService:
    """Singleton service for model inference."""

    def __init__(self):
        self.model = None
        self.device = None
        self.model_type = None  # 'spectra_sat_net' or 'unet_fallback'

    def load_model(self, model_path: str = None):
        """
        Load the trained model.
        
        Priority:
        1. SpectraSatNet (e2e_best.pth) — unified two-stage model
        2. UNetGenerator (generator_best.pth / colorize_best.pth) — colorizer only
        3. Random weights (for development/testing)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Loading model on: {self.device}")

        # Try loading SpectraSatNet first
        e2e_path = model_path or os.environ.get("MODEL_PATH", "model/e2e_best.pth")
        colorize_path = os.environ.get("COLORIZE_MODEL_PATH", "model/colorize_best.pth")
        legacy_path = "model/generator_best.pth"

        if os.path.exists(e2e_path):
            # Load unified SpectraSatNet
            self.model = SpectraSatNet()
            state_dict = torch.load(e2e_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model_type = "spectra_sat_net"
            print(f"✅ SpectraSatNet loaded from: {e2e_path}")
        elif os.path.exists(colorize_path):
            # Fallback to UNetGenerator (colorizer only)
            self.model = UNetGenerator(in_channels=1, out_channels=3, base_filters=64)
            state_dict = torch.load(colorize_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model_type = "unet_fallback"
            print(f"✅ UNetGenerator loaded from: {colorize_path}")
        elif os.path.exists(legacy_path):
            # Legacy fallback
            self.model = UNetGenerator(in_channels=1, out_channels=3, base_filters=64)
            state_dict = torch.load(legacy_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model_type = "unet_fallback"
            print(f"✅ UNetGenerator loaded from: {legacy_path}")
        else:
            # No weights found — run with random weights for dev/testing
            self.model = SpectraSatNet()
            self.model_type = "spectra_sat_net"
            print(f"⚠️  No checkpoint found. Running with random weights (dev mode)")

        self.model.to(self.device)
        self.model.eval()

        params = sum(p.numel() for p in self.model.parameters())
        print(f"   Model type: {self.model_type}")
        print(f"   Parameters: {params:,}")

    def process(self, file_bytes: bytes) -> dict:
        """
        Run two-stage processing on uploaded image.

        Returns dict with:
            - input_image: base64 data URI (200m TIR input)
            - super_resolved_image: base64 data URI (100m TIR from Model A)
            - colorized_image: base64 data URI (100m RGB from Model B)
            - reference_image: base64 data URI (placeholder for ground truth)
            - metrics: processing metrics
            - original_size: [width, height]
            - processed_size: [256, 256]
        """
        start_time = time.time()

        # Preprocess input image
        input_tensor, original_size = load_and_preprocess(file_bytes)
        input_tensor = input_tensor.to(self.device)

        with torch.inference_mode():
            if self.model_type == "spectra_sat_net":
                # SpectraSatNet: single forward pass produces both outputs
                sr_tensor, colorized_tensor = self.model(input_tensor)
            else:
                # UNet fallback: no SR, just colorize
                sr_tensor = input_tensor  # Pass-through (no SR)
                colorized_tensor = self.model(input_tensor)

        processing_time_ms = (time.time() - start_time) * 1000

        # Postprocess all outputs to base64 images
        # Input (grayscale → 3-channel for display)
        input_pil = tensor_to_pil(input_tensor.repeat(1, 3, 1, 1))
        input_b64 = pil_to_base64(input_pil)

        # Super-resolved TIR (grayscale → 3-channel for display)
        sr_pil = tensor_to_pil(sr_tensor.repeat(1, 3, 1, 1))
        sr_b64 = pil_to_base64(sr_pil)

        # Colorized RGB
        colorized_pil = tensor_to_pil(colorized_tensor)
        colorized_b64 = pil_to_base64(colorized_pil)

        # Compute enhancement metrics
        ir_np = np.array(input_pil).astype(np.float32) / 255.0
        color_np = np.array(colorized_pil).astype(np.float32) / 255.0

        from app.services.metrics import compute_enhancement_metrics
        enhancement = compute_enhancement_metrics(ir_np, color_np)

        return {
            "input_image": input_b64,
            "super_resolved_image": sr_b64,
            "colorized_image": colorized_b64,
            "reference_image": colorized_b64,  # Placeholder until user uploads reference
            "metrics": {
                "processing_time_ms": round(processing_time_ms, 1),
                **enhancement,
            },
            "original_size": list(original_size),
            "processed_size": [256, 256],
        }


# Global singleton
inference_service = InferenceService()

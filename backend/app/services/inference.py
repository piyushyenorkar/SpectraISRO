"""
SPECTRA — Model Inference Service
Loads trained generator and runs colorization inference.
"""

import os
import torch
import numpy as np
from PIL import Image

from app.models.generator import UNetGenerator
from app.utils.preprocessing import load_and_preprocess
from app.utils.postprocessing import tensor_to_pil, pil_to_base64


class InferenceService:
    """Singleton service for model inference."""

    def __init__(self):
        self.model = None
        self.device = None

    def load_model(self, model_path: str = None):
        """Load the trained generator model."""
        if model_path is None:
            model_path = os.environ.get("MODEL_PATH", "model/generator_best.pth")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Loading model on: {self.device}")

        self.model = UNetGenerator(in_channels=1, out_channels=3, base_filters=64)

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"✅ Model loaded from: {model_path}")
        else:
            print(f"⚠️  No checkpoint found at {model_path}")
            print("   Running with random weights (for development/testing)")

        self.model.to(self.device)
        self.model.eval()

        params = sum(p.numel() for p in self.model.parameters())
        print(f"   Parameters: {params:,}")

    def colorize(self, file_bytes: bytes) -> dict:
        """
        Run colorization on uploaded image.

        Returns dict with:
            - colorized_image: base64 data URI
            - ir_preview: base64 data URI of preprocessed IR input
            - metrics: enhancement metrics
            - original_size: [width, height]
            - processed_size: [256, 256]
        """
        import time

        start_time = time.time()

        # Preprocess
        input_tensor, original_size = load_and_preprocess(file_bytes)
        input_tensor = input_tensor.to(self.device)

        # Inference
        with torch.inference_mode():
            output_tensor = self.model(input_tensor)

        processing_time_ms = (time.time() - start_time) * 1000

        # Postprocess
        colorized_pil = tensor_to_pil(output_tensor)
        colorized_b64 = pil_to_base64(colorized_pil)

        # Create IR preview (grayscale → 3-channel for display)
        ir_pil = tensor_to_pil(input_tensor.repeat(1, 3, 1, 1))
        ir_b64 = pil_to_base64(ir_pil)

        # Compute enhancement metrics
        ir_np = np.array(ir_pil).astype(np.float32) / 255.0
        color_np = np.array(colorized_pil).astype(np.float32) / 255.0

        from app.services.metrics import compute_enhancement_metrics
        enhancement = compute_enhancement_metrics(ir_np, color_np)

        return {
            "colorized_image": colorized_b64,
            "ir_preview": ir_b64,
            "metrics": {
                "processing_time_ms": round(processing_time_ms, 1),
                **enhancement,
            },
            "original_size": list(original_size),
            "processed_size": [256, 256],
        }


# Global singleton
inference_service = InferenceService()

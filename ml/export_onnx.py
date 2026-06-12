"""
SPECTRA — ONNX Export Script
Exports the trained U-Net Generator to ONNX format for browser inference.

Usage:
    python export_onnx.py --checkpoint checkpoints/generator_best.pth
    python export_onnx.py --checkpoint checkpoints/generator_best.pth --quantize
"""

import argparse
import os

import torch
import torch.onnx

import config
from models.generator import UNetGenerator


def export_to_onnx(checkpoint_path: str, output_path: str, quantize: bool = False):
    """Export trained generator to ONNX format."""

    # Load model
    device = torch.device("cpu")  # Export on CPU for compatibility
    model = UNetGenerator(
        in_channels=config.INPUT_CHANNELS,
        out_channels=config.OUTPUT_CHANNELS,
        base_filters=config.GENERATOR_FILTERS,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print(f"✅ Loaded model from: {checkpoint_path}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy input
    dummy_input = torch.randn(1, config.INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)

    # Export to ONNX
    print(f"📦 Exporting to ONNX (opset {config.ONNX_OPSET})...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=config.ONNX_OPSET,
        do_constant_folding=True,
        input_names=["ir_image"],
        output_names=["rgb_image"],
        dynamic_axes={
            "ir_image": {0: "batch_size"},
            "rgb_image": {0: "batch_size"},
        },
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ ONNX model saved: {output_path} ({file_size_mb:.1f} MB)")

    # Optional quantization for smaller browser download
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = output_path.replace(".onnx", "_quantized.onnx")
            quantize_dynamic(
                output_path,
                quantized_path,
                weight_type=QuantType.QUInt8,
            )
            q_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
            print(f"✅ Quantized model: {quantized_path} ({q_size_mb:.1f} MB)")
            print(f"   Size reduction: {(1 - q_size_mb / file_size_mb) * 100:.1f}%")
        except ImportError:
            print("⚠️  onnxruntime not installed. Skipping quantization.")
            print("   Install with: pip install onnxruntime")

    # Verify exported model
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: dummy_input.numpy()})
        print(f"✅ Verification passed! Output shape: {result[0].shape}")
    except ImportError:
        print("⚠️  onnxruntime not installed. Skipping verification.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SPECTRA generator to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(config.CHECKPOINT_DIR, "generator_best.pth"),
        help="Path to generator checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.ONNX_OUTPUT,
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic quantization (reduces model size ~60%%)",
    )
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.quantize)

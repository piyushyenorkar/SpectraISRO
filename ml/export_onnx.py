"""
SPECTRA — ONNX Export Script
Exports SpectraSatNet (or UNetGenerator fallback) to ONNX format.

Usage:
    python export_onnx.py --model e2e --checkpoint checkpoints/e2e_best.pth
    python export_onnx.py --model colorize --checkpoint checkpoints/colorize_best.pth
    python export_onnx.py --model e2e --checkpoint checkpoints/e2e_best.pth --quantize
"""

import argparse
import os

import torch
import torch.onnx

import config
from model import SpectraSatNet
from models.generator import UNetGenerator


def export_to_onnx(model_type: str, checkpoint_path: str, output_path: str, quantize: bool = False):
    """Export trained model to ONNX format."""

    device = torch.device("cpu")  # Export on CPU for compatibility

    if model_type == "e2e":
        model = SpectraSatNet()
        output_names = ["sr_tir", "colorized_rgb"]
        print("📦 Exporting SpectraSatNet (end-to-end)")
    else:
        model = UNetGenerator(
            in_channels=config.INPUT_CHANNELS,
            out_channels=config.OUTPUT_CHANNELS,
            base_filters=config.GENERATOR_FILTERS,
        )
        output_names = ["rgb_image"]
        print("📦 Exporting UNetGenerator (colorizer only)")

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"✅ Loaded weights from: {checkpoint_path}")
    else:
        print(f"⚠️  No checkpoint at {checkpoint_path}, exporting with random weights")

    model.eval()
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy input
    dummy_input = torch.randn(1, config.INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)

    # Dynamic axes for flexible batch size
    dynamic_axes = {"ir_image": {0: "batch_size"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    # Export
    print(f"📦 Exporting to ONNX (opset {config.ONNX_OPSET})...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=config.ONNX_OPSET,
        do_constant_folding=True,
        input_names=["ir_image"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ ONNX model saved: {output_path} ({file_size_mb:.1f} MB)")

    # Optional quantization
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = output_path.replace(".onnx", "_quantized.onnx")
            quantize_dynamic(output_path, quantized_path, weight_type=QuantType.QUInt8)
            q_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
            print(f"✅ Quantized model: {quantized_path} ({q_size_mb:.1f} MB)")
            print(f"   Size reduction: {(1 - q_size_mb / file_size_mb) * 100:.1f}%")
        except ImportError:
            print("⚠️  onnxruntime not installed. Skipping quantization.")

    # Verify
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: dummy_input.numpy()})
        for i, name in enumerate(output_names):
            print(f"✅ Output '{name}': shape={result[i].shape}")
    except ImportError:
        print("⚠️  onnxruntime not installed. Skipping verification.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SPECTRA model to ONNX")
    parser.add_argument("--model", type=str, default="e2e", choices=["e2e", "colorize"],
                        help="Model type: e2e (SpectraSatNet) or colorize (UNetGenerator)")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "e2e_best.pth"),
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=config.ONNX_OUTPUT,
                        help="Output ONNX file path")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply dynamic quantization")
    args = parser.parse_args()

    export_to_onnx(args.model, args.checkpoint, args.output, args.quantize)

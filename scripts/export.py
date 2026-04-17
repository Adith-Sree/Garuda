#!/usr/bin/env python3
"""
Export Script — Project Garuda

Export trained YOLOv8 models to ONNX, TFLite, and TensorRT formats
with quantization and optimization support.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("garuda.export", log_file="logs/export.log")

SUPPORTED_FORMATS = {
    "onnx": {"suffix": ".onnx", "desc": "ONNX (CPU/GPU portable)"},
    "tflite": {"suffix": ".tflite", "desc": "TFLite (Raspberry Pi)"},
    "engine": {"suffix": ".engine", "desc": "TensorRT (NVIDIA Jetson)"},
    "torchscript": {"suffix": ".torchscript", "desc": "TorchScript"},
    "openvino": {"suffix": "_openvino_model", "desc": "OpenVINO"},
    "coreml": {"suffix": ".mlpackage", "desc": "CoreML (Apple)"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to deployment formats"
    )
    parser.add_argument(
        "--weights", type=str, default="yolov8n.pt", help="Source .pt model path"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=list(SUPPORTED_FORMATS.keys()),
        help="Export format",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--half", action="store_true", help="FP16 quantization")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch size")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX graph")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument(
        "--config", type=str, default="configs/model.yaml", help="Config path"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/weights", help="Output directory"
    )
    parser.add_argument("--batch-all", action="store_true", help="Export all formats")
    return parser.parse_args()


def export_model(args: argparse.Namespace) -> None:
    """Export a YOLOv8 model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.exists() and not args.weights.startswith("yolov8"):
        logger.error("Weights not found: %s", args.weights)
        sys.exit(1)

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    formats_to_export = (
        list(SUPPORTED_FORMATS.keys()) if args.batch_all else [args.format]
    )

    model = YOLO(args.weights)

    for fmt in formats_to_export:
        logger.info("=" * 50)
        logger.info("Exporting to: %s (%s)", fmt, SUPPORTED_FORMATS[fmt]["desc"])
        logger.info("=" * 50)

        try:
            export_kwargs = {
                "format": fmt,
                "imgsz": args.imgsz,
                "half": args.half,
                "dynamic": args.dynamic if fmt == "onnx" else False,
                "simplify": args.simplify if fmt == "onnx" else False,
            }
            if fmt == "onnx":
                export_kwargs["opset"] = args.opset
            if args.int8 and fmt in ("tflite", "engine"):
                export_kwargs["int8"] = True

            result = model.export(**export_kwargs)

            logger.info("✅ Export successful: %s", result)
            logger.info("   Format : %s", fmt)
            logger.info("   Size   : %d", args.imgsz)
            logger.info("   FP16   : %s", args.half)
            logger.info("   INT8   : %s", args.int8)

        except Exception as e:
            logger.error("❌ Export failed for '%s': %s", fmt, e)
            if not args.batch_all:
                sys.exit(1)


def export_from_config(config_path: str) -> None:
    """Export using settings from model.yaml."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    export_cfg = config.get("export", {})
    model_cfg = config.get("model", {})

    args = argparse.Namespace(
        weights=model_cfg.get("weights", "yolov8n.pt"),
        format=export_cfg.get("format", "onnx"),
        imgsz=config.get("inference", {}).get("image_size", 640),
        half=export_cfg.get("half", False),
        int8=export_cfg.get("int8", False),
        dynamic=export_cfg.get("dynamic", False),
        simplify=export_cfg.get("simplify", True),
        opset=export_cfg.get("opset", 12),
        output_dir="models/weights",
        batch_all=False,
    )
    export_model(args)


if __name__ == "__main__":
    args = parse_args()
    export_model(args)

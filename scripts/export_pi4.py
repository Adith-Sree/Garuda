#!/usr/bin/env python3
"""
export_pi4.py — Project Garuda
Export YOLOv8 to ONNX for Raspberry Pi 4 deployment.

ONNX + ONNX Runtime is the recommended path for Pi 4:
  - Works on Python 3.9 / macOS / Linux without extra dependencies
  - ~3-8 FPS on Pi 4 @ 320x320 (same as TFLite FP32)
  - Single file, portable, no special runtime needed

Usage:
  python scripts/export_pi4.py --weights yolov8n.pt
  python scripts/export_pi4.py --weights models/weights/best.pt --imgsz 320
"""

import argparse
import sys
import shutil
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("garuda.export_pi4", log_file="logs/export_pi4.log")

PI4_IMGSZ = 320
OUT_DIR = Path("models/pi4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to ONNX for Raspberry Pi 4"
    )
    parser.add_argument(
        "--weights", type=str, default="yolov8n.pt",
        help="Source .pt model (e.g. yolov8n.pt or models/weights/best.pt)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=PI4_IMGSZ,
        help=f"Input image size (default: {PI4_IMGSZ})"
    )
    return parser.parse_args()


def export_onnx(weights: str, imgsz: int) -> Optional[Path]:
    """Export YOLOv8 model to ONNX format."""
    try:
        from ultralytics import YOLO
        model = YOLO(weights)

        logger.info("=" * 55)
        logger.info("EXPORTING TO ONNX @ %dx%d (Pi 4 optimized)", imgsz, imgsz)
        logger.info("=" * 55)

        exported = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=12,       # opset 12 = max onnxruntime on Pi supports
            simplify=True,
            dynamic=False,
        )
        exported_path = Path(exported)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(weights).stem
        dest = OUT_DIR / f"{stem}_pi4_{imgsz}.onnx"
        shutil.copy(str(exported_path), str(dest))

        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info("✅ ONNX export complete!")
        logger.info("   Path : %s", dest)
        logger.info("   Size : %.1f MB", size_mb)
        return dest

    except Exception as e:
        logger.error("❌ ONNX export failed: %s", e)
        return None


def print_summary(model_path: Optional[Path], imgsz: int) -> None:
    print("\n" + "=" * 55)
    print("  📦 PI 4 EXPORT SUMMARY")
    print("=" * 55)
    if model_path and model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ ONNX FP32   → {model_path}  ({size_mb:.1f} MB)")
        print(f"  ℹ️  Resolution  → {imgsz}x{imgsz}")
        print(f"  🎯 Target FPS  → ~3–8 FPS on Raspberry Pi 4")
    else:
        print("  ❌ Export FAILED — check logs/export_pi4.log")
    print("=" * 55)
    print("\n  📋 Next Steps (on Pi, via PuTTY):")
    print("  1. git pull origin main")
    print("     (or scp this file to Pi if not using git)")
    print(f"  2. scp {model_path} pi@<PI_IP>:~/Garuda/{model_path}")
    print("  3. source garuda_env/bin/activate")
    print("  4. python deployment/raspberry_pi/run_pi.py \\")
    print(f"       --model {model_path} \\")
    print("       --source 0 --imgsz 320")
    print("=" * 55)


def main():
    args = parse_args()
    logger.info("Project Garuda — Pi 4 ONNX Export")
    logger.info("Weights : %s", args.weights)
    logger.info("ImgSz   : %d", args.imgsz)

    result = export_onnx(args.weights, args.imgsz)
    print_summary(result, args.imgsz)


if __name__ == "__main__":
    main()

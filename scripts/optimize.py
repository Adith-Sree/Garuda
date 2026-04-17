#!/usr/bin/env python3
"""
Optimization Script — Project Garuda

Model optimisation utilities: quantization, pruning benchmarks,
and edge-device performance profiling.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("garuda.optimize", log_file="logs/optimize.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize YOLOv8 model for edge deployment"
    )
    parser.add_argument(
        "--weights", type=str, default="yolov8n.pt", help="Model weights path"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="benchmark",
        choices=["benchmark", "quantize", "profile", "compare"],
        help="Optimization task",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu/cuda/mps)"
    )
    parser.add_argument("--runs", type=int, default=50, help="Benchmark iterations")
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["pt", "onnx"],
        help="Formats to compare",
    )
    return parser.parse_args()


def benchmark(args: argparse.Namespace) -> None:
    """Benchmark model inference speed."""
    from ultralytics import YOLO

    model = YOLO(args.weights)
    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    logger.info("=" * 50)
    logger.info("BENCHMARK: %s on %s", args.weights, args.device)
    logger.info("=" * 50)

    # Warmup
    for _ in range(5):
        model.predict(dummy, device=args.device, verbose=False)

    # Benchmark
    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        model.predict(dummy, device=args.device, verbose=False, imgsz=args.imgsz)
        times.append(time.perf_counter() - t0)

    times_ms = [t * 1000 for t in times]
    avg = np.mean(times_ms)
    std = np.std(times_ms)
    fps = 1000.0 / avg

    logger.info("Results (%d runs):", args.runs)
    logger.info("  Avg latency : %.1f ms (± %.1f ms)", avg, std)
    logger.info("  Min latency : %.1f ms", np.min(times_ms))
    logger.info("  Max latency : %.1f ms", np.max(times_ms))
    logger.info("  Throughput  : %.1f FPS", fps)
    logger.info("=" * 50)


def quantize(args: argparse.Namespace) -> None:
    """Export quantized models."""
    from ultralytics import YOLO

    model = YOLO(args.weights)

    logger.info("=" * 50)
    logger.info("QUANTIZATION: %s", args.weights)
    logger.info("=" * 50)

    # FP16 ONNX
    try:
        logger.info("Exporting FP16 ONNX...")
        model.export(format="onnx", imgsz=args.imgsz, half=True, simplify=True)
        logger.info("✅ FP16 ONNX export complete")
    except Exception as e:
        logger.error("❌ FP16 ONNX failed: %s", e)

    # INT8 TFLite
    try:
        logger.info("Exporting INT8 TFLite...")
        model.export(format="tflite", imgsz=args.imgsz, int8=True)
        logger.info("✅ INT8 TFLite export complete")
    except Exception as e:
        logger.error("❌ INT8 TFLite failed: %s", e)


def profile_model(args: argparse.Namespace) -> None:
    """Profile model layer-by-layer."""
    from ultralytics import YOLO

    model = YOLO(args.weights)

    logger.info("=" * 50)
    logger.info("PROFILING: %s", args.weights)
    logger.info("=" * 50)

    info = model.info(detailed=True, verbose=True)
    logger.info("Model info: %s", info)

    # Run benchmark with different image sizes
    for size in [320, 416, 640]:
        dummy = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            model.predict(dummy, device=args.device, verbose=False, imgsz=size)
            times.append((time.perf_counter() - t0) * 1000)

        logger.info(
            "  imgsz=%d → avg=%.1f ms (%.1f FPS)",
            size,
            np.mean(times),
            1000.0 / np.mean(times),
        )


def compare_formats(args: argparse.Namespace) -> None:
    """Compare inference speed across model formats."""
    from ultralytics import YOLO

    logger.info("=" * 50)
    logger.info("FORMAT COMPARISON")
    logger.info("=" * 50)

    base_model = YOLO(args.weights)
    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    results = {}

    for fmt in args.formats:
        try:
            if fmt == "pt":
                model = base_model
            else:
                exported = base_model.export(format=fmt, imgsz=args.imgsz)
                model = YOLO(exported)

            # Warmup
            for _ in range(3):
                model.predict(dummy, device=args.device, verbose=False)

            # Benchmark
            times = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                model.predict(
                    dummy, device=args.device, verbose=False, imgsz=args.imgsz
                )
                times.append((time.perf_counter() - t0) * 1000)

            avg = np.mean(times)
            results[fmt] = avg
            logger.info(
                "  %-12s → avg=%.1f ms (%.1f FPS)", fmt, avg, 1000.0 / avg
            )
        except Exception as e:
            logger.error("  %-12s → FAILED: %s", fmt, e)

    if results:
        best = min(results, key=results.get)
        logger.info("  Best format: %s (%.1f ms)", best, results[best])


if __name__ == "__main__":
    args = parse_args()
    tasks = {
        "benchmark": benchmark,
        "quantize": quantize,
        "profile": profile_model,
        "compare": compare_formats,
    }
    tasks[args.task](args)

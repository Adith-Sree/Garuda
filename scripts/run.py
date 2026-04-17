#!/usr/bin/env python3
"""
Run Script — Project Garuda

Run real-time inference with detection, tracking, and alerts
on webcam, video file, or image input.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("garuda.run", log_file="logs/inference.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Garuda detection pipeline"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Input source: webcam index (0), video path, or image path"
    )
    parser.add_argument(
        "--config", type=str, default="configs/model.yaml", help="Config path"
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="Override model weights"
    )
    parser.add_argument(
        "--conf", type=float, default=None, help="Override confidence threshold"
    )
    parser.add_argument(
        "--frame-skip", type=int, default=None, help="Override frame skip"
    )
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument(
        "--no-tracking", action="store_true", help="Disable tracking"
    )
    parser.add_argument(
        "--no-alerts", action="store_true", help="Disable alerts"
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    """Run the detection pipeline."""
    import yaml
    from src.pipeline.pipeline import DetectionPipeline

    # Load and override config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.weights:
        config.setdefault("model", {})["weights"] = args.weights
    if args.conf is not None:
        config.setdefault("model", {})["confidence_threshold"] = args.conf
    if args.frame_skip is not None:
        config.setdefault("inference", {})["frame_skip"] = args.frame_skip
    if args.no_display:
        config.setdefault("inference", {})["show_display"] = False
    if args.save:
        config.setdefault("inference", {})["save_output"] = True
    if args.no_tracking:
        config.setdefault("tracking", {})["enabled"] = False
    if args.no_alerts:
        config.setdefault("alerts", {})["enabled"] = False

    logger.info("=" * 60)
    logger.info("PROJECT GARUDA — REAL-TIME INFERENCE")
    logger.info("=" * 60)
    logger.info("Source     : %s", args.source)
    logger.info("Config     : %s", args.config)
    logger.info("Tracking   : %s", "ON" if not args.no_tracking else "OFF")
    logger.info("Alerts     : %s", "ON" if not args.no_alerts else "OFF")
    logger.info("=" * 60)

    pipeline = DetectionPipeline(config=config)
    pipeline.run(source=args.source)


if __name__ == "__main__":
    args = parse_args()
    run(args)

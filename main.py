#!/usr/bin/env python3
"""
Project Garuda — Main Entry Point

CLI orchestrator for training, inference, export, and optimisation.

Usage:
    python main.py train  [--args]
    python main.py run    [--args]
    python main.py export [--args]
    python main.py optimize [--args]
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="🦅 Project Garuda — UAV Real-Time Object Detection & Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --data configs/data.yaml --epochs 50
  python main.py run --source 0 --conf 0.4
  python main.py run --source video.mp4 --save
  python main.py export --weights best.pt --format onnx --half
  python main.py optimize --task benchmark --weights yolov8n.pt
        """,
    )
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "run", "export", "optimize"],
        help="Operation mode",
    )

    # Parse only the mode, pass rest to sub-scripts
    args, remaining = parser.parse_known_args()

    # Route to appropriate script
    if args.mode == "train":
        sys.argv = ["train.py"] + remaining
        from scripts.train import parse_args, train

        train(parse_args())

    elif args.mode == "run":
        sys.argv = ["run.py"] + remaining
        from scripts.run import parse_args, run

        run(parse_args())

    elif args.mode == "export":
        sys.argv = ["export.py"] + remaining
        from scripts.export import parse_args, export_model

        export_model(parse_args())

    elif args.mode == "optimize":
        sys.argv = ["optimize.py"] + remaining
        from scripts.optimize import parse_args

        opt_args = parse_args()
        tasks = {
            "benchmark": lambda: __import__("scripts.optimize", fromlist=["benchmark"]).benchmark(opt_args),
            "quantize": lambda: __import__("scripts.optimize", fromlist=["quantize"]).quantize(opt_args),
            "profile": lambda: __import__("scripts.optimize", fromlist=["profile_model"]).profile_model(opt_args),
            "compare": lambda: __import__("scripts.optimize", fromlist=["compare_formats"]).compare_formats(opt_args),
        }
        tasks[opt_args.task]()


if __name__ == "__main__":
    main()

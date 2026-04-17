#!/usr/bin/env python3
"""
Training Script — Project Garuda

Train YOLOv8 models on VisDrone or custom UAV datasets with
full config support, resume, checkpointing, and metric logging.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger("garuda.train", log_file="logs/training.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for UAV object detection"
    )
    parser.add_argument(
        "--data", type=str, default="configs/data.yaml", help="Dataset config path"
    )
    parser.add_argument(
        "--config", type=str, default="configs/model.yaml", help="Model config path"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume training from checkpoint (path or 'last')"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cpu, cuda, mps, auto"
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="Override pretrained weights"
    )
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(args: argparse.Namespace) -> None:
    """Execute training run."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    # Resolve parameters (CLI overrides config)
    weights = args.weights or model_cfg.get("weights", "yolov8n.pt")
    epochs = args.epochs or train_cfg.get("epochs", 100)
    batch = args.batch or train_cfg.get("batch_size", 16)
    imgsz = args.imgsz or train_cfg.get("image_size", 640)
    device = args.device or model_cfg.get("device", "auto")
    project = train_cfg.get("project", "runs/train")
    name = args.name or train_cfg.get("name", "garuda_exp")
    patience = train_cfg.get("patience", 20)
    save_period = train_cfg.get("save_period", 10)
    workers = train_cfg.get("workers", 8)
    optimizer = train_cfg.get("optimizer", "AdamW")
    lr0 = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0005)
    warmup_epochs = train_cfg.get("warmup_epochs", 3)

    # Handle resume
    resume = False
    if args.resume:
        if args.resume.lower() == "last":
            resume = True
            logger.info("Resuming from last checkpoint.")
        else:
            weights = args.resume
            resume = True
            logger.info("Resuming from: %s", args.resume)
    elif train_cfg.get("resume", False):
        resume = True
        logger.info("Resuming training (from config).")

    # Resolve device
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "0"  # first GPU
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    logger.info("=" * 60)
    logger.info("PROJECT GARUDA — TRAINING")
    logger.info("=" * 60)
    logger.info("Model      : %s", weights)
    logger.info("Dataset    : %s", args.data)
    logger.info("Epochs     : %d", epochs)
    logger.info("Batch Size : %d", batch)
    logger.info("Image Size : %d", imgsz)
    logger.info("Device     : %s", device)
    logger.info("Optimizer  : %s (lr=%.4f)", optimizer, lr0)
    logger.info("=" * 60)

    # Load model
    model = YOLO(weights)

    # Train
    results = model.train(
        data=args.data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save_period=save_period,
        workers=workers,
        optimizer=optimizer,
        lr0=lr0,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        resume=resume,
        verbose=True,
        plots=True,
    )

    # Log final metrics
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Results saved to: %s/%s", project, name)
    logger.info("=" * 60)

    # Validate
    logger.info("Running validation...")
    val_results = model.val()
    logger.info("mAP50    : %.4f", val_results.box.map50)
    logger.info("mAP50-95 : %.4f", val_results.box.map)
    logger.info("Precision: %.4f", val_results.box.mp)
    logger.info("Recall   : %.4f", val_results.box.mr)


if __name__ == "__main__":
    args = parse_args()
    train(args)

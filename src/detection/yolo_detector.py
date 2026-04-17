from __future__ import annotations

"""
YOLOv8 Object Detector Module

Provides a unified interface for YOLOv8 inference across model formats
(.pt, .onnx, .tflite, .engine). Supports configurable confidence and
IOU thresholds, device auto-selection, and batch inference.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("garuda.detection")


class Detection:
    """Single detection result."""

    __slots__ = ("bbox", "confidence", "class_id", "class_name")

    def __init__(
        self,
        bbox: tuple[int, int, int, int],
        confidence: float,
        class_id: int,
        class_name: str,
    ):
        self.bbox = bbox            # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }

    def __repr__(self) -> str:
        return (
            f"Detection(class={self.class_name}, conf={self.confidence:.2f}, "
            f"bbox={self.bbox})"
        )


class YOLODetector:
    """
    YOLOv8 detector wrapper supporting multiple model formats.

    Args:
        model_path: Path to model file (.pt, .onnx, .tflite, .engine).
        confidence_threshold: Minimum confidence to keep detections.
        iou_threshold: IOU threshold for NMS.
        device: Inference device ('auto', 'cpu', 'cuda', 'mps').
        max_detections: Maximum number of detections per frame.
        image_size: Input image size for inference.
    """

    SUPPORTED_FORMATS = {".pt", ".onnx", ".tflite", ".engine", ".torchscript"}

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "auto",
        max_detections: int = 100,
        image_size: int = 640,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.image_size = image_size
        self.device = self._resolve_device(device)
        self.model = None
        self.class_names: dict[int, str] = {}

        self._load_model()

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' device to best available hardware."""
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self) -> None:
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO

            logger.info(
                "Loading model: %s on device: %s", self.model_path, self.device
            )
            self.model = YOLO(str(self.model_path))
            self.class_names = self.model.names or {}
            logger.info(
                "Model loaded successfully. Classes: %d", len(self.class_names)
            )
        except Exception as e:
            logger.error("Failed to load model '%s': %s", self.model_path, e)
            raise RuntimeError(f"Model loading failed: {e}") from e

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, C).

        Returns:
            List of Detection objects.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received, skipping detection.")
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
            return self._parse_results(results)
        except Exception as e:
            logger.error("Detection inference failed: %s", e)
            return []

    def _parse_results(self, results: Any) -> list[Detection]:
        """Parse Ultralytics Results into Detection objects."""
        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                detections.append(
                    Detection(
                        bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

        return detections

    def warmup(self, rounds: int = 3) -> None:
        """Warm up the model with dummy input."""
        logger.info("Warming up detector (%d rounds)...", rounds)
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        for _ in range(rounds):
            self.detect(dummy)
        logger.info("Warmup complete.")

    def update_thresholds(
        self,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> None:
        """Update detection thresholds at runtime."""
        if confidence is not None:
            self.confidence_threshold = max(0.0, min(1.0, confidence))
        if iou is not None:
            self.iou_threshold = max(0.0, min(1.0, iou))
        logger.info(
            "Thresholds updated — conf: %.2f, iou: %.2f",
            self.confidence_threshold,
            self.iou_threshold,
        )

    @classmethod
    def from_config(cls, config: dict) -> "YOLODetector":
        """Create detector from a config dictionary (model section)."""
        model_cfg = config.get("model", {})
        weights = model_cfg.get("weights", "yolov8n.pt")
        return cls(
            model_path=weights,
            confidence_threshold=model_cfg.get("confidence_threshold", 0.35),
            iou_threshold=model_cfg.get("iou_threshold", 0.45),
            device=model_cfg.get("device", "auto"),
            max_detections=model_cfg.get("max_detections", 100),
            image_size=config.get("inference", {}).get("image_size", 640),
        )

    def __repr__(self) -> str:
        return (
            f"YOLODetector(model={self.model_path}, device={self.device}, "
            f"conf={self.confidence_threshold}, iou={self.iou_threshold})"
        )

#!/usr/bin/env python3
from __future__ import annotations
"""
NVIDIA Jetson Inference — Project Garuda

Optimised inference pipeline for Jetson (Nano/Xavier/Orin) using
TensorRT or ONNX Runtime with CUDA acceleration.
Target: 10–25 FPS.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.logger import setup_logger
from src.utils.visualization import Visualizer
from src.tracking.tracker import ObjectTracker
from src.alerts.alert_manager import AlertManager

logger = setup_logger("garuda.jetson", log_file="logs/jetson_inference.log")


class JetsonDetector:
    """
    GPU-accelerated detector for NVIDIA Jetson.

    Supports:
    - TensorRT (.engine)
    - ONNX Runtime with CUDA (.onnx)
    - YOLOv8 PyTorch with CUDA (.pt)
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        image_size: int = 640,
        class_names: dict | None = None,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.class_names = class_names or {}
        self._backend = None
        self._model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load model based on file extension."""
        suffix = self.model_path.suffix

        if suffix == ".engine":
            self._load_tensorrt()
        elif suffix == ".onnx":
            self._load_onnx()
        elif suffix == ".pt":
            self._load_pytorch()
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def _load_pytorch(self) -> None:
        """Load YOLOv8 PyTorch model with CUDA."""
        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path))
        self._backend = "pytorch"

        if not self.class_names:
            self.class_names = self._model.names or {}

        logger.info("PyTorch/CUDA model loaded: %s", self.model_path)

    def _load_onnx(self) -> None:
        """Load ONNX model with GPU provider."""
        import onnxruntime as ort

        providers = ort.get_available_providers()
        preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        selected = [p for p in preferred if p in providers]

        self._model = ort.InferenceSession(str(self.model_path), providers=selected)
        self._backend = "onnx"

        logger.info(
            "ONNX model loaded: %s (providers: %s)", self.model_path, selected
        )

    def _load_tensorrt(self) -> None:
        """Load TensorRT engine."""
        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path))
        self._backend = "tensorrt"
        logger.info("TensorRT engine loaded: %s", self.model_path)

    def detect(self, frame: np.ndarray) -> list:
        """Run inference on a frame."""
        if self._backend in ("pytorch", "tensorrt"):
            return self._detect_ultralytics(frame)
        elif self._backend == "onnx":
            return self._detect_onnx(frame)
        return []

    def _detect_ultralytics(self, frame: np.ndarray) -> list:
        """Detect using Ultralytics (PyTorch/TensorRT)."""
        from src.detection.yolo_detector import Detection

        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=0,  # First GPU
            verbose=False,
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                detections.append(
                    Detection(
                        bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                    )
                )
        return detections

    def _detect_onnx(self, frame: np.ndarray) -> list:
        """Detect using ONNX Runtime."""
        from src.detection.yolo_detector import Detection

        # Preprocess
        input_name = self._model.get_inputs()[0].name
        h, w = self.image_size, self.image_size
        img = cv2.resize(frame, (w, h))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # Inference
        outputs = self._model.run(None, {input_name: img})
        output = outputs[0]

        # Parse output
        oh, ow = frame.shape[:2]
        detections = []

        if output.ndim == 3:
            output = output[0]
        if output.shape[0] < output.shape[1]:
            output = output.T

        for row in output:
            if len(row) < 6:
                continue
            x_c, y_c, bw, bh = row[0], row[1], row[2], row[3]
            scores = row[4:]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])

            if conf < self.confidence_threshold:
                continue

            x1 = int((x_c - bw / 2) / w * ow)
            y1 = int((y_c - bh / 2) / h * oh)
            x2 = int((x_c + bw / 2) / w * ow)
            y2 = int((y_c + bh / 2) / h * oh)

            detections.append(
                Detection(
                    bbox=(max(0, x1), max(0, y1), min(ow, x2), min(oh, y2)),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                )
            )

        return detections


def main():
    parser = argparse.ArgumentParser(description="Garuda — Jetson Inference")
    parser.add_argument(
        "--model", type=str, default="yolov8s.pt", help="Model path (.pt/.onnx/.engine)"
    )
    parser.add_argument("--source", type=str, default="0", help="Video source")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--frame-skip", type=int, default=1, help="Frame skip")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Config")
    parser.add_argument("--no-tracking", action="store_true", help="Disable tracking")
    args = parser.parse_args()

    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Class names
    class_names = {}
    data_cfg_path = Path("configs/data.yaml")
    if data_cfg_path.exists():
        with open(data_cfg_path) as f:
            data_cfg = yaml.safe_load(f)
            class_names = data_cfg.get("names", {})

    detector = JetsonDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        image_size=args.imgsz,
        class_names=class_names,
    )

    tracker = None if args.no_tracking else ObjectTracker.from_config(config)
    visualizer = Visualizer()
    alert_mgr = AlertManager.from_config(config) if config.get("alerts", {}).get("enabled") else None

    # Open source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Failed to open source: %s", args.source)
        sys.exit(1)

    logger.info("Jetson inference started — model: %s, source: %s", args.model, args.source)

    frame_count = 0
    fps_buffer = []
    last_dets = []
    last_tracks = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            t0 = time.perf_counter()

            if frame_count % args.frame_skip == 0:
                last_dets = detector.detect(frame)

                if tracker:
                    last_tracks = tracker.update(last_dets, frame)

                if alert_mgr:
                    alert_mgr.check_detections(last_tracks or last_dets)

            # Draw
            flagged = alert_mgr.flagged_classes if alert_mgr else None
            if last_tracks:
                display = visualizer.draw_tracks(frame, last_tracks, flagged)
            else:
                display = visualizer.draw_detections(frame, last_dets, flagged)

            # FPS
            elapsed = time.perf_counter() - t0
            if elapsed > 0:
                fps_buffer.append(1.0 / elapsed)
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)
            fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
            display = visualizer.draw_fps(display, fps)

            cv2.imshow("Garuda Jetson", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Jetson inference stopped. Frames: %d", frame_count)


if __name__ == "__main__":
    main()

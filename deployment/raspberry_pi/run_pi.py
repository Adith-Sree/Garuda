#!/usr/bin/env python3
from __future__ import annotations
"""
Raspberry Pi Inference — Project Garuda

Optimised inference pipeline for Raspberry Pi 4 using ONNX Runtime.
Target: 3–8 FPS on Raspberry Pi 4 @ 320x320.
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
from src.alerts.alert_manager import AlertManager
from src.tracking.tracker import ObjectTracker
from src.tracking.gimbal_tracker import GimbalTracker

logger = setup_logger("garuda.pi", log_file="logs/pi_inference.log")


class ONNXDetector:
    """Lightweight ONNX Runtime detector for Raspberry Pi 4."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        class_names: dict = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names or {}

        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError(
                "onnxruntime not installed. Run: pip install onnxruntime"
            )

        # Use CPU provider — Pi 4 doesn't have GPU
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
        self.imgsz = self.input_shape[2] if len(self.input_shape) == 4 else 320

        logger.info(
            "ONNX model loaded: %s (input: %s)", model_path, self.input_shape
        )

    def detect(self, frame: np.ndarray) -> list:
        """Run ONNX inference on a frame."""
        from src.detection.yolo_detector import Detection

        # Preprocess: resize → BGR→RGB → normalize → NCHW
        img = cv2.resize(frame, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))          # HWC → CHW
        img = np.expand_dims(img, axis=0)            # CHW → NCHW

        # Run inference
        outputs = self.session.run(None, {self.input_name: img})
        output = outputs[0]  # shape: [1, 84, N] for YOLOv8 COCO

        return self._parse_output(output, frame.shape[:2])

    def _parse_output(self, output: np.ndarray, original_shape: tuple) -> list:
        """Parse YOLOv8 ONNX output to Detection objects."""
        from src.detection.yolo_detector import Detection

        oh, ow = original_shape
        detections = []

        if output.ndim == 3:
            output = output[0]          # [84, N]
        if output.shape[0] < output.shape[1]:
            output = output.T           # → [N, 84]

        for row in output:
            if len(row) < 6:
                continue
            x_c, y_c, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            scores = row[4:]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])

            if conf < self.confidence_threshold:
                continue

            # Denormalize from [0,1] to pixel coordinates
            x1 = int((x_c - w / 2) * ow)
            y1 = int((y_c - h / 2) * oh)
            x2 = int((x_c + w / 2) * ow)
            y2 = int((y_c + h / 2) * oh)

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
    parser = argparse.ArgumentParser(description="Garuda — Raspberry Pi Inference")
    parser.add_argument(
        "--model", type=str, default="models/pi4/yolov8n_pi4_320.onnx",
        help="ONNX model path"
    )
    parser.add_argument("--source", type=str, default="0", help="Video source")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--frame-skip", type=int, default=3, help="Frame skip")
    parser.add_argument("--imgsz", type=int, default=320, help="Display resolution")
    parser.add_argument(
        "--config", type=str, default="configs/model.yaml", help="Config path"
    )
    args = parser.parse_args()

    # Load config for class names and alerts
    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Load data config for class names
    class_names = {}
    data_config_path = Path("configs/data.yaml")
    if data_config_path.exists():
        with open(data_config_path) as f:
            data_cfg = yaml.safe_load(f)
            class_names = data_cfg.get("names", {})

    detector = ONNXDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        class_names=class_names,
    )
    visualizer = Visualizer()
    alert_mgr = AlertManager.from_config(config) if config.get("alerts", {}).get("enabled") else None
    tracker = ObjectTracker.from_config(config) if config.get("tracking", {}).get("enabled", True) else None
    gimbal_tracker = GimbalTracker.from_config(config) if config.get("gimbal", {}).get("enabled", True) else None

    # Open source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Failed to open source: %s", args.source)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.imgsz)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.imgsz)

    logger.info("Pi inference started — model: %s, source: %s", args.model, args.source)

    frame_count = 0
    fps_buffer = []
    last_dets = []
    last_tracks = []
    latest_gimbal_signal = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            t0 = time.perf_counter()

            if frame_count % args.frame_skip == 0:
                last_dets = detector.detect(frame)
                if alert_mgr:
                    alert_mgr.check_detections(last_dets)
                
                # Tracking
                if tracker:
                    last_tracks = tracker.update(last_dets, frame)
                else:
                    last_tracks = []

            # Gimbal Tracking
            if gimbal_tracker:
                h_f, w_f = frame.shape[:2]
                latest_gimbal_signal = gimbal_tracker.update(last_tracks, w_f, h_f)

            # Visualization
            flagged = alert_mgr.flagged_classes if alert_mgr else None
            
            if last_tracks:
                display = visualizer.draw_tracks(
                    frame, last_tracks, flagged,
                    locked_id=gimbal_tracker.locked_track_id if gimbal_tracker else None
                )
            else:
                display = visualizer.draw_detections(frame, last_dets, flagged)

            # Draw Gimbal Info
            if latest_gimbal_signal and latest_gimbal_signal.locked:
                display = visualizer.draw_gimbal_info(display, latest_gimbal_signal)

            elapsed = time.perf_counter() - t0
            if elapsed > 0:
                fps_buffer.append(1.0 / elapsed)
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)
            fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
            display = visualizer.draw_fps(display, fps)

            cv2.imshow("Garuda Pi", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l"):
                if gimbal_tracker:
                    if gimbal_tracker.locked_track_id is None and last_tracks:
                        gimbal_tracker.lock_target(last_tracks[0].track_id)
                    else:
                        gimbal_tracker.unlock_target()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Pi inference stopped. Frames: %d", frame_count)


if __name__ == "__main__":
    main()

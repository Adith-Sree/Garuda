#!/usr/bin/env python3
from __future__ import annotations
"""
Raspberry Pi Inference — Project Garuda

Optimised inference pipeline for Raspberry Pi using TFLite models.
Target: 3–8 FPS on Raspberry Pi 4/5.
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


class TFLiteDetector:
    """Lightweight TFLite detector for Raspberry Pi."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        class_names: dict | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names or {}

        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]["shape"]

        logger.info(
            "TFLite model loaded: %s (input shape: %s)",
            model_path,
            self.input_shape,
        )

    def detect(self, frame: np.ndarray) -> list:
        """Run inference on a frame."""
        from src.detection.yolo_detector import Detection

        # Preprocess
        h, w = self.input_shape[1], self.input_shape[2]
        input_data = cv2.resize(frame, (w, h))
        input_data = np.expand_dims(input_data, axis=0)

        if self.input_details[0]["dtype"] == np.float32:
            input_data = input_data.astype(np.float32) / 255.0

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        # Parse output
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        detections = self._parse_output(output, frame.shape[:2])
        return detections

    def _parse_output(
        self, output: np.ndarray, original_shape: tuple
    ) -> list:
        """Parse TFLite output tensor to Detection objects."""
        from src.detection.yolo_detector import Detection

        detections = []
        oh, ow = original_shape

        # Output format depends on model export; handle common YOLOv8 TFLite format
        if output.ndim == 3:
            output = output[0]

        if output.shape[0] < output.shape[1]:
            output = output.T

        for row in output:
            if len(row) < 6:
                continue
            x_c, y_c, w, h = row[0], row[1], row[2], row[3]
            scores = row[4:]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])

            if conf < self.confidence_threshold:
                continue

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
        "--model", type=str, default="models/weights/yolov8n_float32.tflite",
        help="TFLite model path"
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

    detector = TFLiteDetector(
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

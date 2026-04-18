#!/usr/bin/env python3
from __future__ import annotations
"""
Raspberry Pi 4 Inference — Project Garuda (Minimal Build)

Optimised inference pipeline for Raspberry Pi 4 using ONNX Runtime.
Target: 3–8 FPS on Raspberry Pi 4 @ 320×320.

Dependencies (all in requirements_pi.txt):
  numpy, opencv-python-headless, PyYAML, onnxruntime

NOT required on Pi:
  torch, ultralytics, deep_sort_realtime, requests, gradio
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Add project root to path so src/ modules are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.utils.visualization import Visualizer
from src.alerts.alert_manager import AlertManager
from src.tracking.tracker import ObjectTracker
from src.tracking.gimbal_tracker import GimbalTracker

logger = setup_logger("garuda.pi", log_file="logs/pi_inference.log")


# ---------------------------------------------------------------------------
# Minimal Detection dataclass (mirrors src/detection/yolo_detector.Detection)
# ---------------------------------------------------------------------------
from dataclasses import dataclass

@dataclass
class Detection:
    bbox: tuple        # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str = ""


# ---------------------------------------------------------------------------
# ONNX Detector
# ---------------------------------------------------------------------------
class ONNXDetector:
    """Lightweight ONNX Runtime detector for Raspberry Pi 4."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        class_names: dict | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names or {}

        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError(
                "onnxruntime not installed.\n"
                "Run:  pip install onnxruntime>=1.16.0"
            )

        model_path = str(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Export it first on your PC:\n"
                "  yolo export model=yolov8n.pt format=onnx imgsz=320 simplify=True\n"
                "  mv yolov8n.onnx models/pi4/yolov8n_pi4_320.onnx"
            )

        # CPU provider only — Pi 4 has no GPU
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4        # Pi 4 has 4 cores
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
        self.imgsz = self.input_shape[2] if len(self.input_shape) == 4 else 320

        logger.info(
            "ONNX model loaded: %s  (input shape: %s)", model_path, self.input_shape
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run ONNX inference on a BGR frame."""
        # Preprocess: resize → BGR→RGB → normalise → NCHW
        img = cv2.resize(frame, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))       # HWC → CHW
        blob = np.expand_dims(img, axis=0)        # → NCHW

        outputs = self.session.run(None, {self.input_name: blob})
        return self._parse_output(outputs[0], frame.shape[:2])

    def _parse_output(self, output: np.ndarray, orig_shape: tuple) -> list[Detection]:
        """Parse YOLOv8 ONNX output → Detection list."""
        oh, ow = orig_shape
        detections: list[Detection] = []

        if output.ndim == 3:
            output = output[0]           # [84, N]
        if output.shape[0] < output.shape[1]:
            output = output.T            # → [N, 84]

        for row in output:
            if len(row) < 6:
                continue
            x_c, y_c, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            scores = row[4:]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])

            if conf < self.confidence_threshold:
                continue

            # Denormalize from [0,1] → pixel coordinates
            x1 = int((x_c - w / 2) * ow)
            y1 = int((y_c - h / 2) * oh)
            x2 = int((x_c + w / 2) * ow)
            y2 = int((y_c + h / 2) * oh)

            detections.append(
                Detection(
                    bbox=(max(0, x1), max(0, y1), min(ow, x2), min(oh, y2)),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, f"cls_{cls_id}"),
                )
            )

        return detections


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Garuda — Raspberry Pi 4 Inference (minimal build)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/pi4/yolov8n_pi4_320.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: 0=webcam, /dev/videoX, or RTSP URL",
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=3,
        help="Run detection every N frames (others reuse last result)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="deployment/raspberry_pi/pi_config.yaml",
        help="Pi config YAML path",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Headless mode — do not call cv2.imshow (use for SSH/no monitor)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config: dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        logger.info("Config loaded: %s", config_path)
    else:
        logger.warning("Config not found: %s — using CLI defaults", config_path)

    # Class names from configs/data.yaml (optional)
    class_names: dict[int, str] = {}
    data_cfg_path = PROJECT_ROOT / "configs" / "data.yaml"
    if data_cfg_path.exists():
        with open(data_cfg_path) as f:
            data_cfg = yaml.safe_load(f) or {}
        raw = data_cfg.get("names", {})
        if isinstance(raw, list):
            class_names = {i: n for i, n in enumerate(raw)}
        elif isinstance(raw, dict):
            class_names = {int(k): v for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Build pipeline components
    # ------------------------------------------------------------------
    detector = ONNXDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        class_names=class_names,
    )
    visualizer = Visualizer()

    # ObjectTracker — force IOU-only mode (no deep_sort_realtime on Pi)
    tracker: ObjectTracker | None = None
    if config.get("tracking", {}).get("enabled", True):
        tracker = ObjectTracker.from_config(config)
        tracker._use_deepsort = False   # always use built-in IOU tracker on Pi
        logger.info("IOU tracker enabled (DeepSORT disabled on Pi).")

    # AlertManager — requests is optional; webhook disabled when not installed
    alert_mgr: AlertManager | None = None
    if config.get("alerts", {}).get("enabled", False):
        alert_mgr = AlertManager.from_config(config)
        logger.info("Alert manager enabled: flagged=%s", alert_mgr.flagged_classes)

    # GimbalTracker
    gimbal_tracker: GimbalTracker | None = None
    if config.get("gimbal", {}).get("enabled", True):
        gimbal_tracker = GimbalTracker.from_config(config)
        logger.info("Gimbal tracker enabled.")

    # ------------------------------------------------------------------
    # Open video source
    # ------------------------------------------------------------------
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Failed to open source: %s", args.source)
        sys.exit(1)

    # Set capture resolution to match model input (saves decode + resize cost)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.imgsz)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.imgsz)

    display = not args.no_display

    logger.info(
        "Pi inference started — model=%s  source=%s  display=%s",
        args.model, args.source, display,
    )
    print(
        f"\n[Garuda Pi] Running. "
        f"{'Press Q to quit.' if display else 'Press Ctrl+C to stop.'}\n"
    )

    frame_count = 0
    fps_buffer: list[float] = []
    last_dets: list[Detection] = []
    last_tracks: list = []
    latest_gimbal_signal = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed — end of stream or camera error.")
                break

            frame_count += 1
            t0 = time.perf_counter()

            # Detection (every frame_skip frames)
            if frame_count % args.frame_skip == 0:
                last_dets = detector.detect(frame)

                if alert_mgr:
                    alert_mgr.check_detections(last_dets)

                if tracker:
                    last_tracks = tracker.update(last_dets, frame)
                else:
                    last_tracks = []

            # Gimbal control signal
            if gimbal_tracker:
                h_f, w_f = frame.shape[:2]
                latest_gimbal_signal = gimbal_tracker.update(last_tracks, w_f, h_f)

            # Timing / FPS
            elapsed = time.perf_counter() - t0
            if elapsed > 0:
                fps_buffer.append(1.0 / elapsed)
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)
            fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0.0

            # Print FPS to console every 30 frames (useful in headless mode)
            if frame_count % 30 == 0:
                dets_n = len(last_dets)
                trk_n = len(last_tracks)
                print(
                    f"  frame={frame_count:5d}  FPS={fps:4.1f}  "
                    f"dets={dets_n}  tracks={trk_n}",
                    flush=True,
                )

            if not display:
                continue

            # Visualization (skipped in headless mode)
            flagged = alert_mgr.flagged_classes if alert_mgr else None

            if last_tracks:
                annotated = visualizer.draw_tracks(
                    frame, last_tracks, flagged,
                    locked_id=gimbal_tracker.locked_track_id if gimbal_tracker else None,
                )
            else:
                annotated = visualizer.draw_detections(frame, last_dets, flagged)

            if latest_gimbal_signal and latest_gimbal_signal.locked:
                annotated = visualizer.draw_gimbal_info(annotated, latest_gimbal_signal)

            annotated = visualizer.draw_fps(annotated, fps)

            cv2.imshow("Garuda Pi", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l") and gimbal_tracker:
                if gimbal_tracker.locked_track_id is None and last_tracks:
                    gimbal_tracker.lock_target(last_tracks[0].track_id)
                    print("  [Gimbal] Locked onto track", last_tracks[0].track_id)
                else:
                    gimbal_tracker.unlock_target()
                    print("  [Gimbal] Unlocked.")

    except KeyboardInterrupt:
        print("\n[Garuda Pi] Stopped by user.")
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        logger.info(
            "Pi inference stopped. frames=%d  avg_fps=%.1f",
            frame_count, (sum(fps_buffer) / len(fps_buffer)) if fps_buffer else 0,
        )


if __name__ == "__main__":
    main()

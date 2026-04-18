from __future__ import annotations

"""
Detection Pipeline Module

Orchestrates the full capture → detect → track → alert → render loop
with frame skipping, async-ready architecture, and performance monitoring.
"""

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.detection.yolo_detector import YOLODetector
from src.tracking.tracker import ObjectTracker
from src.tracking.gimbal_tracker import GimbalTracker
from src.alerts.alert_manager import AlertManager
from src.utils.visualization import Visualizer
from src.utils.preprocessing import FramePreprocessor
from src.utils.logger import setup_logger

logger = logging.getLogger("garuda.pipeline")


class DetectionPipeline:
    """
    End-to-end real-time detection + tracking + alert pipeline.

    Args:
        config_path: Path to model.yaml configuration file.
        config: Pre-loaded config dict (overrides config_path).
    """

    def __init__(
        self,
        config_path: str = "configs/model.yaml",
        config: dict | None = None,
    ):
        self.config = config or self._load_config(config_path)
        self.running = False

        # Initialize components
        self.detector = YOLODetector.from_config(self.config)
        self.tracker = (
            ObjectTracker.from_config(self.config)
            if self.config.get("tracking", {}).get("enabled", True)
            else None
        )
        self.alert_manager = (
            AlertManager.from_config(self.config)
            if self.config.get("alerts", {}).get("enabled", True)
            else None
        )
        self.visualizer = Visualizer()
        self.gimbal_tracker = (
            GimbalTracker.from_config(self.config)
            if self.config.get("gimbal", {}).get("enabled", True)
            else None
        )
        self.preprocessor = FramePreprocessor(
            target_width=self.config.get("inference", {}).get("resolution_width", 1280),
            target_height=self.config.get("inference", {}).get("resolution_height", 720),
        )

        # Pipeline settings
        inf_cfg = self.config.get("inference", {})
        self.frame_skip = inf_cfg.get("frame_skip", 2)
        self.show_display = inf_cfg.get("show_display", True)
        self.save_output = inf_cfg.get("save_output", False)
        self.output_path = inf_cfg.get("output_path", "runs/inference")

        # Performance tracking
        self._frame_count = 0
        self._fps = 0.0
        self._fps_buffer: list[float] = []

        logger.info("Pipeline initialized | %s", self.detector)

    @staticmethod
    def _load_config(path: str) -> dict:
        """Load YAML config file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def run(self, source: Any = None) -> None:
        """
        Run the full pipeline.

        Args:
            source: Video source — int (webcam index), str (video path), or None (from config).
        """
        if source is None:
            source = self.config.get("inference", {}).get("source", 0)

        cap = self._open_source(source)
        if cap is None:
            return

        writer = self._init_writer(cap) if self.save_output else None
        self.running = True
        last_detections = []
        last_tracks = []
        self._latest_gimbal_signal = None

        # Warmup
        self.detector.warmup(rounds=2)

        logger.info("Pipeline started — source: %s | frame_skip: %d", source, self.frame_skip)

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream.")
                    break

                self._frame_count += 1
                t_start = time.perf_counter()

                # Frame skipping: only run inference on selected frames
                if self._frame_count % self.frame_skip == 0:
                    detections = self.detector.detect(frame)
                    last_detections = detections

                    # Tracking
                    if self.tracker:
                        tracks = self.tracker.update(detections, frame)
                        last_tracks = tracks
                    else:
                        last_tracks = []

                    # Alerts
                    if self.alert_manager:
                        alert_targets = last_tracks if last_tracks else last_detections
                        alerts = self.alert_manager.check_detections(alert_targets)

                # Gimbal Tracking
                if self.gimbal_tracker:
                    h, w = frame.shape[:2]
                    self._latest_gimbal_signal = self.gimbal_tracker.update(last_tracks, w, h)
                else:
                    self._latest_gimbal_signal = None

                # Visualize
                display_frame = frame.copy()
                if last_tracks:
                    flagged = (
                        self.alert_manager.flagged_classes
                        if self.alert_manager
                        else None
                    )
                    display_frame = self.visualizer.draw_tracks(
                        display_frame, last_tracks, flagged,
                        locked_id=self.gimbal_tracker.locked_track_id if self.gimbal_tracker else None
                    )
                elif last_detections:
                    flagged = (
                        self.alert_manager.flagged_classes
                        if self.alert_manager
                        else None
                    )
                    display_frame = self.visualizer.draw_detections(
                        display_frame, last_detections, flagged
                    )

                # FPS & Gimbal Info
                elapsed = time.perf_counter() - t_start
                self._update_fps(elapsed)
                display_frame = self.visualizer.draw_fps(display_frame, self._fps)
                
                if self._latest_gimbal_signal and self._latest_gimbal_signal.locked:
                    display_frame = self.visualizer.draw_gimbal_info(
                        display_frame, self._latest_gimbal_signal
                    )

                # Output
                if self.show_display:
                    cv2.imshow("Garuda — UAV Detection", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("User quit (q pressed).")
                        break
                    elif key == ord("+"):
                        self.detector.update_thresholds(
                            confidence=self.detector.confidence_threshold + 0.05
                        )
                    elif key == ord("-"):
                        self.detector.update_thresholds(
                            confidence=self.detector.confidence_threshold - 0.05
                        )
                    elif key == ord("l"):
                        # Lock onto the first available track if none locked, or unlock
                        if self.gimbal_tracker:
                            if self.gimbal_tracker.locked_track_id is None and last_tracks:
                                self.gimbal_tracker.lock_target(last_tracks[0].track_id)
                            else:
                                self.gimbal_tracker.unlock_target()

                if writer:
                    writer.write(display_frame)

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user.")
        finally:
            self._cleanup(cap, writer)

    def _open_source(self, source: Any) -> cv2.VideoCapture | None:
        """Open video capture source with robustness for different platforms."""
        import platform
        is_mac = platform.system() == "Darwin"

        try:
            # Handle string source that might be a digit (e.g., "0")
            actual_source = int(source) if str(source).isdigit() else source
            
            if isinstance(actual_source, str) and not Path(actual_source).exists():
                logger.error("Video file not found: %s", actual_source)
                return None

            # Try default opening
            cap = cv2.VideoCapture(actual_source)
            
            # On Mac, if default fails, try AVFOUNDATION explicitly
            if is_mac and not cap.isOpened():
                logger.info("Retrying with CAP_AVFOUNDATION for macOS...")
                cap = cv2.VideoCapture(actual_source, cv2.CAP_AVFOUNDATION)

            if not cap.isOpened():
                logger.error(
                    "Failed to open video source: %s. "
                    "Please check your camera index or file path.", 
                    source
                )
                return None

            # Set resolution
            inf_cfg = self.config.get("inference", {})
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, inf_cfg.get("resolution_width", 1280))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, inf_cfg.get("resolution_height", 720))

            logger.info(
                "Capture opened successfully: %dx%d (Source: %s)",
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                source
            )
            return cap
        except Exception as e:
            logger.error("Error opening source '%s': %s", source, e)
            return None

    def _init_writer(self, cap: cv2.VideoCapture) -> cv2.VideoWriter:
        """Initialize video writer for saving output."""
        out_dir = Path(self.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"output_{int(time.time())}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))
        logger.info("Saving output to: %s", out_file)
        return writer

    def _update_fps(self, elapsed: float) -> None:
        """Update smoothed FPS counter."""
        if elapsed > 0:
            self._fps_buffer.append(1.0 / elapsed)
            if len(self._fps_buffer) > 30:
                self._fps_buffer.pop(0)
            self._fps = sum(self._fps_buffer) / len(self._fps_buffer)

    def _cleanup(
        self, cap: cv2.VideoCapture | None, writer: cv2.VideoWriter | None
    ) -> None:
        """Release resources."""
        self.running = False
        if cap:
            cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info(
            "Pipeline stopped. Total frames: %d | Alerts: %d",
            self._frame_count,
            self.alert_manager.total_alerts if self.alert_manager else 0,
        )

    def stop(self) -> None:
        """Signal the pipeline to stop."""
        self.running = False

    def __repr__(self) -> str:
        return (
            f"DetectionPipeline(detector={self.detector}, "
            f"tracker={self.tracker}, frame_skip={self.frame_skip})"
        )

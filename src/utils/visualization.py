from __future__ import annotations

"""
Visualization Module

Drawing utilities for bounding boxes, track IDs, class labels,
FPS overlay, and alert indicators on video frames.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger("garuda.utils.visualization")

# Distinct colors for up to 20 classes (BGR)
CLASS_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (64, 255, 64),
    (255, 64, 64),
    (64, 64, 255),
    (192, 192, 0),
    (0, 192, 192),
    (192, 0, 192),
    (128, 128, 128),
]


class Visualizer:
    """
    Draw detection and tracking overlays on frames.

    Args:
        line_thickness: Bounding box line thickness.
        font_scale: Text font scale.
        show_confidence: Whether to show confidence scores.
        show_track_id: Whether to show track IDs.
        alert_color: Color for alert-flagged objects (BGR).
    """

    def __init__(
        self,
        line_thickness: int = 2,
        font_scale: float = 0.6,
        show_confidence: bool = True,
        show_track_id: bool = True,
        alert_color: tuple[int, int, int] = (0, 0, 255),
        lock_color: tuple[int, int, int] = (0, 255, 255),  # Cyan
    ):
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.show_confidence = show_confidence
        self.show_track_id = show_track_id
        self.alert_color = alert_color
        self.lock_color = lock_color

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list,
        flagged_classes: set[str] | None = None,
    ) -> np.ndarray:
        """Draw bounding boxes for raw detections."""
        annotated = frame.copy()
        flagged = flagged_classes or set()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            is_flagged = det.class_name in flagged
            color = self.alert_color if is_flagged else self._get_color(det.class_id)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.line_thickness)

            label_parts = [det.class_name]
            if self.show_confidence:
                label_parts.append(f"{det.confidence:.0%}")
            label = " ".join(label_parts)

            self._draw_label(annotated, label, (x1, y1), color, is_flagged)

        return annotated

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: list,
        flagged_classes: set[str] | None = None,
        locked_id: int | None = None,
    ) -> np.ndarray:
        """Draw bounding boxes with track IDs."""
        annotated = frame.copy()
        flagged = flagged_classes or set()

        for trk in tracks:
            x1, y1, x2, y2 = trk.bbox
            is_flagged = trk.class_name in flagged
            is_locked = trk.track_id == locked_id
            
            if is_locked:
                color = self.lock_color
                thickness = self.line_thickness * 2
            else:
                color = (
                    self.alert_color
                    if is_flagged
                    else CLASS_COLORS[int(trk.track_id) % len(CLASS_COLORS)]
                )
                thickness = self.line_thickness

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            if is_locked:
                # Draw corner brackets for locked target
                len_l = 20
                cv2.line(annotated, (x1, y1), (x1 + len_l, y1), color, thickness)
                cv2.line(annotated, (x1, y1), (x1, y1 + len_l), color, thickness)
                cv2.line(annotated, (x2, y1), (x2 - len_l, y1), color, thickness)
                cv2.line(annotated, (x2, y1), (x2, y1 + len_l), color, thickness)
                cv2.line(annotated, (x1, y2), (x1 + len_l, y2), color, thickness)
                cv2.line(annotated, (x1, y2), (x1, y2 - len_l), color, thickness)
                cv2.line(annotated, (x2, y2), (x2 - len_l, y2), color, thickness)
                cv2.line(annotated, (x2, y2), (x2, y2 - len_l), color, thickness)

            label_parts = []
            if is_locked:
                label_parts.append("[LOCKED]")
            if self.show_track_id:
                label_parts.append(f"ID:{trk.track_id}")
            if trk.class_name:
                label_parts.append(trk.class_name)
            if self.show_confidence and trk.confidence > 0:
                label_parts.append(f"{trk.confidence:.0%}")
            label = " ".join(label_parts)

            self._draw_label(annotated, label, (x1, y1), color, is_flagged)

        return annotated

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter on the frame."""
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return frame

    def draw_alert_banner(
        self, frame: np.ndarray, message: str
    ) -> np.ndarray:
        """Draw a red alert banner at the top of the frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 180), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(
            frame,
            f"⚠ ALERT: {message}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        highlight: bool = False,
    ) -> None:
        """Draw a label with a filled background above the bounding box."""
        x, y = origin
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
        )
        bg_color = self.alert_color if highlight else color
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), bg_color, -1)
        cv2.putText(
            frame,
            label,
            (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def draw_gimbal_info(self, frame: np.ndarray, signal: Any) -> np.ndarray:
        """Draw gimbal control signal info."""
        h, w = frame.shape[:2]
        
        # Draw center crosshair
        cv2.line(frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (255, 255, 255), 1)
        cv2.line(frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (255, 255, 255), 1)
        
        # Display control offsets
        text = f"GIMBAL: LOCK={signal.track_id} DX={signal.dx:+.2f} DY={signal.dy:+.2f}"
        cv2.putText(
            frame,
            text,
            (10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.lock_color,
            2,
            cv2.LINE_AA,
        )
        
        # Draw error vector
        if signal.dx != 0 or signal.dy != 0:
            end_point = (
                int(w // 2 + signal.dx * 100),
                int(h // 2 + signal.dy * 100)
            )
            cv2.arrowedLine(frame, (w // 2, h // 2), end_point, self.lock_color, 2)
            
        return frame

    @staticmethod
    def _get_color(class_id: int) -> tuple[int, int, int]:
        """Get a color for a class ID."""
        return CLASS_COLORS[class_id % len(CLASS_COLORS)]

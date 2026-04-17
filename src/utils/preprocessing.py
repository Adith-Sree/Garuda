from __future__ import annotations

"""
Frame Preprocessing Module

Utilities for frame resizing, letterboxing, normalisation, and
colour-space conversion for optimised edge inference.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger("garuda.utils.preprocessing")


class FramePreprocessor:
    """
    Preprocess frames for model inference.

    Args:
        target_width: Target frame width.
        target_height: Target frame height.
        normalize: Whether to normalise pixel values to [0, 1].
        letterbox: Whether to use letterbox resizing (preserves aspect ratio).
    """

    def __init__(
        self,
        target_width: int = 640,
        target_height: int = 640,
        normalize: bool = False,
        letterbox: bool = True,
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.normalize = normalize
        self.letterbox = letterbox

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target dimensions."""
        if self.letterbox:
            return self._letterbox_resize(frame)
        return cv2.resize(
            frame,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _letterbox_resize(
        self,
        frame: np.ndarray,
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> np.ndarray:
        """Resize with letterboxing to maintain aspect ratio."""
        h, w = frame.shape[:2]
        scale = min(self.target_width / w, self.target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full(
            (self.target_height, self.target_width, 3), color, dtype=np.uint8
        )
        dx = (self.target_width - new_w) // 2
        dy = (self.target_height - new_h) // 2
        canvas[dy : dy + new_h, dx : dx + new_w] = resized

        return canvas

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline: resize → normalize."""
        processed = self.resize(frame)
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0
        return processed

    @staticmethod
    def to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def to_bgr(frame: np.ndarray) -> np.ndarray:
        """Convert RGB to BGR."""
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    @staticmethod
    def adjust_brightness(frame: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """Adjust brightness by a multiplicative factor."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def equalize_histogram(frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE histogram equalization for low-light enhancement."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def __repr__(self) -> str:
        return (
            f"FramePreprocessor(size={self.target_width}x{self.target_height}, "
            f"letterbox={self.letterbox}, normalize={self.normalize})"
        )

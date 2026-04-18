from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("garuda.tracking.gimbal")

@dataclass
class GimbalSignal:
    """Control signal for gimbal centering."""
    dx: float = 0.0          # Normalized x-offset (-1.0 to 1.0)
    dy: float = 0.0          # Normalized y-offset (-1.0 to 1.0)
    locked: bool = False     # Whether a target is currently locked
    track_id: int | None = None

class GimbalTracker:
    """
    Calculates center-to-target error and generates control signals.
    
    Args:
        p_gain: Proportional gain
        i_gain: Integral gain
        d_gain: Derivative gain
        deadzone: Error threshold below which no signal is sent
    """
    def __init__(
        self,
        p_gain: float = 0.5,
        i_gain: float = 0.0,
        d_gain: float = 0.1,
        deadzone: float = 0.02,
    ):
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.deadzone = deadzone
        
        self.locked_track_id: int | None = None
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._integral_x = 0.0
        self._integral_y = 0.0

    def lock_target(self, track_id: int) -> None:
        """Lock onto a specific track ID."""
        if self.locked_track_id != track_id:
            logger.info("Locking onto target ID: %d", track_id)
            self.locked_track_id = track_id
            self.reset_pid()

    def unlock_target(self) -> None:
        """Release the current target lock."""
        if self.locked_track_id is not None:
            logger.info("Unlocking target ID: %d", self.locked_track_id)
            self.locked_track_id = None
            self.reset_pid()

    def reset_pid(self) -> None:
        """Reset PID accumulators."""
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._integral_x = 0.0
        self._integral_y = 0.0

    def update(self, tracks: list, frame_width: int, frame_height: int) -> GimbalSignal:
        """
        Calculate error and return control signal.
        
        Args:
            tracks: List of active Track objects
            frame_width: Width of the current frame
            frame_height: Height of the current frame
            
        Returns:
            GimbalSignal with control offsets
        """
        if self.locked_track_id is None:
            return GimbalSignal()

        # Find the locked track
        target = next((t for t in tracks if t.track_id == self.locked_track_id), None)
        
        if target is None:
            # Target lost
            logger.debug("Locked target %d lost", self.locked_track_id)
            return GimbalSignal(locked=True, track_id=self.locked_track_id)

        # Calculate target center
        x1, y1, x2, y2 = target.bbox
        tx, ty = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        # Calculate normalized error (-1.0 to 1.0)
        # Center is (0.5, 0.5) in normalized coords
        error_x = (tx / frame_width) - 0.5
        error_y = (ty / frame_height) - 0.5
        
        # Apply deadzone
        if abs(error_x) < self.deadzone:
            error_x = 0.0
        if abs(error_y) < self.deadzone:
            error_y = 0.0

        # PID Logic (Simple P for now, can be expanded)
        # We output a signal that would move the gimbal TO center the target
        # If error is positive (target is to the right), signal should be positive (move right)
        sig_x = self.p_gain * error_x
        sig_y = self.p_gain * error_y
        
        # Clamp to [-1.0, 1.0]
        sig_x = max(-1.0, min(1.0, sig_x))
        sig_y = max(-1.0, min(1.0, sig_y))

        return GimbalSignal(
            dx=sig_x,
            dy=sig_y,
            locked=True,
            track_id=self.locked_track_id
        )

    @classmethod
    def from_config(cls, config: dict) -> "GimbalTracker":
        """Load from config dict."""
        cfg = config.get("gimbal", {})
        return cls(
            p_gain=cfg.get("p_gain", 0.5),
            i_gain=cfg.get("i_gain", 0.0),
            d_gain=cfg.get("d_gain", 0.1),
            deadzone=cfg.get("deadzone", 0.02),
        )

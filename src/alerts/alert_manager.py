from __future__ import annotations

"""
Alert Manager Module

Configurable alert system that triggers console alerts, optional
webhook notifications, and file logging when flagged object classes
are detected in the scene.
"""

import json
import logging
import time
from pathlib import Path

# requests is imported lazily inside _webhook_alert() so that the module
# loads cleanly on Raspberry Pi where 'requests' is not installed.

logger = logging.getLogger("garuda.alerts")


class AlertManager:
    """
    Manages detection alerts for flagged object classes.

    Args:
        flagged_classes: Set of class names that trigger alerts.
        cooldown_seconds: Minimum seconds between repeated alerts for the same class.
        webhook_url: Optional URL for webhook POST notifications.
        log_file: Optional file path for alert logging.
    """

    def __init__(
        self,
        flagged_classes: set[str] | None = None,
        cooldown_seconds: float = 5.0,
        webhook_url: str = "",
        log_file: str = "",
    ):
        self.flagged_classes = flagged_classes or set()
        self.cooldown_seconds = cooldown_seconds
        self.webhook_url = webhook_url
        self.log_file = log_file
        self._last_alert_time: dict[str, float] = {}
        self._alert_count = 0

        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    def check_detections(self, detections_or_tracks: list) -> list[dict]:
        """
        Check detections/tracks for flagged classes and trigger alerts.

        Args:
            detections_or_tracks: List of Detection or Track objects.

        Returns:
            List of triggered alert dicts.
        """
        triggered: list[dict] = []
        now = time.time()

        for obj in detections_or_tracks:
            cls_name = obj.class_name
            if cls_name not in self.flagged_classes:
                continue

            # Cooldown check
            last = self._last_alert_time.get(cls_name, 0.0)
            if now - last < self.cooldown_seconds:
                continue

            self._last_alert_time[cls_name] = now
            self._alert_count += 1

            alert = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "class": cls_name,
                "confidence": getattr(obj, "confidence", 0.0),
                "bbox": getattr(obj, "bbox", ()),
                "track_id": getattr(obj, "track_id", None),
                "alert_id": self._alert_count,
            }

            self._console_alert(alert)
            self._log_alert(alert)
            self._webhook_alert(alert)
            triggered.append(alert)

        return triggered

    def _console_alert(self, alert: dict) -> None:
        """Print alert to console."""
        track_info = f" [Track #{alert['track_id']}]" if alert["track_id"] else ""
        logger.warning(
            "🚨 ALERT #%d: '%s' detected (conf=%.2f)%s at %s",
            alert["alert_id"],
            alert["class"],
            alert["confidence"],
            track_info,
            alert["timestamp"],
        )

    def _log_alert(self, alert: dict) -> None:
        """Append alert to log file."""
        if not self.log_file:
            return
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(alert) + "\n")
        except OSError as e:
            logger.error("Failed to write alert log: %s", e)

    def _webhook_alert(self, alert: dict) -> None:
        """Send alert via webhook POST (skipped if 'requests' not installed)."""
        if not self.webhook_url:
            return
        try:
            import requests  # lazy — not available on Raspberry Pi by default
        except ImportError:
            logger.warning(
                "Webhook skipped: 'requests' package not installed. "
                "Install it with: pip install requests"
            )
            self.webhook_url = ""  # disable future attempts
            return
        try:
            resp = requests.post(
                self.webhook_url,
                json=alert,
                timeout=5,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                logger.warning(
                    "Webhook returned status %d for alert #%d",
                    resp.status_code,
                    alert["alert_id"],
                )
        except Exception as e:
            logger.error("Webhook delivery failed: %s", e)

    def add_flagged_class(self, class_name: str) -> None:
        """Add a class to the flagged list at runtime."""
        self.flagged_classes.add(class_name)
        logger.info("Flagged class added: %s", class_name)

    def remove_flagged_class(self, class_name: str) -> None:
        """Remove a class from the flagged list."""
        self.flagged_classes.discard(class_name)
        logger.info("Flagged class removed: %s", class_name)

    def reset(self) -> None:
        """Reset alert state."""
        self._last_alert_time.clear()
        self._alert_count = 0
        logger.info("Alert manager reset.")

    @classmethod
    def from_config(cls, config: dict) -> "AlertManager":
        """Create AlertManager from config dictionary."""
        alert_cfg = config.get("alerts", {})
        return cls(
            flagged_classes=set(alert_cfg.get("flagged_classes", [])),
            cooldown_seconds=alert_cfg.get("cooldown_seconds", 5.0),
            webhook_url=alert_cfg.get("webhook_url", ""),
            log_file=alert_cfg.get("log_file", ""),
        )

    @property
    def total_alerts(self) -> int:
        return self._alert_count

    def __repr__(self) -> str:
        return (
            f"AlertManager(flagged={self.flagged_classes}, "
            f"cooldown={self.cooldown_seconds}s, alerts={self._alert_count})"
        )

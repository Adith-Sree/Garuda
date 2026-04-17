#!/usr/bin/env python3
"""
Test Script — Project Garuda

Tests for module imports, config loading, detector initialisation,
tracker functionality, alert system, and pipeline construction.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestConfigLoading(unittest.TestCase):
    """Test configuration file loading."""

    def test_model_config_exists(self):
        self.assertTrue(Path("configs/model.yaml").exists())

    def test_data_config_exists(self):
        self.assertTrue(Path("configs/data.yaml").exists())

    def test_model_config_parseable(self):
        import yaml
        with open("configs/model.yaml") as f:
            config = yaml.safe_load(f)
        self.assertIn("model", config)
        self.assertIn("training", config)
        self.assertIn("inference", config)
        self.assertIn("tracking", config)
        self.assertIn("alerts", config)

    def test_data_config_parseable(self):
        import yaml
        with open("configs/data.yaml") as f:
            config = yaml.safe_load(f)
        self.assertIn("nc", config)
        self.assertIn("names", config)
        self.assertEqual(config["nc"], 10)


class TestModuleImports(unittest.TestCase):
    """Test that all modules are importable."""

    def test_import_detection(self):
        from src.detection.yolo_detector import YOLODetector, Detection
        det = Detection(bbox=(10, 20, 100, 200), confidence=0.9, class_id=0, class_name="car")
        self.assertEqual(det.class_name, "car")
        self.assertEqual(det.bbox, (10, 20, 100, 200))

    def test_import_tracking(self):
        from src.tracking.tracker import ObjectTracker, Track
        track = Track(track_id=1, bbox=(10, 20, 100, 200), class_name="car")
        self.assertEqual(track.track_id, 1)

    def test_import_alerts(self):
        from src.alerts.alert_manager import AlertManager
        mgr = AlertManager(flagged_classes={"car", "truck"})
        self.assertIn("car", mgr.flagged_classes)

    def test_import_utils(self):
        from src.utils.preprocessing import FramePreprocessor
        from src.utils.visualization import Visualizer
        from src.utils.logger import setup_logger

    def test_import_pipeline(self):
        from src.pipeline.pipeline import DetectionPipeline


class TestDetection(unittest.TestCase):
    """Test Detection dataclass."""

    def test_detection_to_dict(self):
        from src.detection.yolo_detector import Detection
        det = Detection(bbox=(0, 0, 50, 50), confidence=0.85, class_id=3, class_name="car")
        d = det.to_dict()
        self.assertEqual(d["class_name"], "car")
        self.assertAlmostEqual(d["confidence"], 0.85)

    def test_detection_repr(self):
        from src.detection.yolo_detector import Detection
        det = Detection(bbox=(0, 0, 50, 50), confidence=0.85, class_id=3, class_name="car")
        self.assertIn("car", repr(det))


class TestTracker(unittest.TestCase):
    """Test tracker with IOU fallback."""

    def test_tracker_init(self):
        from src.tracking.tracker import ObjectTracker
        tracker = ObjectTracker(max_age=10, n_init=2)
        self.assertIsNotNone(tracker)

    def test_tracker_empty_update(self):
        from src.tracking.tracker import ObjectTracker
        tracker = ObjectTracker(max_age=10, n_init=1)
        tracks = tracker.update([], None)
        self.assertEqual(len(tracks), 0)

    def test_tracker_iou_fallback(self):
        from src.tracking.tracker import ObjectTracker
        from src.detection.yolo_detector import Detection

        tracker = ObjectTracker(max_age=10, n_init=1)
        tracker._use_deepsort = False  # Force IOU fallback

        dets = [
            Detection(bbox=(10, 10, 50, 50), confidence=0.9, class_id=0, class_name="car"),
        ]
        tracks = tracker.update(dets)
        self.assertGreater(len(tracks), 0)
        self.assertEqual(tracks[0].class_name, "car")

    def test_tracker_from_config(self):
        import yaml
        with open("configs/model.yaml") as f:
            config = yaml.safe_load(f)
        from src.tracking.tracker import ObjectTracker
        tracker = ObjectTracker.from_config(config)
        self.assertIsNotNone(tracker)

    def test_tracker_reset(self):
        from src.tracking.tracker import ObjectTracker
        tracker = ObjectTracker()
        tracker.reset()


class TestAlertManager(unittest.TestCase):
    """Test alert system."""

    def test_alert_init(self):
        from src.alerts.alert_manager import AlertManager
        mgr = AlertManager(flagged_classes={"car"}, cooldown_seconds=1.0)
        self.assertEqual(mgr.total_alerts, 0)

    def test_alert_trigger(self):
        from src.alerts.alert_manager import AlertManager
        from src.detection.yolo_detector import Detection

        mgr = AlertManager(flagged_classes={"car"}, cooldown_seconds=0)
        det = Detection(bbox=(0, 0, 50, 50), confidence=0.9, class_id=0, class_name="car")
        alerts = mgr.check_detections([det])
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["class"], "car")

    def test_alert_cooldown(self):
        from src.alerts.alert_manager import AlertManager
        from src.detection.yolo_detector import Detection

        mgr = AlertManager(flagged_classes={"car"}, cooldown_seconds=60)
        det = Detection(bbox=(0, 0, 50, 50), confidence=0.9, class_id=0, class_name="car")
        alerts1 = mgr.check_detections([det])
        alerts2 = mgr.check_detections([det])
        self.assertEqual(len(alerts1), 1)
        self.assertEqual(len(alerts2), 0)  # Cooldown blocks

    def test_alert_unflagged_class(self):
        from src.alerts.alert_manager import AlertManager
        from src.detection.yolo_detector import Detection

        mgr = AlertManager(flagged_classes={"tank"}, cooldown_seconds=0)
        det = Detection(bbox=(0, 0, 50, 50), confidence=0.9, class_id=0, class_name="car")
        alerts = mgr.check_detections([det])
        self.assertEqual(len(alerts), 0)

    def test_add_remove_flagged(self):
        from src.alerts.alert_manager import AlertManager
        mgr = AlertManager(flagged_classes=set())
        mgr.add_flagged_class("drone")
        self.assertIn("drone", mgr.flagged_classes)
        mgr.remove_flagged_class("drone")
        self.assertNotIn("drone", mgr.flagged_classes)

    def test_from_config(self):
        import yaml
        with open("configs/model.yaml") as f:
            config = yaml.safe_load(f)
        from src.alerts.alert_manager import AlertManager
        mgr = AlertManager.from_config(config)
        self.assertIsNotNone(mgr)


class TestPreprocessor(unittest.TestCase):
    """Test frame preprocessing."""

    def test_resize(self):
        from src.utils.preprocessing import FramePreprocessor
        pp = FramePreprocessor(target_width=320, target_height=320, letterbox=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = pp.resize(frame)
        self.assertEqual(resized.shape[:2], (320, 320))

    def test_letterbox(self):
        from src.utils.preprocessing import FramePreprocessor
        pp = FramePreprocessor(target_width=640, target_height=640, letterbox=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pp.resize(frame)
        self.assertEqual(result.shape[:2], (640, 640))

    def test_normalize(self):
        from src.utils.preprocessing import FramePreprocessor
        pp = FramePreprocessor(target_width=32, target_height=32, normalize=True)
        frame = np.full((32, 32, 3), 255, dtype=np.uint8)
        result = pp.preprocess(frame)
        self.assertAlmostEqual(result.max(), 1.0, places=2)


class TestVisualizer(unittest.TestCase):
    """Test visualization module."""

    def test_draw_fps(self):
        from src.utils.visualization import Visualizer
        vis = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = vis.draw_fps(frame, 25.0)
        self.assertEqual(result.shape, frame.shape)

    def test_draw_detections(self):
        from src.utils.visualization import Visualizer
        from src.detection.yolo_detector import Detection
        vis = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [Detection(bbox=(10, 10, 100, 100), confidence=0.9, class_id=0, class_name="car")]
        result = vis.draw_detections(frame, dets)
        self.assertEqual(result.shape, frame.shape)

    def test_draw_tracks(self):
        from src.utils.visualization import Visualizer
        from src.tracking.tracker import Track
        vis = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = [Track(track_id=1, bbox=(10, 10, 100, 100), class_name="car", confidence=0.9)]
        result = vis.draw_tracks(frame, tracks)
        self.assertEqual(result.shape, frame.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)

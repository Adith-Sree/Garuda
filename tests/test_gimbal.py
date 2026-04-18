import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tracking.gimbal_tracker import GimbalTracker
from src.tracking.tracker import Track

class TestGimbalTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = GimbalTracker(p_gain=1.0, deadzone=0.0)
        self.frame_w = 1280
        self.frame_h = 720

    def test_no_lock(self):
        signal = self.tracker.update([], self.frame_w, self.frame_h)
        self.assertFalse(signal.locked)
        self.assertEqual(signal.dx, 0.0)

    def test_lock_target(self):
        tracks = [
            Track(track_id=1, bbox=(0, 0, 100, 100)),
            Track(track_id=2, bbox=(600, 300, 700, 400))
        ]
        self.tracker.lock_target(2)
        signal = self.tracker.update(tracks, 1280, 720)
        
        self.assertTrue(signal.locked)
        self.assertEqual(signal.track_id, 2)
        
        # Center of track 2 is (650, 350)
        # Normalized error: (650/1280 - 0.5), (350/720 - 0.5)
        # 650/1280 = 0.5078125 -> dx = 0.0078125
        # 350/720 = 0.4861111 -> dy = -0.0138889
        self.assertAlmostEqual(signal.dx, 0.0078125)
        self.assertAlmostEqual(signal.dy, -0.0138889)

    def test_target_lost(self):
        self.tracker.lock_target(1)
        signal = self.tracker.update([], self.frame_w, self.frame_h)
        self.assertTrue(signal.locked)
        self.assertEqual(signal.dx, 0.0)
        self.assertEqual(signal.dy, 0.0)

if __name__ == "__main__":
    unittest.main()

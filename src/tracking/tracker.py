from __future__ import annotations

"""
Object Tracker Module

DeepSORT-based multi-object tracker for persistent ID assignment
across frames. Falls back to a simple IOU-based tracker when
deep_sort_realtime is unavailable.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("garuda.tracking")


@dataclass
class Track:
    """Represents a tracked object."""

    track_id: int
    bbox: tuple[int, int, int, int]         # (x1, y1, x2, y2)
    class_name: str = ""
    confidence: float = 0.0
    age: int = 0                             # frames since first seen
    hits: int = 0                            # total detection hits
    time_since_update: int = 0               # frames since last matched


class ObjectTracker:
    """
    Multi-object tracker using DeepSORT with IOU fallback.

    Args:
        max_age: Maximum frames to keep a lost track alive.
        n_init: Minimum consecutive detections to confirm a track.
        max_iou_distance: Maximum IOU distance for matching.
        max_cosine_distance: Maximum cosine distance for appearance matching.
        nn_budget: Maximum size of the appearance feature gallery.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.3,
        nn_budget: int | None = 100,
    ):
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self._tracker = None
        self._use_deepsort = False
        self._fallback_tracks: dict[int, Track] = {}
        self._next_id = 1

        self._init_tracker()

    def _init_tracker(self) -> None:
        """Initialize the DeepSORT tracker, falling back to IOU-based."""
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort

            self._tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance,
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=self.nn_budget,
            )
            self._use_deepsort = True
            logger.info("DeepSORT tracker initialised.")
        except ImportError:
            logger.warning(
                "deep_sort_realtime not installed — using simple IOU tracker."
            )
            self._use_deepsort = False

    def update(
        self,
        detections: list,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of Detection objects (bbox, confidence, class_name).
            frame: Current BGR frame (required for DeepSORT appearance features).

        Returns:
            List of active Track objects with assigned IDs.
        """
        if self._use_deepsort:
            return self._update_deepsort(detections, frame)
        return self._update_iou(detections)

    # -------------------------------------------------------------------
    # DeepSORT path
    # -------------------------------------------------------------------
    def _update_deepsort(
        self, detections: list, frame: np.ndarray | None
    ) -> list[Track]:
        """Update using DeepSORT."""
        if not detections:
            self._tracker.tracker.predict()
            self._tracker.tracker.update([])
            return self._get_deepsort_tracks()

        # DeepSORT expects [[x1, y1, w, h], conf, class_name]
        raw_dets = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w, h = x2 - x1, y2 - y1
            raw_dets.append(([x1, y1, w, h], det.confidence, det.class_name))

        self._tracker.update_tracks(raw_dets, frame=frame)
        return self._get_deepsort_tracks()

    def _get_deepsort_tracks(self) -> list[Track]:
        """Extract confirmed tracks from DeepSORT."""
        tracks: list[Track] = []
        for trk in self._tracker.tracker.tracks:
            if not trk.is_confirmed():
                continue
            bbox = trk.to_ltrb()
            det_class = trk.det_class if hasattr(trk, "det_class") else ""
            det_conf = trk.det_conf if hasattr(trk, "det_conf") else 0.0
            tracks.append(
                Track(
                    track_id=trk.track_id,
                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    class_name=det_class or "",
                    confidence=float(det_conf) if det_conf else 0.0,
                    age=trk.age,
                    hits=trk.hits,
                    time_since_update=trk.time_since_update,
                )
            )
        return tracks

    # -------------------------------------------------------------------
    # Simple IOU fallback
    # -------------------------------------------------------------------
    def _update_iou(self, detections: list) -> list[Track]:
        """Simple IOU-based tracking fallback."""
        if not detections:
            # Age out old tracks
            to_remove = []
            for tid, trk in self._fallback_tracks.items():
                trk.time_since_update += 1
                if trk.time_since_update > self.max_age:
                    to_remove.append(tid)
            for tid in to_remove:
                del self._fallback_tracks[tid]
            return list(self._fallback_tracks.values())

        # Match detections to existing tracks by IOU
        det_bboxes = [d.bbox for d in detections]
        trk_ids = list(self._fallback_tracks.keys())
        trk_bboxes = [self._fallback_tracks[t].bbox for t in trk_ids]

        matched_det = set()
        matched_trk = set()

        if trk_bboxes and det_bboxes:
            iou_matrix = self._compute_iou_matrix(det_bboxes, trk_bboxes)
            for _ in range(min(len(det_bboxes), len(trk_bboxes))):
                idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if iou_matrix[idx] < (1.0 - self.max_iou_distance):
                    break
                d_idx, t_idx = idx
                matched_det.add(d_idx)
                matched_trk.add(t_idx)

                tid = trk_ids[t_idx]
                det = detections[d_idx]
                self._fallback_tracks[tid].bbox = det.bbox
                self._fallback_tracks[tid].confidence = det.confidence
                self._fallback_tracks[tid].class_name = det.class_name
                self._fallback_tracks[tid].hits += 1
                self._fallback_tracks[tid].time_since_update = 0
                self._fallback_tracks[tid].age += 1

                iou_matrix[d_idx, :] = -1
                iou_matrix[:, t_idx] = -1

        # Unmatched detections → new tracks
        for i, det in enumerate(detections):
            if i not in matched_det:
                self._fallback_tracks[self._next_id] = Track(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    hits=1,
                )
                self._next_id += 1

        # Unmatched tracks → age them
        for j, tid in enumerate(trk_ids):
            if j not in matched_trk:
                self._fallback_tracks[tid].time_since_update += 1

        # Remove dead tracks
        to_remove = [
            tid
            for tid, trk in self._fallback_tracks.items()
            if trk.time_since_update > self.max_age
        ]
        for tid in to_remove:
            del self._fallback_tracks[tid]

        return [
            trk
            for trk in self._fallback_tracks.values()
            if trk.hits >= self.n_init
        ]

    @staticmethod
    def _compute_iou_matrix(
        bboxes_a: list[tuple], bboxes_b: list[tuple]
    ) -> np.ndarray:
        """Compute IOU matrix between two lists of bounding boxes."""
        a = np.array(bboxes_a, dtype=float)
        b = np.array(bboxes_b, dtype=float)
        m, n = len(a), len(b)
        iou = np.zeros((m, n), dtype=float)

        for i in range(m):
            for j in range(n):
                x1 = max(a[i, 0], b[j, 0])
                y1 = max(a[i, 1], b[j, 1])
                x2 = min(a[i, 2], b[j, 2])
                y2 = min(a[i, 3], b[j, 3])

                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_a = (a[i, 2] - a[i, 0]) * (a[i, 3] - a[i, 1])
                area_b = (b[j, 2] - b[j, 0]) * (b[j, 3] - b[j, 1])
                union = area_a + area_b - inter
                iou[i, j] = inter / union if union > 0 else 0.0

        return iou

    def reset(self) -> None:
        """Reset all tracks."""
        if self._use_deepsort and self._tracker:
            self._init_tracker()
        self._fallback_tracks.clear()
        self._next_id = 1
        logger.info("Tracker reset.")

    @classmethod
    def from_config(cls, config: dict) -> "ObjectTracker":
        """Create tracker from config dictionary."""
        trk_cfg = config.get("tracking", {})
        return cls(
            max_age=trk_cfg.get("max_age", 30),
            n_init=trk_cfg.get("n_init", 3),
            max_iou_distance=trk_cfg.get("max_iou_distance", 0.7),
            max_cosine_distance=trk_cfg.get("max_cosine_distance", 0.3),
            nn_budget=trk_cfg.get("nn_budget", 100),
        )

    def __repr__(self) -> str:
        backend = "DeepSORT" if self._use_deepsort else "IOU-Fallback"
        return f"ObjectTracker(backend={backend}, max_age={self.max_age})"

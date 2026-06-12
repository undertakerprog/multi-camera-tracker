"""Disabled detector used as the default placeholder.

Keeps the pipeline detector-aware without loading any model. Returns no
detections and reports itself as disabled so the app never calls it.
"""

from __future__ import annotations

import numpy as np

from src.detection.base import Detection, Detector


class NullDetector(Detector):
    detect_interval = 1

    @property
    def enabled(self) -> bool:
        return False

    def detect(self, image: np.ndarray) -> list[Detection]:
        return []

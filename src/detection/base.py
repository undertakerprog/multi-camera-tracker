"""Object-detection interface.

Detection is independent of tracking, UI and the camera implementation. A
detector takes a frame image and returns axis-aligned boxes. It is meant to run
once every ``N`` frames; between detections the tracker and Kalman filter carry
the target. On the current stage no real detector ships -- see
:class:`~src.detection.null_detector.NullDetector` (disabled) and
:mod:`src.detection.tensorrt_detector` (integration point for TensorRT/YOLO).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Detection:
    """A single detected object in full-frame pixel coordinates."""

    bbox: tuple[int, int, int, int]
    score: float = 1.0
    class_id: int = -1
    label: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2


class Detector(ABC):
    """Common interface for all detector implementations."""

    #: run the detector at most once per this many frames
    detect_interval: int = 1

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether this detector should be consulted at all."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return detections for ``image`` (empty list if none)."""

    def should_run(self, frame_index: int) -> bool:
        """True when the detector is due to run on ``frame_index``."""
        if not self.enabled:
            return False
        interval = max(1, self.detect_interval)
        return frame_index % interval == 0

    def close(self) -> None:
        """Release any model/runtime resources. Default is a no-op."""

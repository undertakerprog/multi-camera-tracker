"""Integration point for a TensorRT/YOLO detector on Jetson.

This is a deliberate stub: no model is bundled and the class is disabled until a
concrete engine and a known set of target classes are provided. It documents the
contract a real detector must satisfy and where to wire the TensorRT runtime so
the rest of the pipeline does not change when detection is added (Stage 4 of the
project roadmap).

Do not load an arbitrary model here without knowing the target object classes.
"""

from __future__ import annotations

import logging

import numpy as np

from src.detection.base import Detection, Detector

LOG = logging.getLogger(__name__)


class TensorRTDetector(Detector):
    def __init__(
        self,
        engine_path: str | None = None,
        detect_interval: int = 10,
        score_threshold: float = 0.4,
        class_labels: list[str] | None = None,
    ) -> None:
        self.engine_path = engine_path
        self.detect_interval = max(1, detect_interval)
        self.score_threshold = score_threshold
        self.class_labels = class_labels or []
        self._engine = None  # populated by load()

    @property
    def enabled(self) -> bool:
        # Enabled only once a real engine has been loaded.
        return self._engine is not None

    def load(self) -> None:
        """Load the TensorRT engine. Intentionally not implemented yet."""
        raise NotImplementedError(
            "TensorRTDetector.load is a stub. Provide a TensorRT engine and the "
            "target class labels, then implement engine loading and inference."
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        if self._engine is None:
            return []
        raise NotImplementedError(
            "TensorRTDetector.detect is a stub: implement preprocessing, "
            "inference and post-processing into Detection objects."
        )

    def close(self) -> None:
        self._engine = None

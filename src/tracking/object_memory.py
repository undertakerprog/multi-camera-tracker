"""Persistent appearance memory of the tracked object.

The memory keeps two kinds of templates:

* a **stable** template captured when the operator selects the ROI. It is the
  ground-truth appearance and is never overwritten by tracking, so a single bad
  frame cannot poison re-acquisition.
* a bounded bank of **adaptive** templates added only from confidently confirmed
  observations. They let re-acquire cope with slow appearance/pose changes while
  the stable template guards against drift.

Each template stores its grayscale crop, its size and pre-computed ORB
descriptors so global feature search does not recompute them every call.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class TemplateEntry:
    image: np.ndarray
    size: tuple[int, int]
    keypoints: tuple
    descriptors: np.ndarray | None
    stable: bool


class ObjectMemory:
    def __init__(
        self,
        max_templates: int = 4,
        orb_features: int = 1200,
        min_template_size: int = 6,
    ) -> None:
        self.max_templates = max(1, max_templates)
        self.min_template_size = max(2, min_template_size)
        self._orb = cv2.ORB_create(nfeatures=orb_features)
        self._stable: TemplateEntry | None = None
        self._bank: deque[TemplateEntry] = deque(maxlen=self.max_templates)

    @property
    def has_memory(self) -> bool:
        return self._stable is not None

    @property
    def stable(self) -> TemplateEntry | None:
        return self._stable

    @property
    def stable_size(self) -> tuple[int, int] | None:
        return self._stable.size if self._stable is not None else None

    def set_stable(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        """Record the ground-truth template at selection time. Clears the bank."""
        entry = self._make_entry(image, bbox, stable=True)
        if entry is None:
            return False
        self._stable = entry
        self._bank.clear()
        return True

    def remember(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        """Add a confirmed adaptive template (bounded, oldest dropped)."""
        entry = self._make_entry(image, bbox, stable=False)
        if entry is None:
            return False
        self._bank.append(entry)
        return True

    def entries(self) -> list[TemplateEntry]:
        """Templates to match against: stable first, newest adaptive next."""
        result: list[TemplateEntry] = []
        if self._stable is not None:
            result.append(self._stable)
        result.extend(reversed(self._bank))
        return result

    def stable_descriptors(self):
        if self._stable is None:
            return None, None
        return self._stable.keypoints, self._stable.descriptors

    def clear(self) -> None:
        self._stable = None
        self._bank.clear()

    # -- internal ----------------------------------------------------------
    def _make_entry(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        stable: bool,
    ) -> TemplateEntry | None:
        x, y, w, h = self._clip_bbox(bbox, image.shape)
        if w < self.min_template_size or h < self.min_template_size:
            return None
        crop = self._to_gray(image[y : y + h, x : x + w]).copy()
        keypoints, descriptors = self._orb.detectAndCompute(crop, None)
        return TemplateEntry(
            image=crop,
            size=(w, h),
            keypoints=keypoints if keypoints is not None else (),
            descriptors=descriptors,
            stable=stable,
        )

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _clip_bbox(
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x, y, w, h = bbox
        x1 = max(0, min(width - 1, x))
        y1 = max(0, min(height - 1, y))
        x2 = max(0, min(width, x + w))
        y2 = max(0, min(height, y + h))
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

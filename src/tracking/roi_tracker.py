"""Dynamic-ROI wrapper around an OpenCV tracker.

CSRT/KCF cost scales with the image area they process. Running them on a small
search window around the last known position (or the Kalman prediction) instead
of the full 1280x960 frame is the single biggest speed-up for this pipeline.

``DynamicROITracker`` keeps a window in full-frame coordinates, feeds the
underlying :class:`OpenCVObjectTracker` only the cropped window, and maps the
result back to full-frame coordinates. The window is re-centered (and the inner
tracker re-initialised) only when the target approaches the window border, so
the inner tracker keeps a stable coordinate system and its learned filter state
between re-centers.
"""

from __future__ import annotations

import logging

import numpy as np

from src.tracking.opencv_tracker import OpenCVObjectTracker, TrackerError, TrackingResult

LOG = logging.getLogger(__name__)


class DynamicROITracker:
    def __init__(
        self,
        tracker_type: str = "CSRT",
        tracking_scale: float = 1.0,
        roi_scale: float = 2.5,
        edge_margin: float = 0.15,
        min_window: int = 64,
        inner: OpenCVObjectTracker | None = None,
    ) -> None:
        if roi_scale < 1.0:
            raise ValueError("roi_scale must be >= 1.0")
        if not 0.0 <= edge_margin < 0.5:
            raise ValueError("edge_margin must be in [0.0, 0.5)")

        self._inner = inner or OpenCVObjectTracker(tracker_type, tracking_scale)
        self.roi_scale = roi_scale
        self.edge_margin = edge_margin
        self.min_window = max(8, int(min_window))
        self._window: tuple[int, int, int, int] | None = None
        self._active = False
        self._recenters = 0

    @property
    def name(self) -> str:
        return f"{self._inner.name} roi x{self.roi_scale:g}"

    @property
    def initialized(self) -> bool:
        return self._active and self._inner.initialized

    @property
    def window(self) -> tuple[int, int, int, int] | None:
        return self._window

    @property
    def recenters(self) -> int:
        return self._recenters

    def initialize(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        self.reset()
        self._window = self._window_for(bbox, image.shape)
        self._init_inner(image, bbox)
        self._active = True

    def update(self, image: np.ndarray) -> TrackingResult:
        if not self.initialized or self._window is None:
            return TrackingResult(ok=False)

        wx, wy, ww, wh = self._window
        crop = image[wy : wy + wh, wx : wx + ww]
        if crop.size == 0:
            self.reset()
            return TrackingResult(ok=False)

        result = self._inner.update(crop)
        if not result.ok or result.bbox is None:
            return TrackingResult(ok=False)

        lx, ly, lw, lh = result.bbox
        full_bbox = self._clip_bbox((wx + lx, wy + ly, lw, lh), image.shape)
        if full_bbox is None:
            self.reset()
            return TrackingResult(ok=False)

        if self._near_edge(result.bbox, self._window):
            self._recenter(image, full_bbox)

        return TrackingResult(ok=True, bbox=full_bbox)

    def reset(self) -> None:
        self._inner.reset()
        self._window = None
        self._active = False

    # -- internal ----------------------------------------------------------
    def _init_inner(self, image: np.ndarray, full_bbox: tuple[int, int, int, int]) -> None:
        assert self._window is not None
        wx, wy, ww, wh = self._window
        crop = image[wy : wy + wh, wx : wx + ww]
        if crop.size == 0:
            raise TrackerError("Dynamic ROI produced an empty crop")
        local = self._clip_bbox(
            (full_bbox[0] - wx, full_bbox[1] - wy, full_bbox[2], full_bbox[3]),
            crop.shape,
        )
        if local is None:
            raise TrackerError("Target bbox does not overlap the dynamic ROI")
        self._inner.initialize(crop, local)

    def _recenter(self, image: np.ndarray, full_bbox: tuple[int, int, int, int]) -> None:
        try:
            new_window = self._window_for(full_bbox, image.shape)
            self._window = new_window
            self._init_inner(image, full_bbox)
            self._recenters += 1
        except TrackerError as exc:
            # Re-init failed: drop to "not initialized" so the app's reacquire
            # path takes over instead of silently tracking a stale window.
            LOG.warning("ROI re-center failed, handing off to reacquire: %s", exc)
            self.reset()

    def _window_for(
        self,
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        if len(frame_shape) < 2:
            raise TrackerError("Cannot build dynamic ROI for an empty frame")
        height, width = frame_shape[:2]
        if height <= 0 or width <= 0:
            raise TrackerError("Cannot build dynamic ROI for an empty frame")
        clipped = self._clip_bbox(bbox, frame_shape)
        if clipped is None:
            raise TrackerError("Target bbox is outside the frame")
        x, y, w, h = clipped
        cx = x + w / 2.0
        cy = y + h / 2.0
        ww = min(width, max(self.min_window, int(round(w * self.roi_scale))))
        wh = min(height, max(self.min_window, int(round(h * self.roi_scale))))
        wx = int(round(cx - ww / 2.0))
        wy = int(round(cy - wh / 2.0))
        wx = max(0, min(width - ww, wx))
        wy = max(0, min(height - wh, wy))
        return wx, wy, ww, wh

    def _near_edge(
        self,
        local_bbox: tuple[int, int, int, int],
        window: tuple[int, int, int, int],
    ) -> bool:
        lx, ly, lw, lh = local_bbox
        _, _, ww, wh = window
        margin_x = self.edge_margin * ww
        margin_y = self.edge_margin * wh
        return (
            lx < margin_x
            or ly < margin_y
            or (lx + lw) > (ww - margin_x)
            or (ly + lh) > (wh - margin_y)
        )

    @staticmethod
    def _clip_bbox(
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:
        if len(frame_shape) < 2:
            return None
        height, width = frame_shape[:2]
        if height <= 0 or width <= 0:
            return None
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return x1, y1, x2 - x1, y2 - y1

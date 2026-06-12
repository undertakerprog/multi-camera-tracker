import numpy as np

from tests._util import raises
from src.tracking.opencv_tracker import TrackerError, TrackingResult
from src.tracking.roi_tracker import DynamicROITracker


class FakeInner:
    """Stand-in for OpenCVObjectTracker returning a scripted local bbox."""

    def __init__(self) -> None:
        self._init = False
        self._bbox = None
        self.next_local = None
        self.name = "fake"

    @property
    def initialized(self) -> bool:
        return self._init

    def initialize(self, image, bbox) -> None:
        self._init = True
        self._bbox = bbox

    def update(self, image) -> TrackingResult:
        bbox = self.next_local if self.next_local is not None else self._bbox
        return TrackingResult(ok=True, bbox=bbox)

    def reset(self) -> None:
        self._init = False
        self._bbox = None


def _frame():
    return np.zeros((400, 400), dtype=np.uint8)


def test_local_to_full_coordinate_mapping():
    inner = FakeInner()
    tracker = DynamicROITracker(roi_scale=2.5, edge_margin=0.15, inner=inner)
    tracker.initialize(_frame(), (190, 190, 20, 20))

    result = tracker.update(_frame())
    assert result.ok
    # Centered window keeps the object where it was, in full-frame coords.
    assert result.bbox == (190, 190, 20, 20)
    assert tracker.recenters == 0


def test_recenter_when_target_near_edge():
    inner = FakeInner()
    tracker = DynamicROITracker(roi_scale=2.5, edge_margin=0.15, inner=inner)
    tracker.initialize(_frame(), (190, 190, 20, 20))

    # Push the reported local bbox to the window's far edge.
    inner.next_local = (40, 40, 20, 20)
    result = tracker.update(_frame())
    assert result.ok
    assert tracker.recenters == 1
    # The new window must be re-centred on the mapped full-frame bbox.
    wx, wy, ww, wh = tracker.window
    cx, cy = wx + ww / 2, wy + wh / 2
    assert abs(cx - (result.bbox[0] + result.bbox[2] / 2)) <= 1
    assert abs(cy - (result.bbox[1] + result.bbox[3] / 2)) <= 1


def test_reset_clears_state():
    inner = FakeInner()
    tracker = DynamicROITracker(inner=inner)
    tracker.initialize(_frame(), (10, 10, 20, 20))
    tracker.reset()
    assert not tracker.initialized
    assert tracker.window is None
    assert tracker.update(_frame()).ok is False


def test_initialize_rejects_bbox_outside_frame():
    tracker = DynamicROITracker(inner=FakeInner())
    with raises(TrackerError):
        tracker.initialize(_frame(), (500, 500, 20, 20))


def test_update_rejects_inner_bbox_outside_crop_without_crashing():
    inner = FakeInner()
    tracker = DynamicROITracker(inner=inner)
    tracker.initialize(_frame(), (190, 190, 20, 20))
    inner.next_local = (1000, 1000, 20, 20)

    result = tracker.update(_frame())

    assert not result.ok
    assert not tracker.initialized
    assert tracker.window is None

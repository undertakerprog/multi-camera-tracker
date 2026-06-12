import numpy as np

from tests._util import raises
from src.tracking.opencv_tracker import OpenCVObjectTracker, TrackerError


def test_initialize_rejects_empty_image_before_creating_tracker():
    tracker = OpenCVObjectTracker("KCF", tracking_scale=0.75)
    with raises(TrackerError):
        tracker.initialize(np.empty((0, 0), dtype=np.uint8), (0, 0, 20, 20))


def test_clip_bbox_rejects_non_overlapping_box():
    assert OpenCVObjectTracker._clip_bbox((500, 500, 20, 20), (100, 100)) is None


def test_scaled_bbox_is_clipped_to_resized_frame():
    bbox = OpenCVObjectTracker._clip_bbox((96, 96, 8, 8), (100, 100))
    assert bbox == (96, 96, 4, 4)

    scaled = OpenCVObjectTracker._scale_bbox(bbox, 0.75)
    fitted = OpenCVObjectTracker._clip_bbox(scaled, (75, 75))
    assert fitted is not None
    x, y, w, h = fitted
    assert x >= 0 and y >= 0
    assert x + w <= 75 and y + h <= 75


def test_kcf_prepares_three_channel_image_from_mono8():
    tracker = OpenCVObjectTracker("KCF", tracking_scale=0.75)
    prepared = tracker._prepare_tracking_image(np.zeros((40, 60), dtype=np.uint8))
    assert prepared.shape == (30, 45, 3)
    assert prepared.flags.c_contiguous

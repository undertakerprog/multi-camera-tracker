import numpy as np

from src.tracking import states
from src.tracking.opencv_tracker import TrackingResult
from src.tracking.target_state import TargetState

SHAPE = (480, 640)


def _ok(bbox):
    return TrackingResult(ok=True, bbox=bbox)


def test_initialize_is_confirmed_tracking():
    state = TargetState()
    update = state.initialize((100, 100, 40, 40))
    assert update.status == states.TRACKING
    assert update.confirmed
    assert update.confirmations == 1


def test_confirmations_accumulate_then_reset_on_miss():
    state = TargetState(smoothing_alpha=1.0)
    state.initialize((100, 100, 40, 40))
    state.update(_ok((101, 101, 40, 40)), SHAPE)
    update = state.update(_ok((102, 102, 40, 40)), SHAPE)
    assert update.confirmations >= 2

    missed = state.update(TrackingResult(ok=False), SHAPE)
    assert missed.status == states.PREDICTING
    assert missed.confirmations == 0


def test_expires_after_max_lost_frames():
    state = TargetState(max_lost_frames=3)
    state.initialize((100, 100, 40, 40))
    last = None
    for _ in range(5):
        last = state.update(TrackingResult(ok=False), SHAPE)
    assert last.status == states.TARGET_STALE
    assert last.expired


def test_mark_lost_resets_confirmations():
    state = TargetState()
    state.initialize((100, 100, 40, 40))
    state.update(_ok((100, 100, 40, 40)), SHAPE)
    state.mark_lost()
    assert state.confirmations == 0


def test_reset_clears_everything():
    state = TargetState()
    state.initialize((100, 100, 40, 40))
    state.reset()
    assert state.bbox is None
    assert state.lost_frames == 0
    assert state.confirmations == 0

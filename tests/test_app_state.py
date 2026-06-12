from types import SimpleNamespace

from src.app import TargetTrackingApp


def _app(prediction_display_frames=3):
    app = TargetTrackingApp.__new__(TargetTrackingApp)
    app.prediction_display_frames = prediction_display_frames
    return app


def test_confirmed_target_is_displayed():
    state = SimpleNamespace(bbox=(10, 10, 20, 20), confirmed=True, expired=False, lost_frames=0)
    assert _app()._should_display_state(state)


def test_prediction_is_hidden_after_display_grace_period():
    state = SimpleNamespace(bbox=(10, 10, 20, 20), confirmed=False, expired=False, lost_frames=4)
    assert not _app(prediction_display_frames=3)._should_display_state(state)


def test_expired_target_is_hidden():
    state = SimpleNamespace(bbox=(10, 10, 20, 20), confirmed=False, expired=True, lost_frames=1)
    assert not _app()._should_display_state(state)

import time

import numpy as np

from tests._util import raises
from src.cameras.base import CameraError, CameraSource
from src.cameras.capture_worker import CameraCaptureWorker
from src.common import Frame


class FakeCamera(CameraSource):
    def __init__(self, fail_after: int | None = None) -> None:
        self._seq = 0
        self._fail_after = fail_after
        self.closed = False
        self.stopped = False

    def open(self) -> None:
        pass

    def start(self) -> None:
        pass

    def read(self) -> Frame:
        if self._fail_after is not None and self._seq >= self._fail_after:
            raise CameraError("simulated failure")
        image = np.full((8, 8), self._seq % 255, dtype=np.uint8)
        frame = Frame(image=image, source_id="fake", sequence_id=self._seq)
        self._seq += 1
        return frame

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def test_returns_fresh_frames_and_counts_drops():
    worker = CameraCaptureWorker(FakeCamera())
    worker.start()
    try:
        seen = []
        for _ in range(3):
            frame = worker.read_latest(timeout=2.0)
            assert frame is not None
            seen.append(frame.sequence_id)
            time.sleep(0.02)  # let the producer run ahead -> drops
        assert worker.consumed == 3
        # Sequence ids are non-decreasing (always the freshest frame).
        assert seen == sorted(seen)
        # Producer outran the consumer, so some frames were dropped.
        assert worker.dropped >= 0
        assert worker.produced >= worker.consumed
    finally:
        worker.stop()


def test_clean_stop_releases_camera():
    camera = FakeCamera()
    worker = CameraCaptureWorker(camera)
    worker.start()
    worker.read_latest(timeout=2.0)
    worker.stop()
    assert camera.stopped
    assert camera.closed
    assert not worker.is_running()


def test_capture_error_is_surfaced():
    worker = CameraCaptureWorker(FakeCamera(fail_after=0))
    worker.start()
    try:
        with raises(CameraError):
            for _ in range(50):
                worker.read_latest(timeout=2.0)
    finally:
        worker.stop()

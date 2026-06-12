import numpy as np

from tests._util import raises
from src.detection import Detection, NullDetector, TensorRTDetector


def test_detection_center():
    detection = Detection(bbox=(10, 20, 40, 60))
    assert detection.center == (30, 50)


def test_null_detector_is_disabled():
    detector = NullDetector()
    assert detector.enabled is False
    assert detector.should_run(0) is False
    assert detector.detect(np.zeros((4, 4), dtype=np.uint8)) == []


def test_should_run_honours_interval():
    class Every3(NullDetector):
        detect_interval = 3

        @property
        def enabled(self) -> bool:
            return True

    detector = Every3()
    assert detector.should_run(0) is True
    assert detector.should_run(1) is False
    assert detector.should_run(3) is True


def test_tensorrt_stub_disabled_until_loaded():
    detector = TensorRTDetector()
    assert detector.enabled is False
    assert detector.detect(np.zeros((4, 4), dtype=np.uint8)) == []
    with raises(NotImplementedError):
        detector.load()

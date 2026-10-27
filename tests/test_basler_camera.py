from types import SimpleNamespace

import numpy as np

from src.cameras.basler_camera import BaslerCameraSource


class FakeNode:
    def __init__(self, name, value, changes):
        self.name = name
        self.value = value
        self.changes = changes

    def SetValue(self, value):
        self.value = value
        self.changes.append((self.name, value))


def _configured_source(**settings):
    source = BaslerCameraSource(**settings)
    changes = []
    values = {
        "PixelFormat": "Mono8",
        "ExposureMode": "Timed",
        "ExposureAuto": "Off",
        "ExposureTime": 5000,
        "AutoExposureTimeUpperLimit": 100000,
        "GainAuto": "Off",
        "Gain": 0,
    }
    source._camera = SimpleNamespace(**{
        name: FakeNode(name, value, changes) for name, value in values.items()
    })
    source._apply_settings()
    return source, changes


def test_numeric_exposure_and_gain_do_not_disable_requested_auto_modes():
    source, changes = _configured_source(
        exposure_us=10000, gain_db=12, exposure_auto="Once", gain_auto="Continuous",
        auto_exposure_max_us=30000,
    )
    assert source._camera.ExposureAuto.value == "Once"
    assert source._camera.GainAuto.value == "Continuous"
    assert source._camera.AutoExposureTimeUpperLimit.value == 30000
    assert changes.index(("ExposureTime", 10000.0)) < changes.index(("ExposureAuto", "Once"))
    assert changes.index(("Gain", 12.0)) < changes.index(("GainAuto", "Continuous"))


def test_manual_camera_settings_keep_auto_functions_off():
    source, _ = _configured_source(exposure_us=10000, gain_db=12)
    assert source._camera.ExposureAuto.value == "Off"
    assert source._camera.GainAuto.value == "Off"
    assert source._camera.ExposureTime.value == 10000
    assert source._camera.Gain.value == 12


def test_rotate_180_preserves_mono8_values_without_modifying_input():
    image = np.array([[0, 1, 2], [3, 4, 255]], dtype=np.uint8)
    original = image.copy()
    source = BaslerCameraSource(rotate_180=True)
    result = source._to_opencv_image(image)
    np.testing.assert_array_equal(result, [[255, 4, 3], [2, 1, 0]])
    np.testing.assert_array_equal(image, original)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous


def test_mono8_without_rotation_preserves_orientation_and_brightness():
    image = np.array([[0, 1, 2], [3, 4, 255]], dtype=np.uint8)
    result = BaslerCameraSource()._to_opencv_image(image)
    np.testing.assert_array_equal(result, image)


def test_camera_config_reaches_source():
    from main import build_camera, build_parser

    config = {"camera": {
        "rotate_180": True, "auto_exposure_max_us": 30000,
        "exposure_auto": "Once", "gain_auto": "Once", "exposure_us": 10000,
    }}
    args = build_parser(config, "balanced", parents=[]).parse_args([])
    source = build_camera(args)
    assert source.rotate_180
    assert source.auto_exposure_max_us == 30000
    assert source.exposure_auto == "Once"
    assert source.gain_auto == "Once"
    assert source.exposure_us == 10000

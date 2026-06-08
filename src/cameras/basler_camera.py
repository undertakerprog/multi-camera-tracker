from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.cameras.base import CameraError, CameraSource
from src.common import Frame


class BaslerCameraSource(CameraSource):
    """Camera source backed by Basler pylon/pypylon."""

    def __init__(
        self,
        serial_number: str | None = None,
        source_id: str = "basler-0",
        exposure_us: float | None = None,
        gain_db: float | None = None,
        pixel_format: str = "Mono8",
        timeout_ms: int = 5000,
    ) -> None:
        self.serial_number = serial_number
        self.source_id = source_id
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.pixel_format = pixel_format
        self.timeout_ms = timeout_ms

        self._pylon: Any | None = None
        self._camera: Any | None = None
        self._sequence_id = 0
        self._opened = False
        self._started = False

    def open(self) -> None:
        if self._opened:
            return

        try:
            from pypylon import pylon
        except ImportError as exc:
            raise CameraError(
                "Python module 'pypylon' is not available. "
                "Install Basler pylon and run: python3 -m pip install pypylon"
            ) from exc

        self._pylon = pylon

        try:
            device = self._select_device()
            self._camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateDevice(device)
            )
            self._camera.Open()
            self._apply_settings()
            self._opened = True
        except Exception as exc:
            self.close()
            raise CameraError(f"Failed to open Basler camera: {exc}") from exc

    def start(self) -> None:
        self._require_opened()
        if self._started:
            return

        try:
            self._camera.StartGrabbing(self._pylon.GrabStrategy_LatestImageOnly)
            self._started = True
        except Exception as exc:
            raise CameraError(f"Failed to start Basler acquisition: {exc}") from exc

    def read(self) -> Frame:
        self._require_started()

        result = None
        try:
            result = self._camera.RetrieveResult(
                self.timeout_ms,
                self._pylon.TimeoutHandling_ThrowException,
            )
            if not result.GrabSucceeded():
                raise CameraError(
                    "Basler grab failed: "
                    f"{result.GetErrorCode()} {result.GetErrorDescription()}"
                )

            image = self._to_opencv_image(result.Array)
        except CameraError:
            raise
        except Exception as exc:
            raise CameraError(f"Failed to read frame from Basler camera: {exc}") from exc
        finally:
            if result is not None:
                result.Release()

        frame = Frame(
            image=image,
            source_id=self.source_id,
            sequence_id=self._sequence_id,
            metadata={
                "camera": "basler",
                "serial_number": self.serial_number,
                "pixel_format": self.pixel_format,
            },
        )
        self._sequence_id += 1
        return frame

    def stop(self) -> None:
        if not self._camera or not self._started:
            return

        try:
            if self._camera.IsGrabbing():
                self._camera.StopGrabbing()
        finally:
            self._started = False

    def close(self) -> None:
        if self._camera is None:
            self._opened = False
            self._started = False
            return

        try:
            if self._camera.IsGrabbing():
                self._camera.StopGrabbing()
            if self._camera.IsOpen():
                self._camera.Close()
        finally:
            self._camera = None
            self._opened = False
            self._started = False

    def _select_device(self):
        factory = self._pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()
        if not devices:
            raise CameraError("No Basler cameras found")

        if self.serial_number is None:
            return devices[0]

        for device in devices:
            if device.GetSerialNumber() == self.serial_number:
                return device

        available = ", ".join(device.GetSerialNumber() for device in devices)
        raise CameraError(
            f"Basler camera with serial {self.serial_number} not found. "
            f"Available serials: {available}"
        )

    def _apply_settings(self) -> None:
        self._set_enum("PixelFormat", self.pixel_format)

        if self.exposure_us is not None:
            self._set_optional_enum("ExposureMode", "Timed")
            self._set_optional_enum("ExposureAuto", "Off")
            self._set_float("ExposureTime", self.exposure_us)

        if self.gain_db is not None:
            self._set_optional_enum("GainAuto", "Off")
            self._set_float("Gain", self.gain_db)

    def _set_enum(self, name: str, value: str) -> None:
        node = getattr(self._camera, name)
        node.SetValue(value)

    def _set_optional_enum(self, name: str, value: str) -> None:
        node = getattr(self._camera, name, None)
        if node is not None and node.IsWritable():
            node.SetValue(value)

    def _set_float(self, name: str, value: float) -> None:
        node = getattr(self._camera, name)
        if not node.IsWritable():
            raise CameraError(f"Basler parameter {name} is not writable")
        node.SetValue(float(value))

    def _to_opencv_image(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise CameraError("Basler returned an empty image")

        opencv_image = np.asarray(image)

        if opencv_image.ndim == 2:
            if opencv_image.dtype == np.uint8:
                return cv2.cvtColor(opencv_image, cv2.COLOR_GRAY2BGR)
            normalized = cv2.normalize(opencv_image, None, 0, 255, cv2.NORM_MINMAX)
            return cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        if opencv_image.ndim == 3 and opencv_image.shape[2] == 3:
            return cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        if opencv_image.ndim == 3 and opencv_image.shape[2] == 4:
            return cv2.cvtColor(opencv_image, cv2.COLOR_RGBA2BGRA)

        raise CameraError(f"Unsupported Basler image shape: {opencv_image.shape}")

    def _require_opened(self) -> None:
        if not self._camera or not self._opened:
            raise CameraError("Basler camera is not opened")

    def _require_started(self) -> None:
        self._require_opened()
        if not self._started:
            raise CameraError("Basler acquisition is not started")

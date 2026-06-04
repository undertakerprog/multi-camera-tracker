from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.cameras.base import CameraError, CameraSource
from src.common import Frame


class XimeaCameraSource(CameraSource):
    """Camera source backed by XIMEA xiAPI.

    This module is intentionally the only place that imports and talks to
    `ximea.xiapi`, keeping the rest of the application camera-agnostic.
    """

    def __init__(
        self,
        device_index: int = 0,
        serial_number: str | None = None,
        source_id: str = "ximea-0",
        exposure_us: int | None = None,
        gain_db: float | None = None,
        image_format: str = "XI_RGB24",
        timeout_ms: int = 5000,
    ) -> None:
        self.device_index = device_index
        self.serial_number = serial_number
        self.source_id = source_id
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.image_format = image_format
        self.timeout_ms = timeout_ms

        self._camera: Any | None = None
        self._xi_image: Any | None = None
        self._sequence_id = 0
        self._opened = False
        self._started = False

    def open(self) -> None:
        if self._opened:
            return

        try:
            from ximea import xiapi
        except ImportError as exc:
            raise CameraError(
                "Python module 'ximea.xiapi' is not available. "
                "Install or activate the XIMEA Linux Software Package bindings."
            ) from exc

        try:
            self._camera = xiapi.Camera()
            self._open_device()
            self._xi_image = xiapi.Image()
            self._apply_settings()
            self._opened = True
        except Exception as exc:
            if self._camera is not None:
                try:
                    self._camera.close_device()
                except Exception:
                    pass
            self._camera = None
            self._xi_image = None
            raise CameraError(f"Failed to open XIMEA camera: {exc}") from exc

    def start(self) -> None:
        self._require_opened()
        if self._started:
            return

        try:
            self._camera.start_acquisition()
            self._started = True
        except Exception as exc:
            raise CameraError(f"Failed to start XIMEA acquisition: {exc}") from exc

    def read(self) -> Frame:
        self._require_started()

        try:
            self._camera.get_image(self._xi_image, timeout=self.timeout_ms)
            image = self._to_opencv_image(self._xi_image.get_image_data_numpy())
        except Exception as exc:
            raise CameraError(f"Failed to read frame from XIMEA camera: {exc}") from exc

        frame = Frame(
            image=image,
            source_id=self.source_id,
            sequence_id=self._sequence_id,
            metadata={"camera": "ximea", "device_index": self.device_index},
        )
        self._sequence_id += 1
        return frame

    def stop(self) -> None:
        if not self._camera or not self._started:
            return

        try:
            self._camera.stop_acquisition()
        finally:
            self._started = False

    def close(self) -> None:
        if not self._camera or not self._opened:
            return

        try:
            self._camera.close_device()
        finally:
            self._camera = None
            self._xi_image = None
            self._opened = False
            self._started = False

    def _apply_settings(self) -> None:
        if self.image_format:
            self._set_param("imgdataformat", self.image_format)
        if self.exposure_us is not None:
            self._set_param("exposure", self.exposure_us)
        if self.gain_db is not None:
            self._set_param("gain", self.gain_db)

    def _open_device(self) -> None:
        if self.serial_number:
            open_by_sn = getattr(self._camera, "open_device_by_SN", None)
            if open_by_sn is None:
                raise CameraError("XIMEA SDK does not expose open_device_by_SN")
            open_by_sn(self.serial_number)
            return

        if self.device_index == 0:
            self._camera.open_device()
            return

        open_by = getattr(self._camera, "open_device_by", None)
        if open_by is None:
            raise CameraError(
                "Opening a non-zero XIMEA device index is not supported by this SDK"
            )
        open_by(self.device_index)

    def _set_param(self, name: str, value: Any) -> None:
        setter = getattr(self._camera, f"set_{name}", None)
        if setter is None:
            raise CameraError(f"XIMEA parameter setter set_{name} is not available")
        setter(value)

    def _to_opencv_image(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise CameraError("XIMEA returned an empty image")

        opencv_image = np.asarray(image)

        if opencv_image.ndim == 2:
            return opencv_image

        if opencv_image.ndim == 3 and opencv_image.shape[2] == 3:
            return cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        if opencv_image.ndim == 3 and opencv_image.shape[2] == 4:
            return cv2.cvtColor(opencv_image, cv2.COLOR_RGBA2BGRA)

        raise CameraError(f"Unsupported XIMEA image shape: {opencv_image.shape}")

    def _require_opened(self) -> None:
        if not self._camera or not self._opened:
            raise CameraError("XIMEA camera is not opened")

    def _require_started(self) -> None:
        self._require_opened()
        if not self._started:
            raise CameraError("XIMEA acquisition is not started")

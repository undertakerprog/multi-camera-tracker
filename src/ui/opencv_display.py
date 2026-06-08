from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class DisplayOverlay:
    bbox: tuple[int, int, int, int] | None = None
    selection_bbox: tuple[int, int, int, int] | None = None
    center: tuple[int, int] | None = None
    status: str = "NO TARGET"
    tracker_name: str | None = None
    frame_stats: str | None = None
    telemetry: str | None = None


class OpenCVDisplay:
    def __init__(self, window_name: str = "Target Tracker", auto_contrast: bool = True) -> None:
        self.window_name = window_name
        self.auto_contrast = auto_contrast
        self._created = False

    def create(self, image_shape: tuple[int, ...] | None = None) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if image_shape is not None:
            height, width = image_shape[:2]
            cv2.resizeWindow(self.window_name, width, height)
        self._created = True

    def set_mouse_callback(self, callback) -> None:
        cv2.setMouseCallback(self.window_name, callback)

    def is_open(self) -> bool:
        if not self._created:
            return False
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE) >= 0
        except cv2.error:
            return False

    def show(self, image: np.ndarray, overlay: DisplayOverlay) -> int:
        if not self.is_open():
            return ord("q")

        canvas = self._prepare_canvas(image)
        self._draw_overlay(canvas, overlay)
        cv2.imshow(self.window_name, canvas)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        if self._created:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
            self._created = False

    def _prepare_canvas(self, image: np.ndarray) -> np.ndarray:
        canvas = image.copy()
        if not self.auto_contrast:
            return canvas

        if canvas.dtype != np.uint8:
            return cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        min_value = int(canvas.min())
        max_value = int(canvas.max())
        if max_value <= 40 or max_value - min_value < 20:
            return cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX)

        return canvas

    def _draw_overlay(self, image: np.ndarray, overlay: DisplayOverlay) -> None:
        if overlay.selection_bbox is not None:
            x, y, w, h = overlay.selection_bbox
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

        if overlay.bbox is not None:
            x, y, w, h = overlay.bbox
            color = (0, 255, 0) if overlay.status == "TRACKING" else (0, 165, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        if overlay.center is not None:
            cx, cy = overlay.center
            cv2.drawMarker(
                image,
                (cx, cy),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=16,
                thickness=2,
            )
            cv2.putText(
                image,
                f"center: ({cx}, {cy})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        tracker_text = f" | {overlay.tracker_name}" if overlay.tracker_name else ""
        cv2.putText(
            image,
            f"{overlay.status}{tracker_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if overlay.frame_stats:
            cv2.putText(
                image,
                overlay.frame_stats,
                (10, image.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if overlay.telemetry:
            cv2.putText(
                image,
                overlay.telemetry,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

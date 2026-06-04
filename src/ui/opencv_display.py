from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class DisplayOverlay:
    bbox: tuple[int, int, int, int] | None = None
    center: tuple[int, int] | None = None
    status: str = "NO TARGET"
    tracker_name: str | None = None


class OpenCVDisplay:
    def __init__(self, window_name: str = "Target Tracker") -> None:
        self.window_name = window_name

    def create(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, image: np.ndarray, overlay: DisplayOverlay) -> int:
        canvas = image.copy()
        self._draw_overlay(canvas, overlay)
        cv2.imshow(self.window_name, canvas)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)

    def _draw_overlay(self, image: np.ndarray, overlay: DisplayOverlay) -> None:
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

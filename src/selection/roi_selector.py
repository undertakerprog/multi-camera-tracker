import cv2
import numpy as np


class ROISelector:
    def __init__(self, window_name: str = "Target Tracker") -> None:
        self.window_name = window_name

    def select(self, image: np.ndarray) -> tuple[int, int, int, int] | None:
        bbox = cv2.selectROI(
            self.window_name,
            image,
            showCrosshair=True,
            fromCenter=False,
        )
        cv2.setWindowTitle(self.window_name, self.window_name)

        x, y, w, h = (int(value) for value in bbox)
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h

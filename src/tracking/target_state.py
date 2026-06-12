from dataclasses import dataclass

import cv2
import numpy as np

from src.tracking import states
from src.tracking.opencv_tracker import TrackingResult


@dataclass(slots=True)
class TargetStateUpdate:
    status: str
    bbox: tuple[int, int, int, int] | None
    center: tuple[int, int] | None
    lost_frames: int
    expired: bool = False
    confirmed: bool = False
    confirmations: int = 0


class TargetState:
    def __init__(
        self,
        smoothing_alpha: float = 0.35,
        max_lost_frames: int = 90,
        min_box_size: int = 4,
    ) -> None:
        if smoothing_alpha <= 0 or smoothing_alpha > 1:
            raise ValueError("smoothing_alpha must be > 0 and <= 1")
        if max_lost_frames < 0:
            raise ValueError("max_lost_frames must be >= 0")

        self.smoothing_alpha = smoothing_alpha
        self.max_lost_frames = max_lost_frames
        self.min_box_size = min_box_size
        self._bbox: tuple[float, float, float, float] | None = None
        self._lost_frames = 0
        self._confirmations = 0
        self._kalman = self._create_kalman()

    @property
    def bbox(self) -> tuple[int, int, int, int] | None:
        return self._round_bbox(self._bbox)

    @property
    def center(self) -> tuple[int, int] | None:
        bbox = self.bbox
        if bbox is None:
            return None
        x, y, w, h = bbox
        return x + w // 2, y + h // 2

    @property
    def lost_frames(self) -> int:
        return self._lost_frames

    @property
    def confirmations(self) -> int:
        return self._confirmations

    def initialize(self, bbox: tuple[int, int, int, int]) -> TargetStateUpdate:
        self._bbox = tuple(float(value) for value in bbox)
        self._lost_frames = 0
        self._confirmations = 1
        self._reset_kalman(bbox)
        return self._update(states.TRACKING, confirmed=True)

    def update(
        self,
        result: TrackingResult,
        frame_shape: tuple[int, ...],
    ) -> TargetStateUpdate:
        predicted_bbox = self._predict(frame_shape)

        if result.ok and result.bbox is not None and self._is_valid(result.bbox, frame_shape):
            self._lost_frames = 0
            self._confirmations += 1
            corrected_bbox = self._clip_bbox(
                self._correct(self._clip_bbox(result.bbox, frame_shape)),
                frame_shape,
            )
            self._bbox = self._smooth(corrected_bbox)
            return self._update(states.TRACKING, confirmed=True)

        self._lost_frames += 1
        self._confirmations = 0
        if predicted_bbox is not None:
            self._bbox = tuple(float(value) for value in predicted_bbox)

        if self._lost_frames > self.max_lost_frames:
            return TargetStateUpdate(
                status=states.TARGET_STALE,
                bbox=self.bbox,
                center=self.center,
                lost_frames=self._lost_frames,
                expired=True,
                confirmations=self._confirmations,
            )

        return self._update(states.PREDICTING)

    def mark_lost(self) -> None:
        """Force the next observation to be treated as a fresh confirmation.

        Used when an external verifier (e.g. periodic template check) rejects the
        tracker output so a stuck box is not trusted indefinitely.
        """
        self._confirmations = 0

    def reset(self) -> None:
        self._bbox = None
        self._lost_frames = 0
        self._confirmations = 0
        self._kalman = self._create_kalman()

    def _smooth(self, bbox: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
        current = tuple(float(value) for value in bbox)
        if self._bbox is None:
            return current

        alpha = self.smoothing_alpha
        return tuple(
            previous * (1 - alpha) + incoming * alpha
            for previous, incoming in zip(self._bbox, current)
        )

    def _update(self, status: str, confirmed: bool = False) -> TargetStateUpdate:
        return TargetStateUpdate(
            status=status,
            bbox=self.bbox,
            center=self.center,
            lost_frames=self._lost_frames,
            confirmed=confirmed,
            confirmations=self._confirmations,
        )

    def _is_valid(self, bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> bool:
        height, width = frame_shape[:2]
        x, y, w, h = bbox

        if w < self.min_box_size or h < self.min_box_size:
            return False
        if x + w <= 0 or y + h <= 0:
            return False
        if x >= width or y >= height:
            return False
        return True

    def _predict(self, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
        if self._bbox is None:
            return None

        prediction = self._kalman.predict()
        cx, cy, w, h = (float(value) for value in prediction[:4, 0])
        return self._clip_bbox(
            (
                int(round(cx - w / 2)),
                int(round(cy - h / 2)),
                max(1, int(round(w))),
                max(1, int(round(h))),
            ),
            frame_shape,
        )

    def _correct(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = bbox
        measurement = np.array(
            [[x + w / 2], [y + h / 2], [w], [h]],
            dtype=np.float32,
        )
        corrected = self._kalman.correct(measurement)
        cx, cy, corrected_w, corrected_h = (float(value) for value in corrected[:4, 0])
        return (
            int(round(cx - corrected_w / 2)),
            int(round(cy - corrected_h / 2)),
            max(1, int(round(corrected_w))),
            max(1, int(round(corrected_h))),
        )

    def _reset_kalman(self, bbox: tuple[int, int, int, int]) -> None:
        self._kalman = self._create_kalman()
        x, y, w, h = bbox
        self._kalman.statePre = np.array(
            [[x + w / 2], [y + h / 2], [w], [h], [0], [0]],
            dtype=np.float32,
        )
        self._kalman.statePost = self._kalman.statePre.copy()

    @staticmethod
    def _create_kalman():
        kalman = cv2.KalmanFilter(6, 4)
        kalman.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.4
        kalman.errorCovPost = np.eye(6, dtype=np.float32)
        return kalman

    @staticmethod
    def _clip_bbox(
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x, y, w, h = bbox

        x1 = max(0, min(width - 1, x))
        y1 = max(0, min(height - 1, y))
        x2 = max(0, min(width, x + w))
        y2 = max(0, min(height, y + h))

        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    @staticmethod
    def _round_bbox(
        bbox: tuple[float, float, float, float] | None,
    ) -> tuple[int, int, int, int] | None:
        if bbox is None:
            return None
        x, y, w, h = bbox
        return (
            int(round(x)),
            int(round(y)),
            max(1, int(round(w))),
            max(1, int(round(h))),
        )

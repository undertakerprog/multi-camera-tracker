from dataclasses import dataclass

from src.tracking.opencv_tracker import TrackingResult


@dataclass(slots=True)
class TargetStateUpdate:
    status: str
    bbox: tuple[int, int, int, int] | None
    center: tuple[int, int] | None
    lost_frames: int
    expired: bool = False


class TargetState:
    def __init__(
        self,
        smoothing_alpha: float = 0.35,
        max_lost_frames: int = 10,
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

    def initialize(self, bbox: tuple[int, int, int, int]) -> TargetStateUpdate:
        self._bbox = tuple(float(value) for value in bbox)
        self._lost_frames = 0
        return self._update("TRACKING")

    def update(
        self,
        result: TrackingResult,
        frame_shape: tuple[int, ...],
    ) -> TargetStateUpdate:
        if result.ok and result.bbox is not None and self._is_valid(result.bbox, frame_shape):
            self._lost_frames = 0
            self._bbox = self._smooth(self._clip_bbox(result.bbox, frame_shape))
            return self._update("TRACKING")

        self._lost_frames += 1
        if self._lost_frames > self.max_lost_frames:
            self.reset()
            return TargetStateUpdate(
                status="NO TARGET",
                bbox=None,
                center=None,
                lost_frames=0,
                expired=True,
            )

        return self._update("TARGET LOST")

    def reset(self) -> None:
        self._bbox = None
        self._lost_frames = 0

    def _smooth(self, bbox: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
        current = tuple(float(value) for value in bbox)
        if self._bbox is None:
            return current

        alpha = self.smoothing_alpha
        return tuple(
            previous * (1 - alpha) + incoming * alpha
            for previous, incoming in zip(self._bbox, current)
        )

    def _update(self, status: str) -> TargetStateUpdate:
        return TargetStateUpdate(
            status=status,
            bbox=self.bbox,
            center=self.center,
            lost_frames=self._lost_frames,
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

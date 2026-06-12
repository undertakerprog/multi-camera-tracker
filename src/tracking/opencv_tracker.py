from dataclasses import dataclass
import math

import cv2
import numpy as np


class TrackerError(RuntimeError):
    """Raised when an OpenCV tracker cannot be created or initialized."""


@dataclass(slots=True)
class TrackingResult:
    ok: bool
    bbox: tuple[int, int, int, int] | None = None

    @property
    def center(self) -> tuple[int, int] | None:
        if self.bbox is None:
            return None
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2


class OpenCVObjectTracker:
    def __init__(self, tracker_type: str = "KCF", tracking_scale: float = 0.5) -> None:
        self.tracker_type = tracker_type.upper()
        if tracking_scale <= 0 or tracking_scale > 1:
            raise TrackerError("tracking_scale must be > 0 and <= 1")
        self.tracking_scale = tracking_scale
        self._tracker = None
        self._initialized = False

    @property
    def name(self) -> str:
        return f"{self.tracker_type} x{self.tracking_scale:g}"

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        self.reset()
        clipped_bbox = self._clip_bbox(bbox, image.shape if image is not None else ())
        if clipped_bbox is None:
            raise TrackerError("Cannot initialize tracker with an empty frame or invalid bbox")

        tracking_image = self._prepare_tracking_image(image)
        tracking_bbox = self._clip_bbox(
            self._scale_bbox(clipped_bbox, self.tracking_scale),
            tracking_image.shape,
        )
        if tracking_bbox is None:
            raise TrackerError("Tracker bbox became invalid after image scaling")

        self._tracker = self._create_tracker()
        try:
            ok = self._tracker.init(tracking_image, tracking_bbox)
        except cv2.error as exc:
            self.reset()
            raise TrackerError(
                f"OpenCV failed to initialize {self.tracker_type} tracker: {exc}"
            ) from exc
        if ok is False:
            self.reset()
            raise TrackerError(f"Failed to initialize {self.tracker_type} tracker")
        self._initialized = True

    def update(self, image: np.ndarray) -> TrackingResult:
        if not self._tracker or not self._initialized:
            return TrackingResult(ok=False)

        if not self._valid_image(image):
            self.reset()
            return TrackingResult(ok=False)

        try:
            tracking_image = self._prepare_tracking_image(image)
        except TrackerError:
            self.reset()
            return TrackingResult(ok=False)
        try:
            ok, bbox = self._tracker.update(tracking_image)
        except cv2.error:
            self.reset()
            return TrackingResult(ok=False)
        if not ok:
            return TrackingResult(ok=False)

        tracked_bbox = self._clip_bbox(bbox, tracking_image.shape)
        if tracked_bbox is None:
            self.reset()
            return TrackingResult(ok=False)
        full_bbox = self._clip_bbox(
            self._scale_bbox(tracked_bbox, 1 / self.tracking_scale),
            image.shape,
        )
        if full_bbox is None:
            self.reset()
            return TrackingResult(ok=False)
        return TrackingResult(ok=True, bbox=full_bbox)

    def reset(self) -> None:
        self._tracker = None
        self._initialized = False

    def _create_tracker(self):
        constructors = self._candidate_constructors(self.tracker_type)
        for owner, name in constructors:
            factory = getattr(owner, name, None)
            if factory is not None:
                return factory()
        raise TrackerError(
            f"OpenCV tracker '{self.tracker_type}' is not available. "
            "Install opencv-contrib-python or choose another tracker."
        )

    def _candidate_constructors(self, tracker_type: str):
        names = {
            "CSRT": ["TrackerCSRT_create"],
            "KCF": ["TrackerKCF_create"],
            "MOSSE": ["TrackerMOSSE_create"],
            "MIL": ["TrackerMIL_create"],
        }.get(tracker_type)

        if names is None:
            raise TrackerError(f"Unsupported OpenCV tracker type: {tracker_type}")

        owners = [cv2]
        legacy = getattr(cv2, "legacy", None)
        if legacy is not None:
            owners.append(legacy)

        return [(owner, name) for owner in owners for name in names]

    def _resize_for_tracking(self, image: np.ndarray) -> np.ndarray:
        if not self._valid_image(image):
            raise TrackerError("Cannot resize an empty tracking image")
        if self.tracking_scale == 1:
            return np.ascontiguousarray(image)

        width = max(1, int(round(image.shape[1] * self.tracking_scale)))
        height = max(1, int(round(image.shape[0] * self.tracking_scale)))
        try:
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        except cv2.error as exc:
            raise TrackerError(f"OpenCV failed to resize tracking image: {exc}") from exc

    def _prepare_tracking_image(self, image: np.ndarray) -> np.ndarray:
        tracking_image = self._resize_for_tracking(image)
        # OpenCV 4.5.x KCF enables ColorNames features by default and is not
        # reliable with a single-channel image. Convert only the small tracking
        # ROI, keeping the camera and the rest of the pipeline in Mono8.
        if self.tracker_type == "KCF" and tracking_image.ndim == 2:
            try:
                tracking_image = cv2.cvtColor(tracking_image, cv2.COLOR_GRAY2BGR)
            except cv2.error as exc:
                raise TrackerError(f"OpenCV failed to prepare KCF image: {exc}") from exc
        return np.ascontiguousarray(tracking_image)

    @staticmethod
    def _scale_bbox(
        bbox: tuple[int, int, int, int] | tuple[float, float, float, float],
        scale: float,
    ) -> tuple[int, int, int, int]:
        x, y, w, h = bbox
        return (
            int(round(x * scale)),
            int(round(y * scale)),
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        )

    @staticmethod
    def _valid_image(image: np.ndarray | None) -> bool:
        return (
            isinstance(image, np.ndarray)
            and image.ndim >= 2
            and image.size > 0
            and image.shape[0] > 0
            and image.shape[1] > 0
        )

    @staticmethod
    def _clip_bbox(
        bbox: tuple[int, int, int, int] | tuple[float, float, float, float],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:
        if len(frame_shape) < 2:
            return None
        height, width = frame_shape[:2]
        if height <= 0 or width <= 0:
            return None

        x, y, w, h = (float(value) for value in bbox)
        if not all(math.isfinite(value) for value in (x, y, w, h)) or w <= 0 or h <= 0:
            return None

        x1 = max(0, int(math.floor(x)))
        y1 = max(0, int(math.floor(y)))
        x2 = min(width, int(math.ceil(x + w)))
        y2 = min(height, int(math.ceil(y + h)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return x1, y1, x2 - x1, y2 - y1

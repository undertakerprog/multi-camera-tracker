from dataclasses import dataclass

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
    def __init__(self, tracker_type: str = "CSRT") -> None:
        self.tracker_type = tracker_type.upper()
        self._tracker = None
        self._initialized = False

    @property
    def name(self) -> str:
        return self.tracker_type

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        self._tracker = self._create_tracker()
        ok = self._tracker.init(image, bbox)
        if ok is False:
            self._tracker = None
            self._initialized = False
            raise TrackerError(f"Failed to initialize {self.tracker_type} tracker")
        self._initialized = True

    def update(self, image: np.ndarray) -> TrackingResult:
        if not self._tracker or not self._initialized:
            return TrackingResult(ok=False)

        ok, bbox = self._tracker.update(image)
        if not ok:
            return TrackingResult(ok=False)

        x, y, w, h = (int(round(value)) for value in bbox)
        return TrackingResult(ok=True, bbox=(x, y, w, h))

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

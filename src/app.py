import logging

from src.cameras.base import CameraError, CameraSource
from src.selection import ROISelector
from src.tracking import OpenCVObjectTracker, TrackerError
from src.ui import DisplayOverlay, OpenCVDisplay


LOG = logging.getLogger(__name__)


class TargetTrackingApp:
    def __init__(
        self,
        camera_source: CameraSource,
        tracker: OpenCVObjectTracker | None = None,
        tracker_type: str = "CSRT",
        display: OpenCVDisplay | None = None,
        roi_selector: ROISelector | None = None,
    ) -> None:
        self.camera_source = camera_source
        self.tracker = tracker or OpenCVObjectTracker(tracker_type)
        self.display = display or OpenCVDisplay()
        self.roi_selector = roi_selector or ROISelector(self.display.window_name)
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._last_center: tuple[int, int] | None = None
        self._status = "NO TARGET"
        self._frame_count = 0

    def run(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        self.display.create()

        try:
            LOG.info("Opening camera")
            self.camera_source.open()
            self.camera_source.start()
            LOG.info("Camera acquisition started")
            self._loop()
        except CameraError as exc:
            LOG.error("Camera error: %s", exc)
        except TrackerError as exc:
            LOG.error("Tracker error: %s", exc)
        finally:
            self._shutdown()

    def _loop(self) -> None:
        while True:
            frame = self.camera_source.read()
            self._frame_count += 1
            image = frame.image

            if self.tracker.initialized:
                result = self.tracker.update(image)
                if result.ok:
                    self._last_bbox = result.bbox
                    self._last_center = result.center
                    self._status = "TRACKING"
                    if self._last_center is not None and self._frame_count % 15 == 0:
                        LOG.info("Target center: x=%d y=%d", *self._last_center)
                else:
                    self._status = "TARGET LOST"
                    self._last_center = None

            key = self.display.show(
                image,
                DisplayOverlay(
                    bbox=self._last_bbox,
                    center=self._last_center,
                    status=self._status,
                    tracker_name=self.tracker.name,
                ),
            )

            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                self._reset_target()
            if key == ord("s"):
                self._select_target(image)

    def _select_target(self, image) -> None:
        bbox = self.roi_selector.select(image)
        if bbox is None:
            self._status = "NO TARGET"
            return

        self.tracker.initialize(image, bbox)
        self._last_bbox = bbox
        self._last_center = self._bbox_center(bbox)
        self._status = "TRACKING"
        LOG.info("Selected target bbox: x=%d y=%d w=%d h=%d", *bbox)

    def _reset_target(self) -> None:
        self.tracker.reset()
        self._last_bbox = None
        self._last_center = None
        self._status = "NO TARGET"
        LOG.info("Target reset")

    def _shutdown(self) -> None:
        LOG.info("Shutting down")
        try:
            self.camera_source.stop()
        except CameraError as exc:
            LOG.warning("Failed to stop camera cleanly: %s", exc)
        finally:
            self.camera_source.close()
            self.display.close()

    @staticmethod
    def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        x, y, w, h = bbox
        return x + w // 2, y + h // 2

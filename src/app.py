import logging
from time import perf_counter

from src.cameras.base import CameraError, CameraSource
from src.selection import ROISelector
from src.tracking import OpenCVObjectTracker, TargetState, TrackerError
from src.ui import DisplayOverlay, OpenCVDisplay


LOG = logging.getLogger(__name__)


class TargetTrackingApp:
    def __init__(
        self,
        camera_source: CameraSource,
        tracker: OpenCVObjectTracker | None = None,
        tracker_type: str = "KCF",
        tracking_scale: float = 0.5,
        smoothing_alpha: float = 0.35,
        max_lost_frames: int = 10,
        display: OpenCVDisplay | None = None,
        roi_selector: ROISelector | None = None,
    ) -> None:
        self.camera_source = camera_source
        self.tracker = tracker or OpenCVObjectTracker(tracker_type, tracking_scale)
        self.target_state = TargetState(
            smoothing_alpha=smoothing_alpha,
            max_lost_frames=max_lost_frames,
        )
        self.display = display or OpenCVDisplay()
        self.roi_selector = roi_selector or ROISelector(self.display.window_name)
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._last_center: tuple[int, int] | None = None
        self._status = "NO TARGET"
        self._frame_count = 0
        self._fps = 0.0
        self._last_frame_time: float | None = None
        self._last_center_log_time = 0.0

    def run(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        try:
            LOG.info("Opening camera")
            self.camera_source.open()
            self.camera_source.start()
            LOG.info("Camera acquisition started")
            first_frame = self._read_frame()
            self.display.create(first_frame.image.shape)
            self.roi_selector.attach()
            self._loop(first_frame)
        except CameraError as exc:
            LOG.error("Camera error: %s", exc)
        except TrackerError as exc:
            LOG.error("Tracker error: %s", exc)
        finally:
            self._shutdown()

    def _loop(self, first_frame=None) -> None:
        frame = first_frame
        while True:
            if frame is None:
                frame = self._read_frame()

            image = frame.image
            frame_stats = self._frame_stats(image)
            self._update_fps()

            if self.tracker.initialized:
                result = self.tracker.update(image)
                state = self.target_state.update(result, image.shape)
                self._last_bbox = state.bbox
                self._last_center = state.center
                self._status = state.status
                if state.expired:
                    self.tracker.reset()
                    LOG.info("Target expired after lost frames")
                elif self._status == "TRACKING" and self._last_center is not None:
                    now = perf_counter()
                    if now - self._last_center_log_time >= 1.0:
                        LOG.info("Target center: x=%d y=%d", *self._last_center)
                        self._last_center_log_time = now

            key = self.display.show(
                image,
                DisplayOverlay(
                    bbox=self._last_bbox,
                    selection_bbox=self.roi_selector.pending_bbox,
                    center=self._last_center,
                    status=self._status,
                    tracker_name=self.tracker.name,
                    frame_stats=frame_stats,
                    telemetry=self._telemetry(),
                ),
            )

            if key in (ord("q"), 27):
                LOG.info("Exit key received: %s", key)
                break
            if key == ord("r"):
                self._reset_target()
            if key == ord("s"):
                LOG.info("Use left mouse drag in the video window to select target")

            mouse_bbox = self.roi_selector.pop_selected()
            if mouse_bbox is not None:
                self._initialize_target(image, mouse_bbox)

            frame = None

    def _read_frame(self):
        frame = self.camera_source.read()
        self._frame_count += 1
        if self._frame_count == 1:
            LOG.info("First %s", self._frame_stats(frame.image))
        return frame

    def _initialize_target(self, image, bbox: tuple[int, int, int, int]) -> None:
        self.tracker.initialize(image, bbox)
        state = self.target_state.initialize(bbox)
        self._last_bbox = state.bbox
        self._last_center = state.center
        self._status = state.status
        LOG.info("Selected target bbox: x=%d y=%d w=%d h=%d", *bbox)

    def _reset_target(self) -> None:
        self.tracker.reset()
        self.target_state.reset()
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
    def _frame_stats(image) -> str:
        return (
            f"frame: {image.shape[1]}x{image.shape[0]} "
            f"min={int(image.min())} max={int(image.max())} mean={float(image.mean()):.1f}"
        )

    def _update_fps(self) -> None:
        now = perf_counter()
        if self._last_frame_time is None:
            self._last_frame_time = now
            return

        elapsed = now - self._last_frame_time
        self._last_frame_time = now
        if elapsed <= 0:
            return

        current_fps = 1.0 / elapsed
        if self._fps == 0:
            self._fps = current_fps
        else:
            self._fps = self._fps * 0.9 + current_fps * 0.1

    def _telemetry(self) -> str:
        return f"fps={self._fps:.1f} lost={self.target_state.lost_frames}"

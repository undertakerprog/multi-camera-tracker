import logging
from time import perf_counter

from src.cameras.base import CameraError, CameraSource
from src.selection import ROISelector
from src.tracking import OpenCVObjectTracker, TargetState, TemplateReacquirer, TrackerError
from src.tracking.opencv_tracker import TrackingResult
from src.ui import DisplayOverlay, OpenCVDisplay


LOG = logging.getLogger(__name__)


class TargetTrackingApp:
    def __init__(
        self,
        camera_source: CameraSource,
        tracker: OpenCVObjectTracker | None = None,
        tracker_type: str = "CSRT",
        tracking_scale: float = 0.75,
        smoothing_alpha: float = 0.35,
        max_lost_frames: int = 90,
        reacquire_enabled: bool = True,
        reacquire_min_score: float = 0.62,
        reacquire_search_expansion: float = 3.0,
        global_reacquire_enabled: bool = True,
        global_reacquire_after: int = 8,
        global_reacquire_interval: int = 5,
        global_reacquire_score: float = 0.72,
        global_reacquire_scale: float = 0.5,
        template_update_interval: int = 15,
        display: OpenCVDisplay | None = None,
        roi_selector: ROISelector | None = None,
    ) -> None:
        self.camera_source = camera_source
        self.tracker = tracker or OpenCVObjectTracker(tracker_type, tracking_scale)
        self.target_state = TargetState(
            smoothing_alpha=smoothing_alpha,
            max_lost_frames=max_lost_frames,
        )
        self.reacquirer = TemplateReacquirer(
            search_expansion=reacquire_search_expansion,
            min_score=reacquire_min_score,
            global_min_score=global_reacquire_score,
            global_search_scale=global_reacquire_scale,
        )
        self.reacquire_enabled = reacquire_enabled
        self.global_reacquire_enabled = global_reacquire_enabled
        self.global_reacquire_after = global_reacquire_after
        self.global_reacquire_interval = global_reacquire_interval
        self.template_update_interval = template_update_interval
        self.display = display or OpenCVDisplay()
        self.roi_selector = roi_selector or ROISelector(self.display.window_name)
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._last_center: tuple[int, int] | None = None
        self._status = "NO TARGET"
        self._frame_count = 0
        self._fps = 0.0
        self._last_frame_time: float | None = None
        self._last_center_log_time = 0.0
        self._last_reacquire_score = 0.0

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

            if self.tracker.initialized or self.reacquirer.has_template:
                result = TrackingResult(ok=False)
                if self.tracker.initialized:
                    result = self.tracker.update(image)
                    if result.ok:
                        self._last_reacquire_score = 0.0

                previous_status = self._status
                state = self.target_state.update(result, image.shape)

                if not result.ok and self.reacquire_enabled:
                    reacquired_bbox, score = self.reacquirer.search(image, state.bbox)
                    if (
                        reacquired_bbox is None
                        and self._should_global_reacquire()
                    ):
                        reacquired_bbox, score = self.reacquirer.search_global(image)
                    self._last_reacquire_score = score
                    if reacquired_bbox is not None:
                        self.tracker.initialize(image, reacquired_bbox)
                        self.reacquirer.initialize(image, reacquired_bbox)
                        state = self.target_state.initialize(reacquired_bbox)
                        LOG.info(
                            "Target reacquired: x=%d y=%d w=%d h=%d score=%.3f",
                            *reacquired_bbox,
                            score,
                        )

                self._last_bbox = state.bbox
                self._last_center = state.center
                self._status = state.status
                if state.expired and self.tracker.initialized:
                    self.tracker.reset()
                    if previous_status != "TARGET STALE":
                        LOG.info("Target stale, continuing prediction and reacquire search")
                elif self._status == "TRACKING" and self._last_center is not None:
                    if self.template_update_interval > 0:
                        if self._frame_count % self.template_update_interval == 0:
                            self.reacquirer.initialize(image, self._last_bbox)
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
        self.reacquirer.initialize(image, bbox)
        state = self.target_state.initialize(bbox)
        self._last_bbox = state.bbox
        self._last_center = state.center
        self._status = state.status
        LOG.info("Selected target bbox: x=%d y=%d w=%d h=%d", *bbox)

    def _reset_target(self) -> None:
        self.tracker.reset()
        self.reacquirer.reset()
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
        return (
            f"fps={self._fps:.1f} "
            f"lost={self.target_state.lost_frames} "
            f"match={self._last_reacquire_score:.2f}"
        )

    def _should_global_reacquire(self) -> bool:
        if not self.global_reacquire_enabled:
            return False
        if self.target_state.lost_frames < self.global_reacquire_after:
            return False
        if self.global_reacquire_interval <= 1:
            return True
        return self._frame_count % self.global_reacquire_interval == 0

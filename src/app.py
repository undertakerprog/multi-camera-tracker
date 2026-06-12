import logging
from time import perf_counter

from src.cameras import CameraCaptureWorker
from src.cameras.base import CameraError, CameraSource
from src.common import StageProfiler
from src.detection import Detector, NullDetector
from src.selection import ROISelector
from src.tracking import (
    DynamicROITracker,
    OpenCVObjectTracker,
    TargetState,
    TemplateReacquirer,
    TrackerError,
    states,
)
from src.tracking.opencv_tracker import TrackingResult
from src.ui import DisplayOverlay, OpenCVDisplay


LOG = logging.getLogger(__name__)


class TargetTrackingApp:
    """Coordinates capture, tracking, prediction, re-acquire and display.

    The heavy/blocking camera read runs in :class:`CameraCaptureWorker`; the loop
    here only ever processes the freshest frame. Tracking runs in a dynamic ROI,
    Kalman fills short gaps, the reacquirer recovers longer losses, and an
    optional detector can be plugged in without touching this class.
    """

    def __init__(
        self,
        camera_source: CameraSource,
        tracker=None,
        tracker_type: str = "CSRT",
        tracking_scale: float = 0.75,
        use_roi: bool = True,
        roi_scale: float = 2.5,
        roi_edge_margin: float = 0.15,
        roi_min_window: int = 64,
        smoothing_alpha: float = 0.35,
        max_lost_frames: int = 90,
        reacquire_enabled: bool = True,
        reacquire_min_score: float = 0.62,
        reacquire_search_expansion: float = 3.0,
        reacquire_cooldown: int = 8,
        global_reacquire_enabled: bool = True,
        global_reacquire_after: int = 8,
        global_reacquire_interval: int = 5,
        global_reacquire_score: float = 0.72,
        global_reacquire_scale: float = 0.5,
        template_update_interval: int = 15,
        template_min_confirmations: int = 5,
        verify_interval: int = 20,
        verify_min_score: float = 0.45,
        detector: Detector | None = None,
        profiler: StageProfiler | None = None,
        display: OpenCVDisplay | None = None,
        roi_selector: ROISelector | None = None,
    ) -> None:
        self.camera_source = camera_source
        self.tracker = tracker or self._build_tracker(
            tracker_type, tracking_scale, use_roi, roi_scale, roi_edge_margin, roi_min_window
        )
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
        self.reacquire_cooldown_frames = max(0, reacquire_cooldown)
        self.global_reacquire_enabled = global_reacquire_enabled
        self.global_reacquire_after = global_reacquire_after
        self.global_reacquire_interval = global_reacquire_interval
        self.template_update_interval = template_update_interval
        self.template_min_confirmations = template_min_confirmations
        self.verify_interval = verify_interval
        self.verify_min_score = verify_min_score
        self.detector = detector or NullDetector()
        self.profiler = profiler or StageProfiler(enabled=True)
        self.display = display or OpenCVDisplay()
        self.roi_selector = roi_selector or ROISelector(self.display.window_name)

        self._worker: CameraCaptureWorker | None = None
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._last_center: tuple[int, int] | None = None
        self._status = states.NO_TARGET
        self._frame_count = 0
        self._fps = 0.0
        self._last_frame_time: float | None = None
        self._last_center_log_time = 0.0
        self._last_reacquire_score = 0.0
        self._cooldown = 0
        self._frame_stats = ""
        self._capture_start_time: float | None = None

    @staticmethod
    def _build_tracker(
        tracker_type: str,
        tracking_scale: float,
        use_roi: bool,
        roi_scale: float,
        roi_edge_margin: float,
        roi_min_window: int,
    ):
        if use_roi:
            return DynamicROITracker(
                tracker_type=tracker_type,
                tracking_scale=tracking_scale,
                roi_scale=roi_scale,
                edge_margin=roi_edge_margin,
                min_window=roi_min_window,
            )
        return OpenCVObjectTracker(tracker_type, tracking_scale)

    # -- lifecycle ---------------------------------------------------------
    def run(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        self._worker = CameraCaptureWorker(self.camera_source)
        try:
            LOG.info("Opening camera and starting capture thread")
            self._worker.start()
            self._capture_start_time = perf_counter()
            first_frame = self._read_frame()
            if first_frame is None:
                raise CameraError("No frame received from camera")
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
                with self.profiler.stage("capture"):
                    frame = self._read_frame()
                if frame is None:
                    LOG.warning("Capture timed out without a fresh frame")
                    if not self._worker or not self._worker.is_running():
                        break
                    continue

            image = frame.image
            self._update_fps()
            self._refresh_frame_stats(image)

            with self.profiler.stage("tracking_total"):
                self._process_tracking(image)

            self._maybe_run_detector(image)

            with self.profiler.stage("display"):
                key = self.display.show(image, self._build_overlay())

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

            self.profiler.end_frame()
            frame = None

    # -- per-frame tracking ------------------------------------------------
    def _process_tracking(self, image) -> None:
        if not (self.tracker.initialized or self.reacquirer.has_template):
            return

        result = TrackingResult(ok=False)
        if self.tracker.initialized:
            with self.profiler.stage("tracker.update"):
                result = self.tracker.update(image)
            if result.ok:
                self._last_reacquire_score = 0.0

        previous_status = self._status
        with self.profiler.stage("kalman"):
            state = self.target_state.update(result, image.shape)
        self._status = state.status

        if not result.ok and self.reacquire_enabled and self._cooldown == 0:
            state = self._attempt_reacquire(image, state)
        elif state.confirmed and self._status == states.TRACKING:
            self._on_confirmed_tracking(image, state)

        self._last_bbox = state.bbox
        self._last_center = state.center

        if state.expired and self.tracker.initialized:
            # Drop the tracker but keep object memory so reacquire can continue.
            self.tracker.reset()
            if previous_status != states.TARGET_STALE:
                LOG.info("Target stale, continuing prediction and reacquire search")

        if self._cooldown > 0:
            self._cooldown -= 1

    def _attempt_reacquire(self, image, state):
        self._status = states.REACQUIRING
        with self.profiler.stage("local_match"):
            bbox, score = self.reacquirer.search(image, state.bbox)
        if bbox is None and self._should_global_reacquire():
            with self.profiler.stage("global_reacquire"):
                bbox, score = self.reacquirer.search_global(image)
        self._last_reacquire_score = score

        if bbox is None:
            return state

        try:
            self.tracker.initialize(image, bbox)
        except TrackerError as exc:
            LOG.warning("Failed to re-init tracker after reacquire: %s", exc)
            return state

        state = self.target_state.initialize(bbox)
        self._status = states.TRACKING
        self._cooldown = self.reacquire_cooldown_frames
        LOG.info(
            "Target reacquired via %s: x=%d y=%d w=%d h=%d score=%.3f",
            self.reacquirer.last_source,
            *bbox,
            score,
        )
        return state

    def _on_confirmed_tracking(self, image, state) -> None:
        if self._last_center is None or state.bbox is None:
            return

        # Periodic appearance verification guards against a "stuck" box that the
        # tracker keeps confirming while the real object has gone.
        if self.verify_interval > 0 and self._frame_count % self.verify_interval == 0:
            _, score = self.reacquirer.search(image, state.bbox)
            if score < self.verify_min_score:
                self.target_state.mark_lost()
                LOG.info("Appearance verification failed (score=%.2f), distrusting box", score)
                return

        # Adaptive template update only on a repeatedly confirmed observation.
        if (
            self.template_update_interval > 0
            and self._cooldown == 0
            and state.confirmations >= self.template_min_confirmations
            and self._frame_count % self.template_update_interval == 0
        ):
            self.reacquirer.remember(image, state.bbox)

        now = perf_counter()
        if now - self._last_center_log_time >= 1.0:
            LOG.info("Target center: x=%d y=%d", *self._last_center)
            self._last_center_log_time = now

    def _maybe_run_detector(self, image) -> None:
        if not self.detector.should_run(self._frame_count):
            return
        with self.profiler.stage("detector"):
            detections = self.detector.detect(image)
        # Detections are surfaced for future auto-acquire; with no known target
        # classes they are not used to initialise the tracker automatically.
        if detections:
            LOG.debug("Detector produced %d detections", len(detections))

    # -- target management -------------------------------------------------
    def _initialize_target(self, image, bbox: tuple[int, int, int, int]) -> None:
        self.tracker.initialize(image, bbox)
        self.reacquirer.initialize(image, bbox)
        state = self.target_state.initialize(bbox)
        self._last_bbox = state.bbox
        self._last_center = state.center
        self._status = state.status
        self._cooldown = 0
        LOG.info("Selected target bbox: x=%d y=%d w=%d h=%d", *bbox)

    def _reset_target(self) -> None:
        # Manual reset fully clears tracker, Kalman state and object memory.
        self.tracker.reset()
        self.reacquirer.reset()
        self.target_state.reset()
        self._last_bbox = None
        self._last_center = None
        self._status = states.NO_TARGET
        self._last_reacquire_score = 0.0
        self._cooldown = 0
        LOG.info("Target reset")

    def _shutdown(self) -> None:
        LOG.info("Shutting down")
        if self._worker is not None:
            self._worker.stop()
        self.display.close()

    # -- helpers -----------------------------------------------------------
    def _read_frame(self):
        if self._worker is None:
            return None
        frame = self._worker.read_latest(timeout=5.0)
        if frame is None:
            return None
        self._frame_count += 1
        if self._frame_count == 1:
            LOG.info("First %s", self._compute_frame_stats(frame.image))
        return frame

    def _build_overlay(self) -> DisplayOverlay:
        return DisplayOverlay(
            bbox=self._last_bbox,
            selection_bbox=self.roi_selector.pending_bbox,
            roi_window=getattr(self.tracker, "window", None),
            center=self._last_center,
            status=self._status,
            tracker_name=self.tracker.name,
            frame_stats=self._frame_stats,
            telemetry=self._telemetry(),
        )

    def _refresh_frame_stats(self, image) -> None:
        # Min/max/mean over a full frame is not free; refresh ~3x/second.
        if self._frame_count <= 1 or self._frame_count % 20 == 0:
            self._frame_stats = self._compute_frame_stats(image)

    @staticmethod
    def _compute_frame_stats(image) -> str:
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
        dropped = self._worker.dropped if self._worker is not None else 0
        return (
            f"fps={self._fps:.1f} "
            f"drop={dropped} "
            f"lost={self.target_state.lost_frames} "
            f"conf={self._last_reacquire_score:.2f} "
            f"src={self.reacquirer.last_source}"
        )

    def _should_global_reacquire(self) -> bool:
        if not self.global_reacquire_enabled:
            return False
        if self.target_state.lost_frames < self.global_reacquire_after:
            return False
        if self.global_reacquire_interval <= 1:
            return True
        return self._frame_count % self.global_reacquire_interval == 0

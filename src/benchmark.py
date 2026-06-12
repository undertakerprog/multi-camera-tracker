"""Headless benchmark for the capture/tracking pipeline.

Runs two short phases and prints comparable numbers so a change can be measured
before/after:

* capture-only  -- camera throughput, consumer rate, latency, dropped frames;
* with-tracking -- the full per-frame tracking cost when an ROI is supplied,
  including p50/p95 of every pipeline stage.

The benchmark never opens a window, so it is safe to run over SSH on the Jetson.
"""

from __future__ import annotations

import logging
from time import perf_counter, time

from src.cameras import CameraCaptureWorker
from src.cameras.base import CameraSource
from src.common import StageProfiler
from src.tracking import TargetState, TemplateReacquirer, TrackerError
from src.tracking.opencv_tracker import TrackingResult

LOG = logging.getLogger(__name__)


class BenchmarkResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.frames = 0
        self.produced = 0
        self.dropped = 0
        self.elapsed = 0.0
        self.latencies_ms: list[float] = []
        self.stage_stats: dict[str, dict[str, float]] = {}

    @property
    def processing_fps(self) -> float:
        return self.frames / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def capture_fps(self) -> float:
        return self.produced / self.elapsed if self.elapsed > 0 else 0.0

    def latency(self, fraction: float) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        rank = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
        return ordered[rank]


def run_benchmark(
    camera_source: CameraSource,
    tracking_config: dict,
    roi: tuple[int, int, int, int] | None = None,
    capture_seconds: float = 3.0,
    tracking_seconds: float = 4.0,
) -> list[BenchmarkResult]:
    worker = CameraCaptureWorker(camera_source)
    results: list[BenchmarkResult] = []
    worker.start()
    try:
        results.append(_run_capture_phase(worker, capture_seconds))
        if roi is not None:
            results.append(
                _run_tracking_phase(worker, tracking_config, roi, tracking_seconds)
            )
        else:
            LOG.info("No ROI provided; skipping tracking benchmark phase")
    finally:
        worker.stop()
    return results


def _run_capture_phase(worker: CameraCaptureWorker, seconds: float) -> BenchmarkResult:
    result = BenchmarkResult("capture-only")
    start_produced = worker.produced
    start = perf_counter()
    while perf_counter() - start < seconds:
        frame = worker.read_latest(timeout=5.0)
        if frame is None:
            break
        result.frames += 1
        result.latencies_ms.append((time() - frame.timestamp) * 1000.0)
    result.elapsed = perf_counter() - start
    result.produced = worker.produced - start_produced
    result.dropped = worker.dropped
    return result


def _run_tracking_phase(
    worker: CameraCaptureWorker,
    tracking_config: dict,
    roi: tuple[int, int, int, int],
    seconds: float,
) -> BenchmarkResult:
    from src.app import TargetTrackingApp  # local import to avoid display deps at module load

    result = BenchmarkResult("with-tracking")
    profiler = StageProfiler(enabled=True, log_interval_s=0)

    # Reuse the exact tracker construction the app uses for parity.
    tracker = TargetTrackingApp._build_tracker(
        tracking_config.get("tracker", "CSRT"),
        tracking_config.get("scale", 0.75),
        tracking_config.get("use_roi", True),
        tracking_config.get("roi_scale", 2.5),
        tracking_config.get("roi_edge_margin", 0.15),
        tracking_config.get("roi_min_window", 64),
    )
    reacquirer = TemplateReacquirer(
        search_expansion=tracking_config.get("reacquire_search", 3.0),
        min_score=tracking_config.get("reacquire_score", 0.62),
        global_min_score=tracking_config.get("global_reacquire_score", 0.72),
        global_search_scale=tracking_config.get("global_reacquire_scale", 0.5),
    )
    target_state = TargetState(
        smoothing_alpha=tracking_config.get("smooth_alpha", 0.35),
        max_lost_frames=tracking_config.get("max_lost_frames", 90),
    )

    first = worker.read_latest(timeout=5.0)
    if first is None:
        return result
    try:
        tracker.initialize(first.image, roi)
        reacquirer.initialize(first.image, roi)
        target_state.initialize(roi)
    except TrackerError as exc:
        LOG.error("Benchmark tracker init failed: %s", exc)
        return result

    start_produced = worker.produced
    start = perf_counter()
    while perf_counter() - start < seconds:
        frame = worker.read_latest(timeout=5.0)
        if frame is None:
            break
        image = frame.image
        with profiler.stage("tracker.update"):
            tracking = tracker.update(image) if tracker.initialized else TrackingResult(ok=False)
        with profiler.stage("kalman"):
            state = target_state.update(tracking, image.shape)
        if not tracking.ok:
            with profiler.stage("local_match"):
                reacquirer.search(image, state.bbox)
        result.frames += 1
        result.latencies_ms.append((time() - frame.timestamp) * 1000.0)
        profiler.end_frame()

    result.elapsed = perf_counter() - start
    result.produced = worker.produced - start_produced
    result.dropped = worker.dropped
    result.stage_stats = profiler.stats()
    return result


def format_report(results: list[BenchmarkResult]) -> str:
    lines = ["=== Benchmark report ==="]
    for result in results:
        lines.append(f"\n[{result.name}] {result.elapsed:.1f}s, {result.frames} frames")
        lines.append(
            f"  capture_fps={result.capture_fps:.1f} "
            f"processing_fps={result.processing_fps:.1f} "
            f"dropped={result.dropped}"
        )
        lines.append(
            f"  latency p50={result.latency(0.50):.1f}ms "
            f"p95={result.latency(0.95):.1f}ms"
        )
        for stage, stat in result.stage_stats.items():
            lines.append(
                f"  stage {stage:<16} "
                f"p50={stat['p50']:.2f}ms p95={stat['p95']:.2f}ms "
                f"max={stat['max']:.2f}ms"
            )
    return "\n".join(lines)

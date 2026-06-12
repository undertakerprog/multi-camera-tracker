"""Lightweight per-stage timing for the tracking pipeline.

The profiler records the duration of named stages using ``time.perf_counter``
and keeps a bounded ring buffer of recent samples so it can report
mean/max/p50/p95 without unbounded memory growth. It is intentionally cheap:
when disabled the context manager does almost nothing, so it can stay enabled in
production telemetry.
"""

from __future__ import annotations

import logging
from collections import deque
from time import perf_counter
from typing import Dict, Iterable

LOG = logging.getLogger(__name__)


class _StageTimer:
    """Context manager returned by :meth:`StageProfiler.stage`."""

    __slots__ = ("_profiler", "_name", "_start")

    def __init__(self, profiler: "StageProfiler", name: str) -> None:
        self._profiler = profiler
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "_StageTimer":
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        elapsed_ms = (perf_counter() - self._start) * 1000.0
        self._profiler.record(self._name, elapsed_ms)


class _NullTimer:
    __slots__ = ()

    def __enter__(self) -> "_NullTimer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


_NULL_TIMER = _NullTimer()


class StageProfiler:
    """Collects timing samples for named pipeline stages.

    Parameters
    ----------
    enabled:
        When ``False`` the profiler is a no-op (zero allocation per stage).
    window:
        Number of recent samples kept per stage for percentile statistics.
    log_interval_s:
        Period between automatic INFO summaries. ``0`` disables logging.
    """

    def __init__(
        self,
        enabled: bool = True,
        window: int = 600,
        log_interval_s: float = 5.0,
    ) -> None:
        self.enabled = enabled
        self.window = max(1, window)
        self.log_interval_s = log_interval_s
        self._samples: Dict[str, deque] = {}
        self._order: list[str] = []
        self._frames = 0
        self._last_log_time = perf_counter()

    def stage(self, name: str):
        """Time a block of code: ``with profiler.stage("resize"): ...``."""
        if not self.enabled:
            return _NULL_TIMER
        return _StageTimer(self, name)

    def record(self, name: str, elapsed_ms: float) -> None:
        """Record a single timing sample for ``name`` (milliseconds)."""
        if not self.enabled:
            return
        bucket = self._samples.get(name)
        if bucket is None:
            bucket = deque(maxlen=self.window)
            self._samples[name] = bucket
            self._order.append(name)
        bucket.append(elapsed_ms)

    def end_frame(self) -> None:
        """Mark the end of a frame and optionally emit a periodic summary."""
        if not self.enabled:
            return
        self._frames += 1
        if self.log_interval_s <= 0:
            return
        now = perf_counter()
        if now - self._last_log_time >= self.log_interval_s:
            self._last_log_time = now
            LOG.info("profile %s", self.format_summary())

    def stats(self) -> Dict[str, Dict[str, float]]:
        """Return ``{stage: {mean, max, p50, p95, count}}`` in record order."""
        result: Dict[str, Dict[str, float]] = {}
        for name in self._order:
            samples = self._samples.get(name)
            if not samples:
                continue
            result[name] = _summarize(samples)
        return result

    def format_summary(self, stages: Iterable[str] | None = None) -> str:
        stats = self.stats()
        names = list(stages) if stages is not None else list(stats.keys())
        parts = []
        for name in names:
            entry = stats.get(name)
            if entry is None:
                continue
            parts.append(
                f"{name}=avg{entry['mean']:.1f}/p95{entry['p95']:.1f}/max{entry['max']:.1f}ms"
            )
        return " ".join(parts)

    def reset(self) -> None:
        self._samples.clear()
        self._order.clear()
        self._frames = 0
        self._last_log_time = perf_counter()

    @property
    def frames(self) -> int:
        return self._frames


def _summarize(samples) -> Dict[str, float]:
    ordered = sorted(samples)
    count = len(ordered)
    total = 0.0
    maximum = ordered[-1]
    for value in ordered:
        total += value
    return {
        "mean": total / count,
        "max": maximum,
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "count": float(count),
    }


def _percentile(ordered_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile on an already-sorted list."""
    if not ordered_values:
        return 0.0
    count = len(ordered_values)
    if count == 1:
        return ordered_values[0]
    rank = fraction * (count - 1)
    low = int(rank)
    high = min(low + 1, count - 1)
    weight = rank - low
    return ordered_values[low] * (1.0 - weight) + ordered_values[high] * weight

from src.common.profiling import StageProfiler, _percentile


def test_percentile_basic():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(values, 0.0) == 1.0
    assert _percentile(values, 1.0) == 5.0
    assert _percentile(values, 0.5) == 3.0


def test_stage_records_and_summarizes():
    profiler = StageProfiler(enabled=True, log_interval_s=0)
    for value in (2.0, 4.0, 6.0):
        profiler.record("stage", value)
    stats = profiler.stats()
    assert "stage" in stats
    assert stats["stage"]["count"] == 3
    assert stats["stage"]["mean"] == 4.0
    assert stats["stage"]["max"] == 6.0


def test_disabled_profiler_is_noop():
    profiler = StageProfiler(enabled=False)
    with profiler.stage("x"):
        pass
    profiler.record("x", 5.0)
    assert profiler.stats() == {}


def test_window_bounds_samples():
    profiler = StageProfiler(enabled=True, window=2, log_interval_s=0)
    for value in (1.0, 2.0, 3.0):
        profiler.record("s", value)
    # Only the last two samples are kept.
    assert profiler.stats()["s"]["count"] == 2
    assert profiler.stats()["s"]["mean"] == 2.5

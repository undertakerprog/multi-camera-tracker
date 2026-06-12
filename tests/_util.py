"""Tiny helpers so the suite runs both under pytest and via run_tests.py."""

from contextlib import contextmanager


@contextmanager
def raises(exc_type):
    """Assert the wrapped block raises ``exc_type`` (pytest.raises-like)."""
    try:
        yield
    except exc_type:
        return
    except Exception as other:  # noqa: BLE001
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(other).__name__}: {other}"
        )
    raise AssertionError(f"expected {exc_type.__name__}, nothing raised")

"""Threaded capture so a slow tracker never throttles the camera.

``CameraCaptureWorker`` runs the blocking :meth:`CameraSource.read` loop on its
own thread and keeps only the single freshest frame (queue depth 1). The
consumer (tracking/UI loop) asks for the latest frame; frames produced while the
consumer was busy are counted as *dropped* rather than queued, which guarantees
the tracker always works on a fresh frame and end-to-end latency stays low.
"""

from __future__ import annotations

import logging
import threading

from src.cameras.base import CameraError, CameraSource
from src.common import Frame

LOG = logging.getLogger(__name__)


class CameraCaptureWorker:
    def __init__(self, camera_source: CameraSource, name: str = "camera-capture") -> None:
        self._source = camera_source
        self._name = name
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._running = threading.Event()

        self._latest: Frame | None = None
        self._latest_seq = 0
        self._consumed_seq = 0
        self._produced = 0
        self._consumed = 0
        self._dropped = 0
        self._error: BaseException | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Open the camera, begin acquisition and launch the capture thread."""
        self._source.open()
        self._source.start()
        self._running.set()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and release the camera. Safe to call more than once."""
        self._running.clear()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        # Wake any consumer blocked in read_latest so it can exit.
        with self._frame_ready:
            self._frame_ready.notify_all()
        try:
            self._source.stop()
        except CameraError as exc:
            LOG.warning("Failed to stop camera cleanly: %s", exc)
        finally:
            self._source.close()

    # -- consumer API ------------------------------------------------------
    def read_latest(self, timeout: float | None = 5.0) -> Frame | None:
        """Return the freshest unconsumed frame, or ``None`` on timeout.

        Blocks until a frame newer than the last consumed one is available so the
        caller never re-processes a stale frame. Re-raises any capture error from
        the worker thread instead of hiding it.
        """
        with self._frame_ready:
            ready = self._frame_ready.wait_for(
                lambda: self._latest_seq > self._consumed_seq
                or self._error is not None
                or not self._running.is_set(),
                timeout=timeout,
            )
            if self._error is not None:
                error = self._error
                self._error = None
                raise error
            if not ready or self._latest_seq <= self._consumed_seq:
                return None
            self._consumed_seq = self._latest_seq
            self._consumed += 1
            return self._latest

    # -- stats -------------------------------------------------------------
    @property
    def produced(self) -> int:
        return self._produced

    @property
    def consumed(self) -> int:
        return self._consumed

    @property
    def dropped(self) -> int:
        return self._dropped

    def stats(self) -> dict[str, int]:
        return {
            "produced": self._produced,
            "consumed": self._consumed,
            "dropped": self._dropped,
        }

    def is_running(self) -> bool:
        return self._running.is_set()

    # -- worker thread -----------------------------------------------------
    def _run(self) -> None:
        while self._running.is_set():
            try:
                frame = self._source.read()
            except CameraError as exc:
                with self._frame_ready:
                    self._error = exc
                    self._frame_ready.notify_all()
                LOG.error("Capture thread stopping after camera error: %s", exc)
                return
            except Exception as exc:  # noqa: BLE001 - surface, never swallow
                with self._frame_ready:
                    self._error = CameraError(f"Unexpected capture failure: {exc}")
                    self._frame_ready.notify_all()
                LOG.exception("Capture thread crashed")
                return

            with self._frame_ready:
                # An un-consumed previous frame is overwritten -> a real drop.
                if self._latest is not None and self._latest_seq > self._consumed_seq:
                    self._dropped += 1
                self._latest = frame
                self._latest_seq += 1
                self._produced += 1
                self._frame_ready.notify_all()

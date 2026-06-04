from abc import ABC, abstractmethod

from src.common import Frame


class CameraError(RuntimeError):
    """Raised when a camera source cannot be opened or read."""


class CameraSource(ABC):
    """Common interface for all camera implementations."""

    @abstractmethod
    def open(self) -> None:
        """Open device resources."""

    @abstractmethod
    def start(self) -> None:
        """Start frame acquisition."""

    @abstractmethod
    def read(self) -> Frame:
        """Read one frame from the source."""

    @abstractmethod
    def stop(self) -> None:
        """Stop frame acquisition."""

    @abstractmethod
    def close(self) -> None:
        """Release device resources."""

    def __enter__(self) -> "CameraSource":
        self.open()
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
        self.close()

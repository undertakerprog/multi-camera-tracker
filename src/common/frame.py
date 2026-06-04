from dataclasses import dataclass, field
from time import time
from typing import Any

import numpy as np


@dataclass(slots=True)
class Frame:
    """Unified video frame passed between subsystems."""

    image: np.ndarray
    timestamp: float = field(default_factory=time)
    source_id: str = "camera"
    sequence_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

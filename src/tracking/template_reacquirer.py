import cv2
import numpy as np


class TemplateReacquirer:
    def __init__(
        self,
        search_expansion: float = 3.0,
        min_score: float = 0.62,
    ) -> None:
        if search_expansion < 1:
            raise ValueError("search_expansion must be >= 1")
        if min_score <= 0 or min_score > 1:
            raise ValueError("min_score must be > 0 and <= 1")

        self.search_expansion = search_expansion
        self.min_score = min_score
        self._template: np.ndarray | None = None
        self._template_size: tuple[int, int] | None = None

    @property
    def has_template(self) -> bool:
        return self._template is not None and self._template_size is not None

    def initialize(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        x, y, w, h = self._clip_bbox(bbox, image.shape)
        if w <= 1 or h <= 1:
            self.reset()
            return

        crop = image[y : y + h, x : x + w]
        self._template = self._to_gray(crop)
        self._template_size = (w, h)

    def search(
        self,
        image: np.ndarray,
        predicted_bbox: tuple[int, int, int, int] | None,
    ) -> tuple[tuple[int, int, int, int] | None, float]:
        if not self.has_template or predicted_bbox is None:
            return None, 0.0

        template = self._template
        template_w, template_h = self._template_size
        search_bbox = self._expanded_bbox(predicted_bbox, image.shape)
        sx, sy, sw, sh = search_bbox

        if sw < template_w or sh < template_h:
            return None, 0.0

        search_area = self._to_gray(image[sy : sy + sh, sx : sx + sw])
        result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)

        if score < self.min_score:
            return None, float(score)

        match_x = sx + max_loc[0]
        match_y = sy + max_loc[1]
        return (match_x, match_y, template_w, template_h), float(score)

    def reset(self) -> None:
        self._template = None
        self._template_size = None

    def _expanded_bbox(
        self,
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        search_w = max(w * self.search_expansion, self._template_size[0])
        search_h = max(h * self.search_expansion, self._template_size[1])

        return self._clip_bbox(
            (
                int(round(cx - search_w / 2)),
                int(round(cy - search_h / 2)),
                int(round(search_w)),
                int(round(search_h)),
            ),
            frame_shape,
        )

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _clip_bbox(
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x, y, w, h = bbox

        x1 = max(0, min(width - 1, x))
        y1 = max(0, min(height - 1, y))
        x2 = max(0, min(width, x + w))
        y2 = max(0, min(height, y + h))

        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

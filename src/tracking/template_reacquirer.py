"""Re-acquire a lost target from appearance memory.

Two search modes are exposed:

* :meth:`search` -- a cheap **local** template search in an expanded window
  around the Kalman prediction. Run only after the tracker fails.
* :meth:`search_global` -- an **expensive** ORB-feature (with template fallback)
  search over the whole frame. Run at a limited frequency, never during normal
  tracking.

Both fuse the raw match score with a position-consistency term so a high but
implausibly located match (false re-acquire) is rejected. Templates come from
:class:`~src.tracking.object_memory.ObjectMemory`, which keeps the stable
selection template separate from adaptive ones.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.tracking.object_memory import ObjectMemory


class TemplateReacquirer:
    def __init__(
        self,
        search_expansion: float = 3.0,
        min_score: float = 0.62,
        global_min_score: float = 0.72,
        global_search_scale: float = 0.5,
        orb_features: int = 1200,
        orb_min_matches: int = 12,
        orb_min_inliers: int = 8,
        max_templates: int = 4,
        max_size_ratio: float = 2.5,
    ) -> None:
        if search_expansion < 1:
            raise ValueError("search_expansion must be >= 1")
        for label, value in (("min_score", min_score), ("global_min_score", global_min_score)):
            if value <= 0 or value > 1:
                raise ValueError(f"{label} must be > 0 and <= 1")
        if global_search_scale <= 0 or global_search_scale > 1:
            raise ValueError("global_search_scale must be > 0 and <= 1")

        self.search_expansion = search_expansion
        self.min_score = min_score
        self.global_min_score = global_min_score
        self.global_search_scale = global_search_scale
        self.orb_min_matches = orb_min_matches
        self.orb_min_inliers = orb_min_inliers
        self.max_size_ratio = max_size_ratio

        self.memory = ObjectMemory(max_templates=max_templates, orb_features=orb_features)
        self._orb = self.memory._orb  # reuse the same detector instance
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last_source = "none"
        self.last_score = 0.0

    @property
    def has_template(self) -> bool:
        return self.memory.has_memory

    def initialize(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        """Capture the stable ground-truth template (resets adaptive bank)."""
        if not self.memory.set_stable(image, bbox):
            self.reset()

    def remember(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        """Add a confirmed adaptive template (call only on confident frames)."""
        self.memory.remember(image, bbox)

    def reset(self) -> None:
        self.memory.clear()
        self.last_source = "none"
        self.last_score = 0.0

    # -- local search ------------------------------------------------------
    def search(
        self,
        image: np.ndarray,
        predicted_bbox: tuple[int, int, int, int] | None,
    ) -> tuple[tuple[int, int, int, int] | None, float]:
        if not self.has_template or predicted_bbox is None:
            return None, 0.0

        search_bbox = self._expanded_bbox(predicted_bbox, image.shape)
        sx, sy, sw, sh = search_bbox
        search_area = self._to_gray(image[sy : sy + sh, sx : sx + sw])

        best_bbox: tuple[int, int, int, int] | None = None
        best_score = 0.0
        for entry in self.memory.entries():
            template_w, template_h = entry.size
            if sw < template_w or sh < template_h:
                continue
            result = cv2.matchTemplate(search_area, entry.image, cv2.TM_CCOEFF_NORMED)
            _, raw_score, _, max_loc = cv2.minMaxLoc(result)
            candidate = (sx + max_loc[0], sy + max_loc[1], template_w, template_h)
            consistency = self._position_consistency(candidate, predicted_bbox, search_bbox)
            fused = _clamp01(raw_score) * (0.6 + 0.4 * consistency)
            if fused > best_score:
                best_score = fused
                best_bbox = candidate

        self.last_score = float(best_score)
        if best_bbox is not None and best_score >= self.min_score:
            self.last_source = "local"
            return best_bbox, float(best_score)
        return None, float(best_score)

    # -- global search -----------------------------------------------------
    def search_global(
        self,
        image: np.ndarray,
    ) -> tuple[tuple[int, int, int, int] | None, float]:
        if not self.has_template:
            return None, 0.0

        feature_bbox, feature_score = self._search_global_features(image)
        if feature_bbox is not None:
            self.last_source = "global-orb"
            self.last_score = float(feature_score)
            return feature_bbox, float(feature_score)

        template_bbox, template_score = self._search_global_template(image)
        self.last_score = float(template_score)
        if template_bbox is not None:
            self.last_source = "global-template"
        return template_bbox, float(template_score)

    # -- helpers -----------------------------------------------------------
    def _expanded_bbox(
        self,
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        stable_size = self.memory.stable_size or (w, h)
        search_w = max(w * self.search_expansion, stable_size[0])
        search_h = max(h * self.search_expansion, stable_size[1])
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
    def _position_consistency(
        candidate: tuple[int, int, int, int],
        predicted: tuple[int, int, int, int],
        search_bbox: tuple[int, int, int, int],
    ) -> float:
        cx = candidate[0] + candidate[2] / 2
        cy = candidate[1] + candidate[3] / 2
        px = predicted[0] + predicted[2] / 2
        py = predicted[1] + predicted[3] / 2
        distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        half_diag = 0.5 * (search_bbox[2] ** 2 + search_bbox[3] ** 2) ** 0.5
        if half_diag <= 0:
            return 1.0
        return max(0.0, 1.0 - distance / half_diag)

    def _search_global_features(
        self,
        image: np.ndarray,
    ) -> tuple[tuple[int, int, int, int] | None, float]:
        keypoints, descriptors = self.memory.stable_descriptors()
        if descriptors is None or keypoints is None or len(keypoints) < self.orb_min_matches:
            return None, 0.0

        frame_gray = self._to_gray(image)
        frame_keypoints, frame_descriptors = self._orb.detectAndCompute(frame_gray, None)
        if frame_descriptors is None or len(frame_keypoints) < self.orb_min_matches:
            return None, 0.0

        matches = self._matcher.knnMatch(descriptors, frame_descriptors, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) != 2:
                continue
            best, second = match_pair
            if best.distance < 0.75 * second.distance:
                good_matches.append(best)

        if len(good_matches) < self.orb_min_matches:
            return None, 0.0

        template_points = np.float32(
            [keypoints[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        frame_points = np.float32(
            [frame_keypoints[match.trainIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(template_points, frame_points, cv2.RANSAC, 5.0)
        if homography is None or mask is None:
            return None, 0.0

        inliers = int(mask.sum())
        if inliers < self.orb_min_inliers:
            return None, inliers / max(1, self.orb_min_inliers)

        template_w, template_h = self.memory.stable_size
        corners = np.float32(
            [[0, 0], [template_w, 0], [template_w, template_h], [0, template_h]]
        ).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(transformed.astype(np.float32))
        if not self._size_is_plausible((w, h), (template_w, template_h)):
            return None, 0.0
        bbox = self._clip_bbox((x, y, w, h), image.shape)
        score = min(1.0, inliers / max(self.orb_min_matches, len(good_matches) * 0.5))
        return bbox, float(score)

    def _search_global_template(
        self,
        image: np.ndarray,
    ) -> tuple[tuple[int, int, int, int] | None, float]:
        frame_gray = self._to_gray(image)
        scale = self.global_search_scale
        scaled_frame = frame_gray
        if scale != 1:
            scaled_frame = cv2.resize(
                frame_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        best_bbox: tuple[int, int, int, int] | None = None
        best_score = 0.0
        for entry in self.memory.entries():
            template = entry.image
            if scale != 1:
                template = cv2.resize(
                    template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
            if (
                scaled_frame.shape[1] < template.shape[1]
                or scaled_frame.shape[0] < template.shape[0]
            ):
                continue
            result = cv2.matchTemplate(scaled_frame, template, cv2.TM_CCOEFF_NORMED)
            _, raw_score, _, max_loc = cv2.minMaxLoc(result)
            if raw_score > best_score:
                template_w, template_h = entry.size
                best_score = raw_score
                best_bbox = (
                    int(round(max_loc[0] / scale)),
                    int(round(max_loc[1] / scale)),
                    template_w,
                    template_h,
                )

        if best_bbox is not None and best_score >= self.global_min_score:
            return best_bbox, float(best_score)
        return None, float(best_score)

    def _size_is_plausible(
        self,
        candidate_size: tuple[int, int],
        reference_size: tuple[int, int],
    ) -> bool:
        cw, ch = candidate_size
        rw, rh = reference_size
        if cw <= 1 or ch <= 1 or rw <= 1 or rh <= 1:
            return False
        ratio_w = max(cw / rw, rw / cw)
        ratio_h = max(ch / rh, rh / ch)
        return ratio_w <= self.max_size_ratio and ratio_h <= self.max_size_ratio

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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

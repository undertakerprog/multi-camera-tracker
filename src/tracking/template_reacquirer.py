import cv2
import numpy as np


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
    ) -> None:
        if search_expansion < 1:
            raise ValueError("search_expansion must be >= 1")
        if min_score <= 0 or min_score > 1:
            raise ValueError("min_score must be > 0 and <= 1")
        if global_min_score <= 0 or global_min_score > 1:
            raise ValueError("global_min_score must be > 0 and <= 1")
        if global_search_scale <= 0 or global_search_scale > 1:
            raise ValueError("global_search_scale must be > 0 and <= 1")

        self.search_expansion = search_expansion
        self.min_score = min_score
        self.global_min_score = global_min_score
        self.global_search_scale = global_search_scale
        self.orb_min_matches = orb_min_matches
        self.orb_min_inliers = orb_min_inliers
        self._orb = cv2.ORB_create(nfeatures=orb_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._template: np.ndarray | None = None
        self._template_size: tuple[int, int] | None = None
        self._template_keypoints = None
        self._template_descriptors = None

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
        self._template_keypoints, self._template_descriptors = self._orb.detectAndCompute(
            self._template,
            None,
        )

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

    def search_global(self, image: np.ndarray) -> tuple[tuple[int, int, int, int] | None, float]:
        if not self.has_template:
            return None, 0.0

        feature_bbox, feature_score = self._search_global_features(image)
        if feature_bbox is not None:
            return feature_bbox, feature_score

        return self._search_global_template(image)

    def reset(self) -> None:
        self._template = None
        self._template_size = None
        self._template_keypoints = None
        self._template_descriptors = None

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

    def _search_global_features(
        self,
        image: np.ndarray,
    ) -> tuple[tuple[int, int, int, int] | None, float]:
        if (
            self._template_descriptors is None
            or self._template_keypoints is None
            or len(self._template_keypoints) < self.orb_min_matches
        ):
            return None, 0.0

        frame_gray = self._to_gray(image)
        frame_keypoints, frame_descriptors = self._orb.detectAndCompute(frame_gray, None)
        if frame_descriptors is None or len(frame_keypoints) < self.orb_min_matches:
            return None, 0.0

        matches = self._matcher.knnMatch(self._template_descriptors, frame_descriptors, k=2)
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
            [self._template_keypoints[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        frame_points = np.float32(
            [frame_keypoints[match.trainIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(
            template_points,
            frame_points,
            cv2.RANSAC,
            5.0,
        )
        if homography is None or mask is None:
            return None, 0.0

        inliers = int(mask.sum())
        if inliers < self.orb_min_inliers:
            return None, inliers / max(1, self.orb_min_inliers)

        template_w, template_h = self._template_size
        corners = np.float32(
            [
                [0, 0],
                [template_w, 0],
                [template_w, template_h],
                [0, template_h],
            ]
        ).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(transformed.astype(np.float32))
        bbox = self._clip_bbox((x, y, w, h), image.shape)
        score = min(1.0, inliers / max(self.orb_min_matches, len(good_matches) * 0.5))
        return bbox, float(score)

    def _search_global_template(
        self,
        image: np.ndarray,
    ) -> tuple[tuple[int, int, int, int] | None, float]:
        template_w, template_h = self._template_size
        frame_gray = self._to_gray(image)
        template = self._template
        scale = self.global_search_scale

        if scale != 1:
            frame_gray = cv2.resize(
                frame_gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            template = cv2.resize(
                template,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        if frame_gray.shape[1] < template.shape[1] or frame_gray.shape[0] < template.shape[0]:
            return None, 0.0

        result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)
        if score < self.global_min_score:
            return None, float(score)

        match_x = int(round(max_loc[0] / scale))
        match_y = int(round(max_loc[1] / scale))
        return (match_x, match_y, template_w, template_h), float(score)

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

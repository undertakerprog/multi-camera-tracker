import cv2


class ROISelector:
    def __init__(self, window_name: str = "Target Tracker") -> None:
        self.window_name = window_name
        self._drag_start: tuple[int, int] | None = None
        self._pending_bbox: tuple[int, int, int, int] | None = None
        self._selected_bbox: tuple[int, int, int, int] | None = None

    def attach(self) -> None:
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    @property
    def pending_bbox(self) -> tuple[int, int, int, int] | None:
        return self._pending_bbox

    def pop_selected(self) -> tuple[int, int, int, int] | None:
        bbox = self._selected_bbox
        self._selected_bbox = None
        return bbox

    def _on_mouse(self, event, x: int, y: int, flags, userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = (x, y)
            self._pending_bbox = (x, y, 0, 0)
            return

        if event == cv2.EVENT_MOUSEMOVE and self._drag_start is not None:
            self._pending_bbox = self._make_bbox(self._drag_start, (x, y))
            return

        if event == cv2.EVENT_LBUTTONUP and self._drag_start is not None:
            bbox = self._make_bbox(self._drag_start, (x, y))
            self._drag_start = None
            self._pending_bbox = None
            if bbox[2] > 0 and bbox[3] > 0:
                self._selected_bbox = bbox

    @staticmethod
    def _make_bbox(
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        x1, y1 = start
        x2, y2 = end
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        return x, y, w, h

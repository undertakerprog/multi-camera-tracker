import numpy as np

from src.tracking.template_reacquirer import TemplateReacquirer


def _scene(square_xy):
    image = np.zeros((200, 200), dtype=np.uint8)
    x, y = square_xy
    image[y : y + 20, x : x + 20] = 255
    # a little texture so ORB/template have signal
    image[y + 5 : y + 8, x + 5 : x + 15] = 128
    return image


def test_local_search_finds_template_near_prediction():
    reacquirer = TemplateReacquirer(min_score=0.5)
    reacquirer.initialize(_scene((100, 100)), (100, 100, 20, 20))

    # Target moved slightly; prediction is close.
    bbox, score = reacquirer.search(_scene((108, 104)), (104, 102, 20, 20))
    assert bbox is not None
    assert score >= 0.5
    assert reacquirer.last_source == "local"
    # Found near the true location.
    assert abs(bbox[0] - 108) <= 4
    assert abs(bbox[1] - 104) <= 4


def test_local_search_rejects_when_absent_from_window():
    reacquirer = TemplateReacquirer(min_score=0.6, search_expansion=2.0)
    reacquirer.initialize(_scene((100, 100)), (100, 100, 20, 20))

    # Empty region far from any patch -> no confident match.
    empty = np.zeros((200, 200), dtype=np.uint8)
    bbox, _ = reacquirer.search(empty, (20, 20, 20, 20))
    assert bbox is None


def test_reset_drops_memory():
    reacquirer = TemplateReacquirer()
    reacquirer.initialize(_scene((100, 100)), (100, 100, 20, 20))
    assert reacquirer.has_template
    reacquirer.reset()
    assert not reacquirer.has_template

import numpy as np

from src.tracking.object_memory import ObjectMemory


def _image():
    rng = np.zeros((200, 200), dtype=np.uint8)
    rng[50:90, 50:90] = 255  # a distinctive patch
    return rng


def test_set_stable_and_entries():
    memory = ObjectMemory(max_templates=2)
    assert memory.set_stable(_image(), (50, 50, 40, 40))
    assert memory.has_memory
    assert memory.stable_size == (40, 40)
    assert len(memory.entries()) == 1


def test_remember_is_bounded_and_keeps_stable_first():
    memory = ObjectMemory(max_templates=2)
    memory.set_stable(_image(), (50, 50, 40, 40))
    for _ in range(4):
        memory.remember(_image(), (50, 50, 40, 40))
    entries = memory.entries()
    # stable + at most 2 adaptive
    assert len(entries) == 3
    assert entries[0].stable is True
    assert all(not e.stable for e in entries[1:])


def test_set_stable_clears_bank():
    memory = ObjectMemory()
    memory.set_stable(_image(), (50, 50, 40, 40))
    memory.remember(_image(), (50, 50, 40, 40))
    memory.set_stable(_image(), (50, 50, 40, 40))
    assert len(memory.entries()) == 1


def test_tiny_bbox_rejected():
    memory = ObjectMemory(min_template_size=6)
    assert not memory.set_stable(_image(), (10, 10, 2, 2))
    assert not memory.has_memory


def test_clear():
    memory = ObjectMemory()
    memory.set_stable(_image(), (50, 50, 40, 40))
    memory.clear()
    assert not memory.has_memory

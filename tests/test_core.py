"""Core checks (see PLAN.md section 13). Tests are added as each module is implemented."""
from src.config import skeleton_bar, skeleton_loop


def test_skeleton_shape_and_hits():
    bar = skeleton_bar()
    assert bar.shape == (16, 9)
    assert bar[:, 0].sum() == 4 and bar[:, 1].sum() == 2
    assert bar[:, 2:].sum() == 0
    assert skeleton_loop(8).shape == (128, 9)

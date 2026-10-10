"""Core checks (see PLAN.md section 13). Tests are added as each module is implemented."""
import numpy as np

from src.config import skeleton_bar, skeleton_loop
from src.data import PITCH_TO_CHANNEL, make_windows, quantize, trim_empty


def test_skeleton_shape_and_hits():
    bar = skeleton_bar()
    assert bar.shape == (16, 9)
    assert bar[:, 0].sum() == 4 and bar[:, 1].sum() == 2
    assert bar[:, 2:].sum() == 0
    assert skeleton_loop(8).shape == (128, 9)


# --- data ---


def test_pitch_mapping_known_pitches():
    assert PITCH_TO_CHANNEL[36] == 0   # kick
    assert PITCH_TO_CHANNEL[38] == 1   # snare head
    assert PITCH_TO_CHANNEL[40] == 1   # snare rim
    assert PITCH_TO_CHANNEL[44] == 2   # pedal hat -> closed hat
    assert PITCH_TO_CHANNEL[26] == 3   # open hat edge
    assert PITCH_TO_CHANNEL[43] == 4   # floor tom -> low tom
    assert PITCH_TO_CHANNEL[48] == 6   # tom 1 -> high tom
    assert PITCH_TO_CHANNEL[49] == 7   # crash
    assert PITCH_TO_CHANNEL[53] == 8   # ride bell


def test_quantize_known_times():
    bpm = 120  # one 16th = 0.125 s, one bar = 2 s
    starts = np.array([0.0, 0.51, 0.97, 2.0, 3.94])  # steps 0, 4, 8, 16 (bar 2 step 0), 32 (beyond)
    pitches = np.array([36, 38, 42, 36, 49])
    bars, stats = quantize(starts, pitches, bpm, end=4.0)
    assert bars.shape == (2, 16, 9)
    assert bars[0, 0, 0] == 1 and bars[0, 4, 1] == 1 and bars[0, 8, 2] == 1
    assert bars[1, 0, 0] == 1
    assert bars.sum() == 4 and stats["after_last_bar"] == 1


def test_quantize_unmappable_dropped():
    bars, stats = quantize(np.array([0.0, 0.5]), np.array([36, 99]), 120, end=2.0)
    assert bars.sum() == 1 and stats["unmappable"] == 1


def test_make_windows_discards_short_files():
    long, short = np.zeros((10, 16, 9), np.int8), np.zeros((5, 16, 9), np.int8)
    w = make_windows([long, short], n_bars=8, stride=1)
    assert w.shape == (3, 128, 9)


def test_trim_empty_keeps_middle_pauses():
    bars = np.zeros((6, 16, 9), np.int8)
    bars[2, 0, 0] = bars[4, 0, 0] = 1  # bars 0-1 and 5 empty at the edges, bar 3 empty in the middle
    trimmed = trim_empty(bars)
    assert len(trimmed) == 3 and not trimmed[1].any()

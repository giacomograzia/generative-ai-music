"""Core checks (see PLAN.md section 13). Tests are added as each module is implemented."""
import numpy as np

from src.config import SEEDS, skeleton_bar, skeleton_loop
from src.data import PITCH_TO_CHANNEL, make_windows, quantize, trim_empty
from src.generate import add_fill, rule_loop
from src.metrics import bar_change, density, diversity, exact_bar_rate, repeat_rate, skeleton_accuracy


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


# --- metrics ---


def test_skeleton_accuracy_penalizes_extra_hits():
    perfect = skeleton_loop(8)
    assert skeleton_accuracy(perfect) == 1.0 and exact_bar_rate(perfect) == 1.0
    extra = perfect.copy()
    extra[2, 0] = 1  # an extra kick in bar 1; all required hits still present
    assert skeleton_accuracy(extra) < 1.0
    assert exact_bar_rate(extra) == 7 / 8


def test_diversity_and_density():
    loop = rule_loop(0)
    assert diversity(np.stack([loop, loop])) == 0.0
    assert density(skeleton_loop(8)) == 0.0  # kick and snare are not decoration
    hats = skeleton_loop(8)
    hats[::2, 2] = 1  # closed hat on every 8th note: 8 per bar
    assert density(hats) == 8.0


def test_bar_change():
    bar = np.zeros((16, 9), np.int8)
    bar[::2, 2] = 1
    assert bar_change(np.tile(bar, (8, 1))) == 0.0
    two = np.tile(bar, (2, 1))
    two[16 + 1, 3] = 1  # one decoration cell differs between the two bars
    assert bar_change(two) == 1 / 112


def test_repeat_rate():
    bar = np.zeros((16, 9), np.int8)
    bar[::2, 2] = 1
    loop = np.tile(bar, (8, 1))
    assert repeat_rate(loop) == 1.0
    loop[7 * 16 + 1, 2] = 1  # only the last bar differs: 6 of 7 pairs are repeats
    assert repeat_rate(loop) == 6 / 7
    loop[:, 0] = 1  # kick and snare do not count
    assert repeat_rate(loop) == 6 / 7


# --- generation ---


def test_rule_loop_keeps_skeleton_and_is_seeded():
    for seed in SEEDS:
        loop = rule_loop(seed)
        assert (loop[:, :2] == skeleton_loop(8)[:, :2]).all()
    assert (rule_loop(3) == rule_loop(3)).all()
    assert not (rule_loop(3) == rule_loop(4)).all()


def test_fill_changes_only_toms_at_end_of_last_bar():
    loop = rule_loop(0)
    filled = add_fill(loop)
    changed = np.argwhere(filled != loop)
    assert all(step >= 7 * 16 + 12 and ch in (4, 5, 6) for step, ch in changed)
    assert filled[7 * 16 + 12, 6] == 1 and filled[7 * 16 + 15, 4] == 1



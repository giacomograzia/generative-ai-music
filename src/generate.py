"""Rule baseline (A), free LSTM (B), constrained LSTM (C), and the listening-only fill."""
import numpy as np

from src.config import FILL, N_BARS, N_DRUMS, STEPS_PER_BAR, skeleton_loop

# --- A: rules -------------------------------------------------------------------------------
# Hand-written chance of a hit per drum and position in the bar (16 steps), applied independently
# to every cell on top of the skeleton. Set once so the expected decoration density is close to
# d_GMD (about 10 hits per bar, data_analysis/README.md), then frozen. Not fitted to anything else.
EIGHTHS = np.arange(0, STEPS_PER_BAR, 2)          # 1 & 2 & 3 & 4 &
SIXTEENTH_OFFBEATS = np.arange(1, STEPS_PER_BAR, 2)  # e and a
BEATS = np.arange(0, STEPS_PER_BAR, 4)            # 1 2 3 4
BAR_END = np.arange(13, STEPS_PER_BAR)            # last three 16ths, where tom pickups go


def rule_probabilities() -> np.ndarray:
    """Chance of a hit per (step, drum) for one bar, shape (16, 9); kick/snare (0-1) are 0 here."""
    p = np.zeros((STEPS_PER_BAR, N_DRUMS))
    p[EIGHTHS, 2] = 0.9               # closed hat: 8th notes, the classic hat pattern
    p[SIXTEENTH_OFFBEATS, 2] = 0.1    # closed hat: occasional 16th in between
    p[14, 3] = 0.3                    # open hat: on the "&" of 4, leading into the next bar
    p[BAR_END, 4:7] = 0.05            # low/mid/high tom: rare pickups at the end of the bar
    p[0, 7] = 0.25                    # crash: sometimes on the first beat of a bar
    p[BEATS, 8] = 0.3                 # ride: sometimes on the beat
    return p


def rule_loop(seed: int, n_bars: int = N_BARS) -> np.ndarray:
    """System A: skeleton plus independently sampled decoration. Returns (n_bars * 16, 9) int8."""
    rng = np.random.default_rng(seed)
    p = np.tile(rule_probabilities(), (n_bars, 1))
    loop = (rng.random(p.shape) < p).astype(np.int8)
    loop[loop[:, 3] == 1, 2] = 0      # a hi-hat is either open or closed, not both
    loop[:, :2] = skeleton_loop(n_bars)[:, :2]
    return loop


# --- listening fill -------------------------------------------------------------------------
def add_fill(loop: np.ndarray) -> np.ndarray:
    """Overwrite the toms (channels 4-6) on steps 12-15 of the last bar with the fixed fill.

    Listening clips only (PLAN.md section 7); same for all systems. Returns a copy, (steps, 9).
    """
    out = loop.copy()
    start = len(loop) - STEPS_PER_BAR
    out[start + 12:, 4:7] = 0
    for step, ch in FILL.items():
        out[start + step, ch] = 1
    return out

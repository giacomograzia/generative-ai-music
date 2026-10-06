"""Project-wide constants: drum classes, pitches, skeleton, generation settings, paths."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "output"

# Drum channels: index -> (class name, MIDI pitch used when writing files)
CLASSES = ["kick", "snare", "closed_hat", "open_hat", "low_tom", "mid_tom", "high_tom", "crash", "ride"]
PITCHES = [36, 38, 42, 46, 45, 48, 50, 49, 51]
N_DRUMS = len(CLASSES)
SKELETON_CHANNELS = (0, 1)
DECORATION_CHANNELS = tuple(range(2, N_DRUMS))

STEPS_PER_BAR = 16
KICK_STEPS = (0, 4, 8, 12)
SNARE_STEPS = (4, 12)

# Experiment settings
N_BARS = 8
BPM = 128
N_LOOPS = 30
SEEDS = tuple(range(N_LOOPS))
TEMPERATURE = 1.0
TEMPERATURES_QUALITATIVE = (0.8, 1.0, 1.2)

# Listening fill on the last bar: (step, channel) pairs; high tom=6, mid tom=5, low tom=4
FILL = {12: 6, 13: 6, 14: 5, 15: 4}


def skeleton_bar() -> np.ndarray:
    """Return the skeleton for one bar, shape (16, 9), only channels 0-1 set."""
    bar = np.zeros((STEPS_PER_BAR, N_DRUMS), dtype=np.int8)
    bar[list(KICK_STEPS), 0] = 1
    bar[list(SNARE_STEPS), 1] = 1
    return bar


def skeleton_loop(n_bars: int = N_BARS) -> np.ndarray:
    """Return the skeleton tiled over n_bars, shape (n_bars * 16, 9)."""
    return np.tile(skeleton_bar(), (n_bars, 1))

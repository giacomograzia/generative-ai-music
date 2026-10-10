"""Skeleton accuracy, exact-bar rate, decoration diversity, decoration density, bar-to-bar change, repeat rate.

All functions take loops of shape (n_loops, steps, 9) with steps a multiple of 16; a single loop
(steps, 9) is also accepted. Channels 0-1 are the skeleton, 2-8 the decoration.
"""
import numpy as np

from src.config import DECORATION_CHANNELS, N_DRUMS, SKELETON_CHANNELS, STEPS_PER_BAR, skeleton_bar

DEC = list(DECORATION_CHANNELS)
SKEL = list(SKELETON_CHANNELS)


def _bars(loops: np.ndarray) -> np.ndarray:
    """(n_loops, steps, 9) or (steps, 9) -> (n_loops, n_bars, 16, 9)."""
    loops = loops[None] if loops.ndim == 2 else loops
    return loops.reshape(len(loops), -1, STEPS_PER_BAR, N_DRUMS)


def _skeleton_errors(loops: np.ndarray) -> np.ndarray:
    """Boolean (n_loops, n_bars, 16, 2): kick/snare cells that differ from the skeleton."""
    return _bars(loops)[..., SKEL] != skeleton_bar()[:, SKEL]


def skeleton_accuracy(loops: np.ndarray) -> float:
    """1 - share of kick/snare cells that differ from the skeleton (missing and extra hits both count)."""
    return float(1 - _skeleton_errors(loops).mean())


def exact_bar_rate(loops: np.ndarray) -> float:
    """Share of bars whose kick and snare channels match the skeleton exactly."""
    return float((~_skeleton_errors(loops).any(axis=(2, 3))).mean())


def diversity(loops: np.ndarray) -> float:
    """Mean pairwise normalized Hamming distance between loops, decoration channels only. Needs >= 2 loops."""
    dec = loops[..., DEC].reshape(len(loops), -1).astype(bool)
    n = len(dec)
    dists = [(dec[i] != dec[j]).mean() for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(dists))


def density(loops: np.ndarray) -> float:
    """Average decoration hits per bar (channels 2-8)."""
    return float(_bars(loops)[..., DEC].sum(axis=(2, 3)).mean())


def bar_change(loops: np.ndarray) -> float:
    """Mean normalized Hamming distance between the decoration of consecutive bars within each loop."""
    dec = _bars(loops)[..., DEC]
    return float((dec[:, 1:] != dec[:, :-1]).mean())


def repeat_rate(loops: np.ndarray) -> float:
    """Share of consecutive bar pairs within each loop whose decoration is identical."""
    dec = _bars(loops)[..., DEC]
    return float((dec[:, 1:] == dec[:, :-1]).all(axis=(2, 3)).mean())


def evaluate(loops: np.ndarray) -> dict[str, float]:
    """All metrics for one system: loops (n_loops, steps, 9)."""
    return {"skeleton accuracy": skeleton_accuracy(loops), "exact-bar rate": exact_bar_rate(loops),
            "diversity": diversity(loops), "density": density(loops), "bar-to-bar change": bar_change(loops),
            "repeat rate": repeat_rate(loops)}

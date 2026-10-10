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


# --- follow-up analyses (PLAN.md section 5) ----------------------------------------------------
def change_per_boundary(loops: np.ndarray) -> np.ndarray:
    """Decoration change at each bar boundary (1-2 ... 7-8), averaged over loops. Returns (n_bars - 1,)."""
    dec = _bars(loops)[..., DEC]
    return (dec[:, 1:] != dec[:, :-1]).mean(axis=(0, 2, 3))


def pulse_errors(loops: np.ndarray) -> dict[str, float]:
    """Kick/snare differences from the skeleton, per bar, split by type."""
    bars = _bars(loops)
    kick, snare = bars[..., 0], bars[..., 1]
    beats, backbeats = [0, 4, 8, 12], [4, 12]
    off_beat = [s for s in range(STEPS_PER_BAR) if s not in beats]
    off_back = [s for s in range(STEPS_PER_BAR) if s not in backbeats]
    out = {f"missing kick on beat {k + 1}": float((kick[..., b] == 0).mean()) for k, b in enumerate(beats)}
    out["extra kicks off the beat"] = float(kick[..., off_beat].sum(-1).mean())
    out["missing snare on beat 2"] = float((snare[..., 4] == 0).mean())
    out["missing snare on beat 4"] = float((snare[..., 12] == 0).mean())
    out["extra snares (incl. ghost notes)"] = float(snare[..., off_back].sum(-1).mean())
    return out


def copying(loops: np.ndarray, train_bars: np.ndarray, train_windows: np.ndarray) -> dict[str, float]:
    """How close generated decoration is to the training data.

    loops: (n, steps, 9); train_bars: (m, 16, 9); train_windows: (k, steps, 9) with stride 1.
    Returns the share of bars that appear exactly in the training bars, the mean distance to the nearest
    training bar (share of the 112 decoration cells), and the share of whole loops copied from training.
    """
    import torch  # matrix distances on the GPU when available

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.tensor(_bars(loops)[..., DEC].reshape(-1, STEPS_PER_BAR * len(DEC)), dtype=torch.float32, device=dev)
    ref = torch.tensor(train_bars[..., DEC].reshape(len(train_bars), -1), dtype=torch.float32, device=dev)
    nearest = torch.cat([((g @ (1 - ref).T) + ((1 - g) @ ref.T)).min(1).values for g in gen.split(1024)])
    nearest = (nearest / gen.shape[1]).cpu().numpy()
    seen = {w[:, DEC].tobytes() for w in train_windows.astype(np.int8)}
    copied = np.mean([lp[:, DEC].astype(np.int8).tobytes() in seen for lp in loops])
    return {"bars found in training": float((nearest == 0).mean()), "distance to nearest training bar": float(nearest.mean()),
            "loops copied from training": float(copied)}

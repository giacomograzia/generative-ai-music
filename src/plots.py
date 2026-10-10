"""Drum grids, loss curve, results figure. Styled with src/plot_style.py (call apply_style() first)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.config import CLASSES, STEPS_PER_BAR
from src.plot_style import PALETTE, decorate, legend_below, tint

LABELS = [c.replace("_", " ") for c in CLASSES]
HIT_COLOR = PALETTE[0]      # dusty blue
ONSET_COLOR = PALETTE[1]    # peach
EMPTY_COLOR = tint("#8C8C8C", 0.18)
INK = "#333333"


def seq_cmap() -> LinearSegmentedColormap:
    """One-hue sequential ramp: white (0) -> dusty blue -> a darker step of the same blue (max)."""
    dark = tuple(0.55 * c for c in tint(HIT_COLOR, 1.0))
    return LinearSegmentedColormap.from_list("seq", ["white", HIT_COLOR, dark])


def plot_grid(pattern: np.ndarray, title: str = "", ax: plt.Axes | None = None) -> plt.Axes:
    """Draw a binary pattern (steps, 9) as a step-sequencer grid: drums on y, 16th steps on x.

    Empty cells are faint lanes, hits are filled, a white gap separates cells, bar lines are dark.
    """
    ax = ax or plt.subplots(figsize=(max(6, len(pattern) / 8), 3))[1]
    cmap = LinearSegmentedColormap.from_list("hit", [EMPTY_COLOR, HIT_COLOR])
    ax.imshow(pattern.T, aspect="auto", origin="lower", cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(np.arange(len(pattern) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(CLASSES) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=1)
    for b in range(0, len(pattern) + 1, STEPS_PER_BAR):
        ax.axvline(b - 0.5, color=INK, lw=1.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks(range(len(CLASSES)), LABELS)
    bars = len(pattern) // STEPS_PER_BAR
    ax.set_xticks(np.arange(bars) * STEPS_PER_BAR, [f"bar {b + 1}" for b in range(bars)], ha="left")
    if title:
        ax.set_title(title)
    return ax


def plot_quantization(starts: np.ndarray, channels: np.ndarray, pattern: np.ndarray, step_sec: float,
                      title: str = "", ax: plt.Axes | None = None) -> plt.Axes:
    """Overlay raw onsets (seconds, converted to steps) on the quantized grid (pattern (steps, 9))."""
    ax = plot_grid(pattern, title, ax)
    ax.fill_between([], [], color=HIT_COLOR, label="grid cell it snaps to")  # legend entry only
    ax.scatter(starts / step_sec, channels, s=36, color=ONSET_COLOR, edgecolors="white", linewidths=1.5,
               label="when the drummer actually played", zorder=3)
    return ax


def plot_loss_curve(history: dict) -> plt.Figure:
    """Train and validation loss per epoch (mean BCE per cell), best epoch marked; returns the figure."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    epochs = np.arange(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], color=PALETTE[0], lw=2, label="training")
    ax.plot(epochs, history["val"], color=PALETTE[1], lw=2, label="validation")
    ax.axvline(history["best_epoch"], color=INK, lw=1)
    ax.text(history["best_epoch"], ax.get_ylim()[1], " kept model", va="top", color=INK, fontsize=10)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (binary cross-entropy per cell)")
    legend_below(ax, ncol=2)
    decorate(fig, f"Validation Loss Bottoms Out at Epoch {history['best_epoch']}",
             subtitle="Mean loss per drum cell; the model from the best validation epoch is kept")
    return fig

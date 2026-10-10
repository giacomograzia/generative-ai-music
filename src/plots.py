"""Drum grids, loss curve, results figure. Styled with src/plot_style.py (call apply_style() first)."""
import matplotlib.pyplot as plt
import matplotlib.ticker
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
    """Train and validation loss per epoch (nats per step), best epoch marked; returns the figure."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    epochs = np.arange(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], color=PALETTE[0], lw=2, label="training")
    ax.plot(epochs, history["val"], color=PALETTE[1], lw=2, label="validation")
    ax.axvline(history["best_epoch"], color=INK, lw=1)
    ax.text(history["best_epoch"], ax.get_ylim()[1], " kept model", va="top", color=INK, fontsize=10)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (nats per 16th-note step)")
    legend_below(ax, ncol=2)
    decorate(fig, f"Validation Loss Bottoms Out at Epoch {history['best_epoch']}",
             subtitle="Negative log-likelihood of the real next step; the model from the best validation epoch is kept")
    return fig


# --- step 4: results and follow-up analyses ---------------------------------------------------
# One fixed color per system in every figure (color follows the system, never its rank).
SYSTEM_COLORS = {"GMD reference": "#8C8C8C", "A rules": PALETTE[3], "B free": PALETTE[5],
                 "C constrained": PALETTE[0], "D fine-tuned": PALETTE[4], "D free": tint(PALETTE[4], 0.5)}


def _style_hbar_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_results(summary, metrics: list[str], title: str) -> plt.Figure:
    """Small multiples, one panel per metric: each system's mean (dot) and min-max over training seeds (line).

    summary: DataFrame indexed by system with columns (metric, 'mean'|'min'|'max').
    """
    systems = list(summary.index)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    y = np.arange(len(systems))
    for ax, metric in zip(axes.flat, metrics):
        for yi, sys_name in zip(y, systems):
            lo, mid, hi = (summary.loc[sys_name, (metric, k)] for k in ("min", "mean", "max"))
            color = SYSTEM_COLORS[sys_name]
            ax.plot([lo, hi], [yi, yi], color=color, lw=2.5, solid_capstyle="round")
            ax.scatter([mid], [yi], s=70, color=color, edgecolors="white", linewidths=1.5, zorder=3)
            ax.text(hi, yi, f"  {mid:.2f}" if abs(mid) < 10 else f"  {mid:.1f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, systems if ax in axes[:, 0] else [""] * len(systems))
        ax.invert_yaxis()
        ax.set_title(metric, fontsize=12, loc="left")
        ax.margins(x=0.25)
        if metric == "exact-bar rate":
            ax.set_xlim(-0.08, 1.2)
        _style_hbar_axis(ax)
    decorate(fig, title,
             subtitle="Mean over loops (dot) and range over the 5 trained models (line); A and the GMD reference do not depend on training",
             footnote_text="Skeleton accuracy and exact-bar rate are 1.0 for A, C and D by construction.")
    return fig


def plot_phrase_position(changes: dict[str, np.ndarray], title: str) -> plt.Figure:
    """Decoration change at each bar boundary (1-2 ... 7-8), one line per system."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(1, 8)
    for name, values in changes.items():
        ax.plot(x, values, color=SYSTEM_COLORS[name], lw=2, marker="o", ms=6, mec="white", mew=1.5, label=name)
    ax.set_xticks(x, [f"{i}→{i + 1}" for i in x])
    ax.set_xlabel("bar boundary")
    ax.set_ylabel("share of decoration cells that change")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    legend_below(ax, ncol=len(changes))
    decorate(fig, title,
             subtitle="Decoration change between consecutive bars, averaged over loops (and over the 5 trained models for B-D)")
    return fig


def plot_constraint_cost(allowed_mass: dict[str, np.ndarray], title: str) -> plt.Figure:
    """Per position in the bar: how much probability the model gave to the kick/snare the skeleton forces."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    counts = ["1", "e", "&", "a", "2", "e", "&", "a", "3", "e", "&", "a", "4", "e", "&", "a"]
    x = np.arange(STEPS_PER_BAR)
    width = 0.8 / len(allowed_mass)
    for k, (name, values) in enumerate(allowed_mass.items()):
        ax.bar(x + (k - (len(allowed_mass) - 1) / 2) * width, values, width=width, color=SYSTEM_COLORS[name],
               edgecolor="white", linewidth=0.5, label=name)
    ax.set_xticks(x, counts)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("position in the bar")
    ax.set_ylabel("model's own chance of the forced kick/snare")
    legend_below(ax, ncol=len(allowed_mass))
    decorate(fig, title,
             subtitle="Probability the model gave to the kick/snare values the skeleton then forces, per 16th-note position",
             footnote_text="100% means the model would have played exactly the skeleton's kick and snare there anyway.")
    return fig


def plot_pulse_errors(errors, title: str) -> plt.Figure:
    """Grouped horizontal bars: kick/snare errors per bar by type, one bar per system (DataFrame: types x systems)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    types, systems = list(errors.index), list(errors.columns)
    y = np.arange(len(types))
    h = 0.8 / len(systems)
    for k, name in enumerate(systems):
        ax.barh(y + (k - (len(systems) - 1) / 2) * h, errors[name], height=h, color=SYSTEM_COLORS[name], label=name)
    ax.set_yticks(y, types)
    ax.invert_yaxis()
    ax.set_xlabel("errors per bar (against the skeleton)")
    _style_hbar_axis(ax)
    legend_below(ax, ncol=len(systems))
    decorate(fig, title,
             subtitle="Kick and snare differences from the skeleton, per bar, by type")
    return fig

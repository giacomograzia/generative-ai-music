"""Schematic of the experimental design for the notebook's first section, in the look of the web pages
(white cards, dusty blue and peach, small drum grids). The grids show bar 1 of real loops (seed 0).

Run from the project root after src.evaluate: .venv/bin/python -m src.design_figure
Writes output/figures/experiment_design.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from src.config import N_DRUMS, OUTPUT_DIR, SKELETON_CHANNELS, STEPS_PER_BAR, skeleton_bar
from src.plot_style import PALETTE, apply_style

OUT = OUTPUT_DIR / "figures" / "experiment_design.png"
BLUE, PEACH = PALETTE[0], PALETTE[1]
INK, INK2, LINE, EMPTY, BG = "#222222", "#555555", "#E2E2E2", "#ECECEC", "#FAFAF8"
CELL = 0.9  # drum-grid cell size in axis units


def card(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, fill: str = "white",
         ink: str = INK) -> None:
    """Rounded card with a bold title and smaller body text at its top-left corner."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fill, ec=LINE, lw=1))
    ax.text(x + 1.2, y + h - 1.4, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=ink)
    ax.text(x + 1.2, y + h - 4.2, body, ha="left", va="top", fontsize=9.5, color=ink if fill != "white" else INK2,
            linespacing=1.45)


def grid(ax: plt.Axes, bar: np.ndarray, x: float, y: float) -> None:
    """One bar (16, 9) as a step grid: kick/snare hits peach, decoration hits blue, kick in the bottom row."""
    for step in range(STEPS_PER_BAR):
        for drum in range(N_DRUMS):
            color = EMPTY if not bar[step, drum] else (PEACH if drum in SKELETON_CHANNELS else BLUE)
            ax.add_patch(Rectangle((x + step * CELL, y + drum * CELL), CELL * 0.86, CELL * 0.86, fc=color, lw=0))


def arrow(ax: plt.Axes, start: tuple, end: tuple, color: str = INK2, label: str = "", rad: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=color, lw=1.6,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(*mid, label, ha="center", va="center", fontsize=8.5, color=INK2, backgroundcolor=BG)


def main() -> None:
    loops = np.load(OUTPUT_DIR / "results" / "loops.npz")
    first_bar = lambda key: loops[key][0, :STEPS_PER_BAR]  # noqa: E731  bar 1 of loop seed 0
    apply_style()
    fig = plt.figure(figsize=(14, 6.6), facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 66)
    ax.axis("off")
    gx = lambda x, w: x + w - 1.2 - STEPS_PER_BAR * CELL  # noqa: E731  grid at a card's right edge

    # inputs: the skeleton (kept by A, C, D) and the human drumming the model learns from
    card(ax, 2, 40, 34, 21, "Skeleton", "kick on every\nbeat, snare\non 2 and 4\n\nkept exactly\nin A, C, D")
    grid(ax, skeleton_bar(), gx(2, 34), 42.5)
    card(ax, 2, 6, 34, 21, "Human drumming", "Groove MIDI\nDataset: 10\ndrummers, 12.6 h,\ntrain split\n\n"
         "reference row:\ntest split")
    grid(ax, first_bar("GMD reference"), gx(2, 34), 8.5)

    # the learned model
    card(ax, 43, 10, 22, 17, "LSTM", "predicts the next 16th\nnote from everything\nplayed so far\n\n5 training runs",
         fill=BLUE, ink="white")
    arrow(ax, (36, 18.5), (43, 18.5))

    # the four systems
    systems = [("A. Rules", "skeleton + hand-written\nhit chances", "A rules"),
               ("B. LSTM free", "the model plays\nevery drum", "B free|0"),
               ("C. LSTM constrained", "same model, kick and\nsnare forced to skeleton", "C constrained|0"),
               ("D. Fine-tuned", "LSTM tuned on four-on-\nthe-floor bars, then as C", "D fine-tuned|0")]
    tops = [61, 46, 31, 16]
    for (title, body, key), top in zip(systems, tops):
        card(ax, 74, top - 14, 34, 13, title, body)
        grid(ax, first_bar(key), gx(74, 34), top - 12.2)
        arrow(ax, (108, top - 7.5), (115, top - 7.5))
    arrow(ax, (36, 53.5), (74, 53.5), color=PEACH)  # skeleton -> rules
    for top in tops[1:]:
        arrow(ax, (65, 18.5), (74, top - 7.5), color=BLUE)
    ax.text(54, 8.6, "B and C: same model, same seeds;\nonly the constraint differs", ha="center", va="top",
            fontsize=9, color=INK2, style="italic")

    # evaluation
    card(ax, 115, 2, 23, 59, "Evaluate", "30 loops per system\n(seeds 0-29), 8 bars,\n128 BPM\n\n"
         "Steady: B vs C\n  skeleton accuracy\n  exact-bar rate\n\n"
         "Alive: A vs C\n  diversity\n  density\n  bar-to-bar change\n  repeat rate\n\n"
         "read against real\ndrumming (reference)\n\n+ listening")

    ax.text(2, 64.5, "Experimental design", fontsize=15, fontweight="bold", color=INK, va="top")
    ax.text(48, 64.2, "peach: kick and snare    blue: decoration (hats, toms, cymbals)    grids: bar 1 of loop seed 0",
            fontsize=9, color=INK2, va="top")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=BG, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

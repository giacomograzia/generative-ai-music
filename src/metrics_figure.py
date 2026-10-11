"""Figure explaining the six metrics with small hand-made examples: one low and one high example per metric.
Values are computed with src.metrics, so they are exact for the patterns shown.

Run from the project root: .venv/bin/python -m src.metrics_figure
Writes output/figures/metrics_examples.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from src.config import N_DRUMS, OUTPUT_DIR, SKELETON_CHANNELS, STEPS_PER_BAR
from src.design_figure import BG, BLUE, EMPTY, INK, INK2, LINE, PEACH
from src.metrics import bar_change, density, diversity, exact_bar_rate, repeat_rate, skeleton_accuracy
from src.plot_style import apply_style

OUT = OUTPUT_DIR / "figures" / "metrics_examples.png"
CELL, BAR_GAP = 0.9, 0.8
RED = "#D64545"
N_BARS = 4
CLOSED, OPEN, CRASH, RIDE = 2, 3, 7, 8


def loop(*bars: np.ndarray) -> np.ndarray:
    """Concatenate bars (16, 9) into a loop (steps, 9)."""
    return np.concatenate(bars).astype(np.int8)


def bar(kick=(0, 4, 8, 12), snare=(4, 12), **decoration) -> np.ndarray:
    """One bar (16, 9): kick and snare steps, plus decoration given as channel=steps (e.g. closed=range(0, 16, 2))."""
    b = np.zeros((STEPS_PER_BAR, N_DRUMS), np.int8)
    b[list(kick), 0] = 1
    b[list(snare), 1] = 1
    channels = {"closed": CLOSED, "open": OPEN, "crash": CRASH, "ride": RIDE}
    for name, steps in decoration.items():
        b[list(steps), channels[name]] = 1
    return b


EIGHTHS = range(0, 16, 2)
HAT = bar(closed=EIGHTHS)  # skeleton + 8th-note closed hat: the basic groove every example starts from


def variant(steps_on=(), steps_off=(), channel=CLOSED, base=HAT) -> np.ndarray:
    """The basic groove with a few decoration cells switched on or off."""
    b = base.copy()
    b[list(steps_on), channel] = 1
    b[list(steps_off), channel] = 0
    return b


def examples() -> list[tuple[str, str, str, tuple, tuple]]:
    """Per metric: (name, question, formatter, (low caption, low loops), (high caption, high loops))."""
    rock = bar(kick=(0, 8, 10), snare=(4, 7, 12, 15), closed=EIGHTHS)  # kick on 1 and 3, ghost snares
    extra_kick = bar(kick=(0, 4, 8, 11, 12), closed=EIGHTHS)           # skeleton plus one extra kick
    busy = bar(closed=range(16), ride=EIGHTHS, crash=(0,))
    sparse = bar(closed=range(0, 16, 4))
    return [
        ("Skeleton accuracy", "share of kick/snare cells\nthat match the skeleton", "{:.2f}",
         ("a rock beat: kick on 1 and 3, ghost snares", loop(rock, rock, rock, rock)),
         ("the skeleton in every bar", loop(HAT, HAT, HAT, HAT))),
        ("Exact-bar rate", "share of bars whose kick and\nsnare match exactly", "{:.2f}",
         ("one extra kick per bar: accuracy 0.97, but no exact bar", loop(*[extra_kick] * 4)),
         ("every bar exact", loop(HAT, HAT, HAT, HAT))),
        ("Diversity", "how different the loops are\nfrom each other", "{:.2f}",
         ("three loops, almost the same", [loop(HAT), loop(variant([1])), loop(variant([15]))]),
         ("three loops, all different", [loop(HAT), loop(bar(ride=range(0, 16, 4), open=(14,))),
                                         loop(bar(closed=range(16), crash=(0,)))])),
        ("Density", "decoration hits per bar", "{:.1f}",
         ("hat on the beats only", loop(*[sparse] * 4)),
         ("16th-note hat, ride and crash", loop(*[busy] * 4))),
        ("Bar-to-bar change", "share of decoration cells that\nchange from one bar to the next", "{:.3f}",
         ("one bar, repeated", loop(HAT, HAT, HAT, HAT)),
         ("a new pattern every bar", loop(HAT, bar(ride=range(16)), bar(closed=range(1, 16, 2), crash=(0,)),
                                          bar(ride=EIGHTHS, open=(6, 14))))),
        ("Repeat rate", "share of consecutive bars that\nare exactly the same", "{:.2f}",
         ("a small change in every bar (like the rules)",
          loop(variant([3]), variant([7]), variant([11], [8]), variant([5], [14]))),
         ("repeats, then varies at the end (like drummers)", loop(HAT, HAT, HAT, variant([13, 15], [14], base=bar(closed=EIGHTHS, open=(14,)))))),
    ]


def draw_loop(ax: plt.Axes, pattern: np.ndarray, x: float, y: float) -> float:
    """Draw a loop (steps, 9) as a mini grid with a gap between bars; returns its width in axis units."""
    for step in range(len(pattern)):
        bx = x + step * CELL + (step // STEPS_PER_BAR) * BAR_GAP
        for drum in range(N_DRUMS):
            hit = pattern[step, drum]
            color = EMPTY if not hit else (PEACH if drum in SKELETON_CHANNELS else BLUE)
            ax.add_patch(Rectangle((bx, y + drum * CELL), CELL * 0.86, CELL * 0.86, fc=color, lw=0))
    n_bars = len(pattern) // STEPS_PER_BAR
    return len(pattern) * CELL + (n_bars - 1) * BAR_GAP


# What to point at, per (metric, 0 = low / 1 = high panel): a short label and boxes around cells, each box as
# (loop index in the panel, first step, last step, lowest drum, highest drum); steps count from the loop start.
HIGHLIGHTS = {
    ("Skeleton accuracy", 0): ("bar 1: 5 of 32 kick/snare cells wrong (missing or extra)",
                               [(0, 4, 4, 0, 0), (0, 10, 10, 0, 0), (0, 12, 12, 0, 0), (0, 7, 7, 1, 1), (0, 15, 15, 1, 1)]),
    ("Exact-bar rate", 0): ("one extra kick spoils each bar", [(0, 11 + 16 * b, 11 + 16 * b, 0, 0) for b in range(4)]),
    ("Diversity", 0): ("the only differences", [(1, 1, 1, 2, 2), (2, 15, 15, 2, 2)]),
    ("Density", 0): ("4 decoration hits in this bar", [(0, 0, 15, 2, 8)]),
    ("Density", 1): ("25 decoration hits in this bar", [(0, 0, 15, 2, 8)]),
    ("Bar-to-bar change", 1): ("bar 2 shares no hit with bar 1", [(0, 16, 31, 2, 8)]),
    ("Repeat rate", 0): ("a hat added or removed in every bar",
                         [(0, 3, 3, 2, 2), (0, 23, 23, 2, 2), (0, 40, 40, 2, 2), (0, 43, 43, 2, 2), (0, 53, 53, 2, 2),
                          (0, 62, 62, 2, 2)]),
    ("Repeat rate", 1): ("bars 1-3 identical; bar 4 varies", [(0, 61, 63, 2, 3)]),
}


def cell_x(origin: float, step: int) -> float:
    """Left edge of a step's cell in a grid drawn by draw_loop starting at origin."""
    return origin + step * CELL + (step // STEPS_PER_BAR) * BAR_GAP


def highlight(ax: plt.Axes, origins: list[float], gy: float, bounds: tuple[float, float], label: str,
              boxes: list[tuple]) -> None:
    """Red boxes around cells, and a red label above the grid with an arrow down to the first box.

    bounds: (left, right) of the panel in axis units; the label is shifted to stay inside it.
    """
    pad = 0.2
    for i, (loop_i, s0, s1, d0, d1) in enumerate(boxes):
        x = cell_x(origins[loop_i], s0) - pad
        w = cell_x(origins[loop_i], s1) + CELL * 0.86 + pad - x
        y, h = gy + d0 * CELL - pad, (d1 - d0) * CELL + CELL * 0.86 + 2 * pad
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=RED, lw=1.6, zorder=3))
        if i == 0:
            label_y = gy + 9 * CELL + 1.2
            half = len(label) * 0.33  # rough half-width of the label in axis units
            label_x = min(max(x + w / 2, bounds[0] + half), bounds[1] - half)
            ax.text(label_x, label_y, label, fontsize=8.5, color=RED, ha="center", va="bottom", zorder=4)
            ax.add_patch(FancyArrowPatch((x + w / 2, label_y), (x + w / 2, y + h), arrowstyle="-|>", mutation_scale=8,
                                         color=RED, lw=1, shrinkA=1, shrinkB=0, zorder=4))


METRIC_FN = {"Skeleton accuracy": skeleton_accuracy, "Exact-bar rate": exact_bar_rate, "Diversity": diversity,
             "Density": density, "Bar-to-bar change": bar_change, "Repeat rate": repeat_rate}


def main() -> None:
    apply_style()
    rows = examples()
    panel_w = N_BARS * STEPS_PER_BAR * CELL + (N_BARS - 1) * BAR_GAP + 2.4
    row_h, label_w, top = 16.5, 27, 8
    width, height = label_w + 2 * panel_w + 4, top + len(rows) * row_h + 1
    fig = plt.figure(figsize=(16, 16 * height / width), facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    ax.text(1, height - 1.2, "What the metrics measure", fontsize=15, fontweight="bold", color=INK, va="top")
    for k, head in enumerate(("low", "high")):
        ax.text(label_w + k * (panel_w + 2) + panel_w / 2, height - 1.6, head, fontsize=12, fontweight="bold",
                color=INK2, ha="center", va="top")
    ax.text(1, height - 4.2, "peach: kick and snare   blue: decoration\n4 bars per loop; values computed with src/metrics.py",
            fontsize=9, color=INK2, va="top", linespacing=1.4)

    for r, (name, question, fmt, *panels) in enumerate(rows):
        y0 = height - top - (r + 1) * row_h
        ax.text(1, y0 + row_h - 2, name, fontsize=12, fontweight="bold", color=INK, va="top")
        ax.text(1, y0 + row_h - 4.6, question, fontsize=9.5, color=INK2, va="top", linespacing=1.4)
        for k, (caption, pattern) in enumerate(panels):
            x0 = label_w + k * (panel_w + 2)
            ax.add_patch(FancyBboxPatch((x0, y0 + 0.6), panel_w, row_h - 1.2, boxstyle="round,pad=0,rounding_size=1",
                                        fc="white", ec=LINE, lw=1))
            loops = np.stack(pattern) if isinstance(pattern, list) else pattern
            value = METRIC_FN[name](loops)
            ax.text(x0 + 1.2, y0 + row_h - 1.6, f"{name.lower()} = {fmt.format(value)}", fontsize=10.5,
                    fontweight="bold", color=INK, va="top")
            ax.text(x0 + panel_w - 1.2, y0 + row_h - 1.6, caption, fontsize=9, color=INK2, va="top", ha="right")
            gx, origins = x0 + 1.2, []
            for p in (pattern if isinstance(pattern, list) else [pattern]):
                origins.append(gx)
                gx += draw_loop(ax, p, gx, y0 + 1.6) + 3
            if (name, k) in HIGHLIGHTS:
                highlight(ax, origins, y0 + 1.6, (x0 + 1.2, x0 + panel_w - 1.2), *HIGHLIGHTS[(name, k)])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=BG, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

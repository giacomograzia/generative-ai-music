"""Animated GIF of how the LSTM writes a loop, one 16th note at a time: free (B), then constrained (C).

Uses the real trained model (base, training seed 0) and real sampling (listening seed 0).
Run from the project root: .venv/bin/python -m src.animate
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from src.config import LISTEN_SEEDS, N_BARS, OUTPUT_DIR, STEPS_PER_BAR
from src.generate import Generation, lstm_loop
from src.model import from_combo, to_combo
from src.plot_style import PALETTE, apply_style
from src.plots import EMPTY_COLOR, INK, LABELS, plot_grid
from src.train import load_model

OUT = OUTPUT_DIR / "animation" / "generation.gif"
FPS = 8
TOP_K = 6
SHORT = ["kick", "snare", "hat", "open hat", "low tom", "mid tom", "high tom", "crash", "ride"]
CHOSEN, OPTION, BLOCKED = PALETTE[1], PALETTE[0], "#D0D0D0"
PHASES = {
    False: ("B · free", "The model chooses every drum, kick and snare included."),
    True: ("C · constrained", "Kick and snare are fixed by the skeleton; the model only chooses among options that match."),
}


def combo_name(index: int) -> str:
    drums = [SHORT[k] for k in np.flatnonzero(from_combo(torch.tensor(index)).numpy())]
    return " + ".join(drums) if drums else "(nothing)"


def schedule(n_steps: int) -> list[tuple[int, int]]:
    """(step, stage) per frame. Stage 0 input, 1 probabilities, 2 pick and write; slow first, then faster."""
    frames = []
    for t in range(6):                           # the first steps, slowly, stage by stage
        frames += [(t, 0)] * 4 + [(t, 1)] * 5 + [(t, 2)] * 5
    frames += [(t, 2) for t in range(6, 2 * STEPS_PER_BAR)]          # the rest of bar 1-2, one step per frame
    frames += [(t, 2) for t in range(2 * STEPS_PER_BAR + 3, n_steps, 4)]  # bars 3-8, four steps per frame
    frames += [(n_steps - 1, 3)] * (3 * FPS)     # hold the finished loop
    return frames


def draw_pipeline(ax, gen: Generation, t: int, stage: int, constrained: bool) -> None:
    """Left panel: previous step + position -> LSTM -> probabilities, with the feedback loop."""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    prev = gen.loop[t - 1] if t > 0 else np.zeros(9, int)
    on = stage >= 0
    # previous step as a 9-cell column
    ax.text(0.22, 0.97, "previous step", ha="center", va="top", fontsize=10, color=INK)
    for k in range(9):
        y = 0.12 + k * 0.085
        ax.add_patch(Rectangle((0.18, y), 0.08, 0.07, color=OPTION if prev[k] else EMPTY_COLOR))
        ax.text(0.16, y + 0.035, LABELS[k], ha="right", va="center", fontsize=8, color=INK)
    ax.text(0.22, 0.04, f"+ position in bar: {t % STEPS_PER_BAR + 1} of 16", ha="center", fontsize=9, color=INK)
    # LSTM box
    ax.add_patch(FancyBboxPatch((0.42, 0.38), 0.30, 0.24, boxstyle="round,pad=0.02",
                                facecolor=PALETTE[0] if on else EMPTY_COLOR, edgecolor="none", alpha=0.9))
    ax.text(0.57, 0.54, "LSTM", ha="center", va="center", fontsize=14, weight="bold", color="white")
    ax.text(0.57, 0.45, "3 layers × 256\nmemory of the loop", ha="center", va="center", fontsize=8,
            color="white", linespacing=1.3)
    ax.add_patch(FancyArrowPatch((0.29, 0.50), (0.40, 0.50), arrowstyle="-|>", mutation_scale=15, color=INK))
    ax.add_patch(FancyArrowPatch((0.74, 0.50), (1.02, 0.50), arrowstyle="-|>", mutation_scale=15,
                                 color=INK if stage >= 1 else EMPTY_COLOR, clip_on=False))
    ax.text(0.88, 0.55, "chances for\nevery combination", ha="center", va="bottom", fontsize=8, color=INK)
    if constrained:
        ax.text(0.57, 0.18, "constraint: kick and snare\nmust match the skeleton", ha="center", va="center",
                fontsize=9, color=INK, weight="bold")


def draw_probs(ax, gen: Generation, t: int, stage: int) -> None:
    """Right panel: the model's top combinations for this step; blocked ones grey, the pick highlighted."""
    ax.clear()
    chosen = int(to_combo(torch.tensor(gen.loop[t])))
    order = list(np.argsort(gen.probs[t])[::-1][:TOP_K])
    if chosen not in order:
        order[-1] = chosen
    probs = gen.probs[t][order]
    colors = [BLOCKED if not gen.allowed[t][i] else (CHOSEN if stage >= 2 and i == chosen else OPTION)
              for i in order]
    y = np.arange(len(order))
    ax.barh(y, probs if stage >= 1 else np.zeros_like(probs), height=0.65, color=colors)
    for yi, i, p in zip(y, order, probs):
        label = combo_name(i) + ("" if gen.allowed[t][i] else "   (blocked)")
        ax.text(-0.01, yi, label, ha="right", va="center", fontsize=9,
                color=INK if gen.allowed[t][i] else "#8C8C8C", weight="bold" if stage >= 2 and i == chosen else None)
        if stage >= 1:
            ax.text(p + 0.01, yi, f"{p:.0%}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(0.5, probs.max() * 1.25))
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks([])
    ax.grid(False)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("model's chance for the next step", fontsize=9)
    title = {0: "1. read the previous step", 1: "2. predict the next step",
             2: "3. pick one at random, write it, feed it back", 3: "8 bars written"}[stage]
    ax.set_title(title, loc="left", fontsize=12, weight="bold")


def draw_grid(ax, gen: Generation, t: int, stage: int) -> None:
    """Bottom panel: the loop so far; future steps faded, the current step outlined."""
    ax.clear()
    written = gen.loop.copy()
    upto = t + 1 if stage >= 2 else t
    written[upto:] = 0
    plot_grid(written, ax=ax)
    n = len(written)
    if upto < n:
        ax.add_patch(Rectangle((upto - 0.5, -0.5), n - upto, 9, color="white", alpha=0.55, lw=0))
    if stage < 3:
        ax.add_patch(Rectangle((t - 0.5, -0.5), 1, 9, fill=False, edgecolor=CHOSEN, lw=2.5))
    ax.tick_params(labelsize=8)


def make_animation():
    """Build the figure; returns (fig, update function, frame list)."""
    apply_style()
    model = load_model("base", 0)
    seed = LISTEN_SEEDS[0]
    gens = {c: lstm_loop(model, seed, constrained=c) for c in (False, True)}
    n_steps = N_BARS * STEPS_PER_BAR
    frames = [(c, t, s) for c in (False, True) for t, s in schedule(n_steps)]

    fig = plt.figure(figsize=(12, 6.75), dpi=80)
    ax_pipe = fig.add_axes([0.02, 0.40, 0.40, 0.44])
    ax_probs = fig.add_axes([0.62, 0.42, 0.33, 0.40])
    ax_grid = fig.add_axes([0.08, 0.06, 0.90, 0.27])
    title = fig.text(0.02, 0.96, "", fontsize=18, weight="bold", va="top")
    subtitle = fig.text(0.02, 0.905, "", fontsize=11.5, color="#333333", va="top")

    def update(i):
        constrained, t, stage = frames[i]
        name, desc = PHASES[constrained]
        title.set_text(f"How the model writes drums, one 16th note at a time — {name}")
        subtitle.set_text(desc)
        gen = gens[constrained]
        draw_pipeline(ax_pipe, gen, t, stage, constrained)
        draw_probs(ax_probs, gen, t, stage)
        draw_grid(ax_grid, gen, t, stage)
        return []

    return fig, update, frames


def main() -> None:
    fig, update, frames = make_animation()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=len(frames)).save(OUT, writer=PillowWriter(fps=FPS))
    print(f"{OUT}  frames {len(frames)}  {len(frames) / FPS:.0f} s")


if __name__ == "__main__":
    main()

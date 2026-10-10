"""Make the dataset-analysis figures and audio examples for data_analysis/README.md.

Run from the project root: .venv/bin/python -m data_analysis.make_figures
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pretty_midi
import soundfile as sf

from src.audio import SR, render, step_seconds
from src.config import BPM, N_BARS, STEPS_PER_BAR, skeleton_loop
from src.data import GMD_DIR, PITCH_TO_CHANNEL, load_bars
from src.plot_style import PALETTE, apply_style, decorate, legend_below
from src.plots import INK, LABELS, plot_grid, plot_quantization, seq_cmap

HERE = Path(__file__).resolve().parent
FIG, AUDIO = HERE / "figures", HERE / "audio"
EXAMPLE_GENRES = ("rock", "funk", "latin")
SKELETON_COLOR, DECORATION_COLOR = PALETTE[1], PALETTE[0]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def hbar(ax: plt.Axes, labels: list[str], values: np.ndarray, colors, fmt: str) -> None:
    """Horizontal bars with the value at each tip; vertical gridlines only."""
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.7, color=colors)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt.format(v), va="center", color=INK, fontsize=10)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def marker(ax: plt.Axes, x: float, text: str) -> None:
    """A vertical reference line with a short label at the top."""
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)  # headroom so the label clears the bars
    ax.axvline(x, color=INK, lw=1.5)
    ax.text(x, ax.get_ylim()[1], " " + text, va="top", ha="left", color=INK, fontsize=10)


def example_index(info, genre: str) -> int:
    """First training recording of a genre with at least 10 bars (fixed choice, not cherry-picked)."""
    ok = (info.split == "train") & (info["style"].str.split("/").str[0] == genre) & (info.n_bars >= 10)
    return int(info.index[ok][0])


def example_bars(info, bars, genre: str) -> np.ndarray:
    """8 bars of the genre example, starting at bar 2 to skip count-ins and pickups. Shape (8, 16, 9)."""
    return bars[example_index(info, genre)][1:1 + N_BARS]


def figures(info, bars) -> None:
    train = np.concatenate([bars[i] for i in info.index[info.split == "train"]])

    # 1. styles
    genre_bars = info.assign(genre=info["style"].str.split("/").str[0]).groupby("genre").n_bars.sum()
    genre_bars = genre_bars.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    hbar(ax, list(genre_bars.index), genre_bars.values, DECORATION_COLOR, "{:,.0f}")
    ax.set_xlabel("bars of drumming (all splits)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    decorate(fig, "Rock Dominates; Dance Music Is Rare",
             subtitle="Bars of 4/4 drumming per style in the Groove MIDI Dataset")
    save(fig, "styles")

    # 2. tempo
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(info.bpm, bins=np.arange(50, 300, 10), color=DECORATION_COLOR, edgecolor="white")
    marker(ax, BPM, f"our loops: {BPM} BPM")
    ax.set_xlabel("beats per minute")
    ax.set_ylabel("recordings")
    decorate(fig, "Most Grooves Are Played Between 80 and 140 BPM",
             subtitle=f"Tempo of each recording; our {BPM} BPM class tempo sits inside the usual range")
    save(fig, "tempo")

    # 3. length
    cap = 160  # a handful of very long recordings would squash the chart; they go in the last column
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(np.minimum(info.n_bars, cap), bins=np.arange(0, cap + 5, 4), color=DECORATION_COLOR, edgecolor="white")
    ax.set_xticks(range(0, cap + 1, 20), [*map(str, range(0, cap, 20)), f"{cap}+"])
    marker(ax, N_BARS, f"{N_BARS} bars = one training window")
    ax.set_xlabel("complete bars")
    ax.set_ylabel("recordings")
    short = int((info.n_bars < N_BARS).sum())
    decorate(fig, f"{short} Recordings Are Too Short for an 8-Bar Window",
             subtitle="Length of each recording in complete bars",
             footnote_text=f"Recordings longer than {cap} bars are counted in the last column.")
    save(fig, "length")

    # 4. hits per bar per drum
    per_bar = train.sum(1).mean(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    hbar(ax, LABELS, per_bar, [SKELETON_COLOR] * 2 + [DECORATION_COLOR] * 7, "{:.1f}")
    ax.set_xlabel("average hits per bar")
    ax.fill_between([], [], color=SKELETON_COLOR, label="skeleton drums (kick, snare)")  # legend entries only
    ax.fill_between([], [], color=DECORATION_COLOR, label="decoration drums")
    legend_below(ax, ncol=2)
    decorate(fig, "Kick, Snare and Closed Hat Do Most of the Work",
             subtitle=f"Average hits per bar for each drum, training set ({len(train):,} bars)")
    save(fig, "drums_per_bar")

    # 5. where in the bar
    prob = train.mean(0).T  # (9, 16)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(prob, aspect="auto", origin="lower", cmap=seq_cmap(), vmin=0, vmax=1)
    ax.grid(False)
    ax.set_yticks(range(len(LABELS)), LABELS)
    counts = ["1", "e", "&", "a", "2", "e", "&", "a", "3", "e", "&", "a", "4", "e", "&", "a"]
    ax.set_xticks(range(STEPS_PER_BAR), counts)
    ax.set_xticks(np.arange(STEPS_PER_BAR + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(LABELS) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("chance of a hit")
    cb.outline.set_visible(False)
    ax.set_xlabel("position in the bar (counted as 1 e & a 2 e & a ...)")
    decorate(fig, "Kick Lands on 1 and 3, Snare on 2 and 4",
             subtitle="Chance that each drum is hit at each 16th-note position, training set")
    save(fig, "where_in_bar")

    # 6. decoration density
    dens = train[:, :, 2:].sum((1, 2))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(dens, bins=np.arange(0, dens.max() + 2) - 0.5, color=DECORATION_COLOR, edgecolor="white")
    marker(ax, dens.mean(), f"average: {dens.mean():.1f}")
    ax.set_xlabel("decoration hits in one bar (hats, toms, cymbals)")
    ax.set_ylabel("bars")
    decorate(fig, f"A Typical Bar Has About {dens.mean():.0f} Decoration Hits",
             subtitle="Decoration hits per bar (everything except kick and snare), training set")
    save(fig, "density")

    # 7. quantization close-up (bars 2-3 of the funk example)
    i = example_index(info, "funk")
    row = info.loc[i]
    pm = pretty_midi.PrettyMIDI(str(GMD_DIR / row.midi_filename))
    start = STEPS_PER_BAR * step_seconds(row.bpm)  # skip the pickup bar, as in the audio
    half = step_seconds(row.bpm) / 2  # a note within half a step of the window snaps into it
    end = start + 2 * STEPS_PER_BAR * step_seconds(row.bpm)
    notes = [n for n in pm.instruments[0].notes if start - half <= n.start < end - half]
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_quantization(np.array([n.start - start for n in notes]),
                      np.array([PITCH_TO_CHANNEL[n.pitch] for n in notes]),
                      bars[i][1:3].reshape(-1, 9), step_seconds(row.bpm), ax=ax)
    legend_below(ax, ncol=2)
    decorate(fig, "Snapping to the Grid Keeps the Groove",
             subtitle=f"Two bars of funk ({row.bpm} BPM): exact hit times vs. the 16th-note cells they snap to")
    save(fig, "quantization")

    # 8. skeleton and example grids
    fig, ax = plt.subplots(figsize=(11, 4.2))
    plot_grid(skeleton_loop(2), ax=ax)
    decorate(fig, "The Skeleton: Kick on Every Beat, Snare on 2 and 4",
             subtitle="Two bars of the fixed beat every loop is built on")
    save(fig, "skeleton")
    subtitles = {"rock": "Snare on beat 3 instead of 2 and 4: real drummers leave our skeleton",
                 "funk": "Hi-hat on every 16th note, kick dodging around the beat",
                 "latin": "A repeating pattern with a short snare run in the last bar"}
    for genre in EXAMPLE_GENRES:
        row = info.loc[example_index(info, genre)]
        fig, ax = plt.subplots(figsize=(14, 4.2))
        plot_grid(example_bars(info, bars, genre).reshape(-1, 9), ax=ax)
        decorate(fig, f"{genre.capitalize()} Groove, 8 Bars at {row.bpm} BPM", subtitle=subtitles[genre])
        save(fig, f"example_{genre}")


def audio(info, bars) -> None:
    sf.write(AUDIO / "skeleton.wav", render(skeleton_loop(N_BARS)), SR)
    for genre in EXAMPLE_GENRES:
        bpm = info.loc[example_index(info, genre)].bpm
        sf.write(AUDIO / f"example_{genre}.wav", render(example_bars(info, bars, genre).reshape(-1, 9), bpm), SR)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    AUDIO.mkdir(exist_ok=True)
    apply_style()
    info, bars = load_bars()
    figures(info, bars)
    audio(info, bars)


if __name__ == "__main__":
    main()

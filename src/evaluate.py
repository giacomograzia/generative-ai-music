"""Step 4: generate B, C, D for every trained model, compute the results table and the follow-up analyses,
render the listening and temperature clips, and draw the figures.

Run from the project root: .venv/bin/python -m src.evaluate
Outputs go to output/results (tables, figures, loops.npz), output/audio and output/midi (clips).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.audio import save as save_clip
from src.config import (LISTEN_SEEDS, N_BARS, N_LOOPS, OUTPUT_DIR, REFERENCE_SEED, SEEDS, STEPS_PER_BAR,
                        TEMPERATURES_QUALITATIVE, TRAIN_SEEDS)
from src.data import load_bars, make_windows, reference_loops
from src.generate import add_fill, lstm_loop, rule_loop
from src.metrics import change_per_boundary, copying, evaluate, pulse_errors
from src.plot_style import apply_style
from src.plots import plot_constraint_cost, plot_phrase_position, plot_pulse_errors, plot_results
from src.train import load_model

RESULTS = OUTPUT_DIR / "results"
LSTM_SYSTEMS = {"B free": ("base", False), "C constrained": ("base", True),
                "D fine-tuned": ("fotf", True), "D free": ("fotf", False)}
TABLE_ROWS = ["GMD reference", "A rules", "B free", "C constrained", "D fine-tuned"]
METRICS = ["skeleton accuracy", "exact-bar rate", "diversity", "density", "bar-to-bar change", "repeat rate"]

# Figure titles state the finding; written after looking at the numbers (see the printed summary).
TITLES = {
    "results": "C Keeps the Pulse and Varies More Than the Rules, but Repeats Less Than Drummers",
    "phrase": "Drummers Change Most Going Into Bar 8; No Generated System Does",
    "constraint": "The Skeleton Pushes Against the Model Mostly on Beats 2 and 4",
    "pulse": "The Free Model Leaves the Skeleton the Way Real Drummers Do",
}


def generate_all() -> tuple[dict, dict]:
    """Loops for every system; LSTM systems per training seed. Also the constrained runs' allowed mass.

    Returns (loops, allowed_mass): loops[system] is (n_loops, steps, 9) for A and the reference, and
    {train_seed: array} for B-D; allowed_mass[system][train_seed] is (n_loops, steps).
    """
    info, bars = load_bars()
    loops = {"GMD reference": reference_loops([bars[i] for i in info.index[info.split == "test"]], N_LOOPS,
                                              REFERENCE_SEED),
             "A rules": np.stack([rule_loop(s) for s in SEEDS])}
    allowed_mass = {}
    for name, (kind, constrained) in LSTM_SYSTEMS.items():
        loops[name], allowed_mass[name] = {}, {}
        for ts in TRAIN_SEEDS:
            model = load_model(kind, ts)
            gens = [lstm_loop(model, s, constrained) for s in SEEDS]
            loops[name][ts] = np.stack([g.loop for g in gens])
            allowed_mass[name][ts] = np.stack([(g.probs * g.allowed).sum(-1) for g in gens])
        print(f"generated {name}", flush=True)
    return loops, allowed_mass


def per_seed(loops: dict, fn) -> dict:
    """Apply fn to every system; LSTM systems give {train_seed: result}."""
    return {name: ({ts: fn(x) for ts, x in v.items()} if isinstance(v, dict) else fn(v)) for name, v in loops.items()}


def results_table(loops: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long table (system, train seed, metrics) and summary (system x (metric, mean/min/max))."""
    rows = []
    for name, res in per_seed(loops, evaluate).items():
        for ts, m in (res.items() if name in LSTM_SYSTEMS else [(None, res)]):
            rows.append({"system": name, "train seed": ts, **m})
    long = pd.DataFrame(rows)
    summary = long.groupby("system", sort=False)[METRICS].agg(["mean", "min", "max"])
    return long, summary


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    loops, allowed_mass = generate_all()
    np.savez_compressed(RESULTS / "loops.npz", **{
        (f"{name}|{ts}" if ts is not None else name): x
        for name, v in loops.items() for ts, x in (v.items() if isinstance(v, dict) else [(None, v)])})

    # main table
    long, summary = results_table(loops)
    long.to_csv(RESULTS / "results_per_model.csv", index=False)
    summary.to_csv(RESULTS / "results_summary.csv")
    print(summary.round(3).to_string())

    # follow-up 1: copying
    info, bars = load_bars()
    train = [bars[i] for i in info.index[info.split == "train"]]
    train_bars, train_windows = np.concatenate(train), make_windows(train, N_BARS, stride=1)
    copy_rows = []
    for name, res in per_seed(loops, lambda x: copying(x, train_bars, train_windows)).items():
        for ts, m in (res.items() if name in LSTM_SYSTEMS else [(None, res)]):
            copy_rows.append({"system": name, "train seed": ts, **m})
    copy = pd.DataFrame(copy_rows).groupby("system", sort=False).mean(numeric_only=True).drop(columns="train seed")
    copy.to_csv(RESULTS / "copying.csv")
    print(copy.round(3).to_string())

    # follow-up 2: phrase position
    phrase = {name: (np.mean(list(v.values()), axis=0) if isinstance(v, dict) else v)
              for name, v in per_seed(loops, change_per_boundary).items()}
    pd.DataFrame(phrase, index=[f"{i}-{i + 1}" for i in range(1, N_BARS)]).to_csv(RESULTS / "phrase_position.csv")
    print(pd.DataFrame(phrase).round(3).to_string())

    # follow-up 3: cost of the constraint (per position in the bar), and what forcing changes (C vs B)
    cost = {name: np.mean([m.reshape(N_LOOPS, N_BARS, STEPS_PER_BAR).mean(axis=(0, 1)) for m in v.values()], axis=0)
            for name, v in allowed_mass.items() if LSTM_SYSTEMS[name][1]}
    pd.DataFrame(cost, index=range(1, STEPS_PER_BAR + 1)).to_csv(RESULTS / "constraint_cost.csv")
    print(pd.DataFrame(cost).round(3).T.to_string())
    print("overall chance of the forced kick/snare:", {k: round(float(v.mean()), 3) for k, v in cost.items()})

    # follow-up 4: how the free models leave the skeleton
    errors = {name: (pd.DataFrame(list(v.values())).mean() if name in LSTM_SYSTEMS else pd.Series(v))
              for name, v in per_seed({k: loops[k] for k in ("GMD reference", "B free", "D free")}, pulse_errors).items()}
    errors = pd.DataFrame(errors)
    errors.to_csv(RESULTS / "pulse_errors.csv")
    print(errors.round(3).to_string())

    # figures
    apply_style()
    plot_results(summary.loc[TABLE_ROWS], METRICS, TITLES["results"]).savefig(RESULTS / "results.png", bbox_inches="tight")
    plot_phrase_position({k: phrase[k] for k in TABLE_ROWS}, TITLES["phrase"]).savefig(
        RESULTS / "phrase_position.png", bbox_inches="tight")
    plot_constraint_cost(cost, TITLES["constraint"]).savefig(RESULTS / "constraint_cost.png", bbox_inches="tight")
    plot_pulse_errors(errors, TITLES["pulse"]).savefig(RESULTS / "pulse_errors.png", bbox_inches="tight")
    plt.close("all")

    # listening clips (training seed 0, listening seeds fixed in advance, same fill for all systems)
    ts = TRAIN_SEEDS[0]
    for name, letter in (("B free", "B"), ("C constrained", "C"), ("D fine-tuned", "D")):
        for s in LISTEN_SEEDS:
            save_clip(add_fill(loops[name][ts][s]), f"{letter}_seed{s}", OUTPUT_DIR)
    # temperature clips (qualitative only): C, first listening seed
    model = load_model("base", ts)
    for temp in TEMPERATURES_QUALITATIVE:
        save_clip(add_fill(lstm_loop(model, LISTEN_SEEDS[0], True, temperature=temp).loop),
                  f"C_seed{LISTEN_SEEDS[0]}_T{temp}", OUTPUT_DIR)
    print("clips written", flush=True)


if __name__ == "__main__":
    main()

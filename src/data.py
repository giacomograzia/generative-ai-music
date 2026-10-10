"""GMD download, quantize to 16 steps, map pitches to 9 classes, official split, bar arrays, windows."""
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

from src.config import DATA_DIR, N_BARS, N_DRUMS, STEPS_PER_BAR

GMD_URL = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"
GMD_DIR = DATA_DIR / "groove"
CACHE = DATA_DIR / "bars.npz"  # delete to rebuild after changing the processing

# Roland TD-11 pitch -> our channel, following the reduced 9-category mapping on the GMD page
# (magenta.tensorflow.org/datasets/groove). GMD's three toms map as: tom 1 (high) -> high tom,
# tom 2 (low-mid) -> mid tom, tom 3 (high floor) -> low tom. Pedal hi-hat counts as closed hat.
PITCH_TO_CHANNEL = {
    36: 0,                                      # kick
    38: 1, 40: 1, 37: 1,                        # snare head, rim, cross-stick
    42: 2, 22: 2, 44: 2,                        # closed hat bow, edge, pedal
    46: 3, 26: 3,                               # open hat bow, edge
    43: 4, 58: 4,                               # tom 3 (floor) head, rim -> low tom
    45: 5, 47: 5,                               # tom 2 head, rim -> mid tom
    48: 6, 50: 6,                               # tom 1 head, rim -> high tom
    49: 7, 55: 7, 57: 7, 52: 7,                 # crash 1/2 bow, edge
    51: 8, 59: 8, 53: 8,                        # ride bow, edge, bell
}


def download() -> Path:
    """Download and unzip the MIDI-only GMD into data/groove (skipped if already present)."""
    if not (GMD_DIR / "info.csv").exists():
        zip_path = DATA_DIR / "groove-v1.0.0-midionly.zip"
        if not zip_path.exists():
            urllib.request.urlretrieve(GMD_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
    return GMD_DIR


def load_info() -> pd.DataFrame:
    """GMD metadata filtered to 4/4 beats (no fills), with the official split column."""
    info = pd.read_csv(download() / "info.csv")
    return info[(info.time_signature == "4-4") & (info.beat_type == "beat")].reset_index(drop=True)


def quantize(starts: np.ndarray, pitches: np.ndarray, bpm: float, end: float) -> tuple[np.ndarray, dict]:
    """Quantize note onsets to a binary 16th-note grid.

    starts: onset times in seconds, measured from the first bar line (t=0).
    pitches: MIDI pitches, same length as starts.
    end: end of the performance in seconds; only complete bars before it are kept
    (half a step of tolerance), so a trailing partial bar is dropped.
    Returns (bars, stats): bars has shape (n_bars, 16, 9);
    stats counts notes dropped as unmappable or falling after the last complete bar.
    """
    step_sec = 60.0 / bpm / 4
    steps = np.rint(np.asarray(starts) / step_sec).astype(int)
    channels = np.array([PITCH_TO_CHANNEL.get(int(p), -1) for p in pitches], dtype=int)
    n_bars = int((end + step_sec / 2) // (step_sec * STEPS_PER_BAR))
    mappable = channels >= 0
    in_range = steps < n_bars * STEPS_PER_BAR
    keep = mappable & in_range
    grid = np.zeros((n_bars * STEPS_PER_BAR, N_DRUMS), dtype=np.int8)
    grid[steps[keep], channels[keep]] = 1
    stats = {"notes": len(steps), "unmappable": int((~mappable).sum()),
             "after_last_bar": int((mappable & ~in_range).sum())}
    return grid.reshape(n_bars, STEPS_PER_BAR, N_DRUMS), stats


def trim_empty(bars: np.ndarray) -> np.ndarray:
    """Drop empty bars at the start and end (count-ins, trailing silence); keep empty bars in between.

    bars: (n_bars, 16, 9). Returns (n_kept, 16, 9).
    """
    played = np.flatnonzero(bars.any(axis=(1, 2)))
    return bars[played[0]:played[-1] + 1] if len(played) else bars[:0]


def midi_to_bars(path: Path, bpm: float) -> tuple[np.ndarray, dict]:
    """Read one GMD MIDI file, quantize it and trim empty edge bars; returns (bars (n_bars, 16, 9), stats)."""
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    bars, stats = quantize(np.array([n.start for n in notes]), np.array([n.pitch for n in notes]), bpm,
                           pm.get_end_time())
    trimmed = trim_empty(bars)
    stats["trimmed_bars"] = len(bars) - len(trimmed)
    return trimmed, stats


def load_bars() -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Quantized bars for every filtered GMD file, cached in data/bars.npz.

    Returns (info, bars) where bars[i] has shape (n_bars_i, 16, 9) and belongs to info row i.
    info gains columns n_bars, notes, unmappable, after_last_bar, trimmed_bars.
    """
    info = load_info()
    if CACHE.exists():
        cached = np.load(CACHE)
        bars = [cached[f"f{i}"] for i in range(len(info))]
        stats = pd.DataFrame(cached["stats"], columns=["notes", "unmappable", "after_last_bar", "trimmed_bars"])
    else:
        results = [midi_to_bars(GMD_DIR / row.midi_filename, row.bpm) for row in info.itertuples()]
        bars = [b for b, _ in results]
        stats = pd.DataFrame([s for _, s in results])
        np.savez_compressed(CACHE, stats=stats.to_numpy(), **{f"f{i}": b for i, b in enumerate(bars)})
    info = pd.concat([info, stats], axis=1)
    info["n_bars"] = [len(b) for b in bars]
    return info, bars


def make_windows(bars: list[np.ndarray], n_bars: int = N_BARS, stride: int = 2) -> np.ndarray:
    """Cut consecutive bars into overlapping windows; fragments shorter than n_bars are discarded.

    bars: list of (n_bars_i, 16, 9) arrays. Returns (n_windows, n_bars * 16, 9) int8.
    """
    windows = [b[s:s + n_bars].reshape(-1, N_DRUMS)
               for b in bars for s in range(0, len(b) - n_bars + 1, stride)]
    return np.stack(windows) if windows else np.zeros((0, n_bars * STEPS_PER_BAR, N_DRUMS), np.int8)

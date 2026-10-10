"""MIDI writing (pretty_midi) and rendering (fluidsynth or numpy drum synth fallback)."""
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from src.config import BPM, PITCHES

SR = 44100
VELOCITY = 100
TARGET_RMS = 0.1


def step_seconds(bpm: float = BPM) -> float:
    """Duration of one 16th-note step in seconds."""
    return 60.0 / bpm / 4


def to_midi(pattern: np.ndarray, bpm: float = BPM) -> pretty_midi.PrettyMIDI:
    """Binary pattern (steps, 9) -> drum-track PrettyMIDI with fixed velocity, one 16th per step."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    dt = step_seconds(bpm)
    for step, ch in zip(*np.nonzero(pattern)):
        drums.notes.append(pretty_midi.Note(VELOCITY, PITCHES[ch], step * dt, (step + 1) * dt))
    pm.instruments.append(drums)
    return pm


def _hit(ch: int, sr: int) -> np.ndarray:
    """Synthesize one hit for channel ch: sine-sweep kick, noise snare/hats/cymbals, pitched toms."""
    rng = np.random.default_rng(ch)  # fixed noise per drum, so renders are deterministic
    length = {0: 0.35, 1: 0.2, 2: 0.05, 3: 0.3, 7: 1.0, 8: 0.4}.get(ch, 0.3)
    t = np.arange(int(length * sr)) / sr
    noise = rng.uniform(-1, 1, len(t))
    if ch == 0:  # kick: pitch drops from 120 Hz to 45 Hz
        freq = 45 + 75 * np.exp(-t * 30)
        return np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 9)
    if ch == 1:  # snare: tone + noise
        return (0.4 * np.sin(2 * np.pi * 190 * t) + 0.7 * noise) * np.exp(-t * 20)
    if ch in (4, 5, 6):  # toms: low, mid, high
        f0 = {4: 90, 5: 130, 6: 180}[ch]
        freq = f0 * (1 + 0.3 * np.exp(-t * 20))
        return np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 10)
    # hats and cymbals: high-passed noise (first difference) with drum-specific decay
    bright = np.diff(noise, prepend=0.0)
    decay = {2: 80, 3: 10, 7: 4, 8: 12}[ch]
    gain = {2: 0.35, 3: 0.3, 7: 0.3, 8: 0.2}[ch]
    return gain * bright * np.exp(-t * decay)


def synth(pattern: np.ndarray, bpm: float = BPM, sr: int = SR) -> np.ndarray:
    """Render a pattern (steps, 9) with the numpy drum synth; returns mono float audio, loudness-normalized."""
    dt = int(round(step_seconds(bpm) * sr))
    hits = {ch: _hit(ch, sr) for ch in range(pattern.shape[1])}
    out = np.zeros(len(pattern) * dt + max(len(h) for h in hits.values()))
    for step, ch in zip(*np.nonzero(pattern)):
        out[step * dt: step * dt + len(hits[ch])] += hits[ch]
    out = out[: len(pattern) * dt + sr // 2]  # keep half a second of tail
    return normalize(out)


def normalize(audio: np.ndarray) -> np.ndarray:
    """Scale audio to a fixed RMS (equal loudness across clips), clipped to [-1, 1]."""
    rms = np.sqrt(np.mean(audio ** 2))
    return np.clip(audio * (TARGET_RMS / rms), -1, 1) if rms > 0 else audio


def render(pattern: np.ndarray, bpm: float = BPM, soundfont: Path | None = None, sr: int = SR) -> np.ndarray:
    """Render a pattern to audio: fluidsynth with a soundfont if given, otherwise the numpy synth."""
    if soundfont is not None:
        return normalize(to_midi(pattern, bpm).fluidsynth(fs=sr, sf2_path=str(soundfont)))
    return synth(pattern, bpm, sr)


def save(pattern: np.ndarray, name: str, out_dir: Path, bpm: float = BPM, soundfont: Path | None = None) -> None:
    """Write out_dir/midi/<name>.mid and out_dir/audio/<name>.wav."""
    (out_dir / "midi").mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    to_midi(pattern, bpm).write(str(out_dir / "midi" / f"{name}.mid"))
    sf.write(out_dir / "audio" / f"{name}.wav", render(pattern, bpm, soundfont), SR)

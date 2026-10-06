# Learned drum decoration for group fitness music

Generative AI for Music (University of Amsterdam). A small LSTM trained on human drumming adds
decoration to a fixed kick/snare skeleton. We compare rules (A), free LSTM (B) and constrained
LSTM (C). See [PLAN.md](PLAN.md) for the full project plan; the deliverable is `notebook.ipynb`.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Optional: a General MIDI soundfont for `pretty_midi.fluidsynth`; otherwise a numpy drum synth is used.

## Data

[Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) (MIDI-only, ~3 MB), licence
CC BY 4.0. Gillick et al. 2019, "Learning to Groove with Inverse Sequence Transformations".
Downloaded into `data/` (git-ignored) by `src/data.py`.

## Layout

`src/` reusable code · `tests/` unit tests · `checkpoints/` trained model · `output/` MIDI, WAV, results.

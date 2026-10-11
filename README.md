# Learned drum decoration for group fitness music

Generative AI for Music (University of Amsterdam). Aesthetic goal: **steady for the body, alive for
the ear**. Research question: with kick and snare fixed, does a model trained on human drumming
produce more varied drums than hand-written rules? A small LSTM trained on human drumming adds
decoration to a fixed kick/snare skeleton. We compare rules (A), free LSTM (B) and constrained
LSTM (C). See [PLAN.md](PLAN.md) for the full project plan; the deliverable is `notebook.ipynb`.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Run the notebook top to bottom (a few seconds; loads results from `output/`, no GPU needed), or open it in
Jupyter or VS Code:

```
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Optional: a General MIDI soundfont for `pretty_midi.fluidsynth`; otherwise a numpy drum synth is used.

## Data

[Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) (MIDI-only, ~3 MB), licence
CC BY 4.0. Gillick et al. 2019, "Learning to Groove with Inverse Sequence Transformations".
Downloaded into `data/` (git-ignored) by `src/data.py`.

## Web pages

Open directly in a browser (no server, no internet needed):

- `web/listening.html`: blind listening test for three listeners (ratings 1-5 and notes per clip), autosaves in the
  browser and resumes. In Chrome or Edge, **Save into project** writes straight to `output/listening/ratings.json`;
  elsewhere, **Save to file** and move the download there. **Reveal** shows systems and results.
- `web/generation.html`: step through or play the real model writing a loop; switch system (B, C, D), seed and
  temperature; optional sound.

Regenerate their data with `.venv/bin/python -m src.export_web` after re-running training or evaluation.

## Layout

`src/` reusable code · `tests/` unit tests · `checkpoints/` trained models (not committed; `.venv/bin/python -m src.train` recreates them in about 16 min on a GPU) · `output/` MIDI, WAV, results.

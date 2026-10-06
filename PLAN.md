# Project Plan: Learned drum decoration for group fitness music

Course: Generative AI for Music (University of Amsterdam).
Deliverable: one annotated Jupyter notebook (code + generated music + explanation and reflection), with reusable code in a `src/` folder that the notebook imports.

## Progress

- [x] Repo structure initialized (config, stubs, README, requirements, notebook skeleton)
- [ ] Step 1: Setup and data
- [ ] Step 2: Rule baseline and metrics
- [ ] Step 3: Train the LSTM
- [ ] Step 4: Generate B and C, evaluate
- [ ] Step 5: Notebook

## 1. The idea

**Aesthetic goal:** music a fitness instructor can trust: a fixed beat the body can lock onto, with learned variation that keeps it musically interesting.

**Research question:** can a learned model add interesting drum decoration while a fixed rhythmic skeleton preserves the reliable pulse needed for group fitness?

**Core contribution (one sentence for the report):** we train a generative drum model on human drumming, show that unconstrained generation cannot guarantee the stable landmarks group fitness needs, and fix this by letting the model generate freely while deterministic kick and snare constraints are enforced during generation.

**Why one model, two generation modes:** the same trained model, data and seeds are used with and without constraints, so the only thing that changes is the constraint: same AI, same data, same seed, constraint off vs on. This isolates one variable. (A specialized decoration-only model would likely be more efficient but would remove this clean comparison; see the reflection.)

**Skeleton** (per 4/4 bar, 16 steps of 16th notes, steps numbered 0-15):
- kick on steps 0, 4, 8, 12 (four-on-the-floor)
- snare on steps 4 and 12 (beats 2 and 4)

## 2. Three systems

| System | What it does | Question it answers |
|---|---|---|
| **A. Rules** | Skeleton + hand-written probabilistic hats/toms/cymbals | What can we do without learning? (A vs C: does learning add anything beyond probabilistic rules?) |
| **B. LSTM free** | Small LSTM trained on GMD generates every drum | What happens if learning controls everything? |
| **C. LSTM constrained** | Same LSTM, but kick and snare are forced to the skeleton and fed back into the model; it samples only the other drums | Our contribution (B vs C: what do deterministic constraints add to the same learned model?) |

**Fixed settings for the experiment:** 8-bar loops, 128 BPM, 30 loops per system (seeds 0-29), sampling temperature 1.0. Temperature is **not** part of the formal experiment (section 6).

**Fills are not part of the experiment.** The quantitative comparison uses 8 plain bars. Fills exist only for listening clips (section 7).

## 3. Data

**Groove MIDI Dataset (GMD)**, MIDI-only version (about 3 MB), from `magenta.tensorflow.org/datasets/groove` (also loadable via TensorFlow Datasets as `groove/full-midionly`). Licence CC BY 4.0; cite Gillick et al. 2019, "Learning to Groove with Inverse Sequence Transformations". It has 1,150 MIDI files of human drumming on an electronic kit (about 13.6 hours, 22k+ measures).

`info.csv` columns: drummer, session, id, style, bpm, beat_type (`beat` or `fill`), time_signature, midi_filename, audio_filename, duration, split (`train`, `validation`, `test`).

**Processing:**
1. Use the **official `split` column**; do not create a different split. Train on train, early-stop on validation, report test loss once at the end.
2. Keep only `time_signature == 4-4` and `beat_type == beat` (fills are handled by rule, and only for listening).
3. Quantize every note to the nearest 16th-note step using the file's tempo (`bpm` in `info.csv`, check against MIDI tempo events). Verify on a few files that performances start on a bar line and that bars line up. GMD contains human microtiming and velocity; our binary 16-step representation discards both, and the notebook should state this. Log how many files and notes were dropped.
4. Map GMD's pitches to the 9 drum classes below. The Roland TD-11 used for GMD does not follow General MIDI exactly; use the mapping table on the dataset page (it lists a reduced category per pitch) and verify it with a small test. Drop anything unmappable and log the count.
5. Produce an array `(n_bars, 16, 9)` per file, keeping bars in order so consecutive bars can form training windows.

**The 9 drum channels** (index: class, MIDI pitch used when writing files):
0 kick 36, 1 snare 38, 2 closed hat 42, 3 open hat 46, 4 low tom 45, 5 mid tom 48, 6 high tom 50, 7 crash 49, 8 ride 51.
Channels 0-1 are the skeleton; channels 2-8 are the "decoration".

**Exploration for the notebook (keep it short):** number of files and bars after filtering; how often each drum is used; one example groove drawn as a grid; one example before and after quantization; **d_GMD = mean decoration hits per bar (channels 2-8) over the training bars**, used to tune the rule baseline.

## 4. Model

One small LSTM, trained on CPU in minutes.

- Input per step: previous step's 9-dim 0/1 vector + position in bar (one-hot, 16).
- 1-2 LSTM layers, hidden size about 128.
- Output: 9 logits, one independent Bernoulli per drum (binary cross-entropy loss).
- **Training windows: 8 consecutive bars (128 steps), the same length as the generated loops**, so generation does not run past the sequence length the model was trained on. Use stride of 1-4 bars. Discard fragments shorter than the window. Check the number of windows first; if too few remain, fall back to 4-bar windows and say so in the notebook. Do not build padding/masking unless it turns out to be necessary.
- Teacher forcing, Adam, early stopping on validation loss, fixed seed. Save checkpoint and a loss curve.

**Generation** (start from an all-zero input at step 0 in both modes; per step: `p = sigmoid(logits / T)`, sample each channel). **The LSTM hidden state is carried continuously across all 8 bars; it is never reset at bar boundaries.**
- **Free (B):** sample all 9 channels.
- **Constrained (C):** at each step, set kick and snare to the skeleton values (instead of sampling them), feed these forced values back as the next input, and sample only channels 2-8.

## 5. Metrics (three, each aligned with the research question; all computed on all 8 bars)

| Metric | Channels | Question it asks |
|---|---|---|
| **1. Skeleton accuracy** | 0-1 | Can a fitness instructor trust the pulse? |
| **2. Decoration diversity** | 2-8 | Does the system generate varied rhythms? |
| **3. Decoration density** | 2-8 | Are differences in diversity simply because one system plays more notes? |

**1. Skeleton accuracy** penalizes both missing and extra kick/snare hits: `1 - (incorrect kick/snare cells) / (total kick/snare cells)`, comparing every cell of channels 0-1 with the desired skeleton. Required-hit recall alone would reward a free model that plays kick everywhere. Because most cells are "off" in the skeleton (6 of 32 kick/snare cells per bar are on), an empty kick/snare track still scores about 81%, so also report the stricter **exact-bar rate**: the percentage of bars whose kick and snare channels match the skeleton exactly. A and C score 100% on both by construction (this is a sanity check; the informative number is B).

**2. Decoration diversity:** mean pairwise normalized Hamming distance between the 30 loops, computed on the decoration channels only, so the fixed kick/snare in A and C do not reduce their score.

**3. Decoration density:** average decoration hits per bar (channels 2-8). Using decoration only means that missing kicks in B are not misread as "less decorative".

**Rule baseline A:** compute d_GMD once (section 3), set A's probabilities once so its decoration density is approximately d_GMD (for example: GMD training mean 8.7 hits/bar, rule baseline 8.4), then freeze A. No optimization.

Report one table (rows A, B, C; columns: skeleton accuracy, exact-bar rate, diversity, decoration density).

**Expectations to state before looking at results** (they may be wrong; report honestly):
- B breaks the pulse a noticeable share of the time.
- C keeps the pulse; whether its decoration is more varied or more convincing than A's is decided by the table and the listening.

## 6. Temperature (qualitative only)

Not a formal experiment. For one seed, render three clips from C (and optionally B) at T = 0.8, 1.0, 1.2, and discuss in the notebook whether higher temperature gives more variation or just chaos.

## 7. Listening

- Render clips (15 s each = 8 bars at 128 BPM) for A, B, C: 2 per system, using seeds chosen in advance (not cherry-picked), same drum kit/renderer, loudness-normalized.
- **Fill (listening clips only, same for all systems):** after generating the 8 bars, overwrite the tom channels (4-6) on steps 12-15 of bar 8 with a fixed pattern: step 12 high tom, 13 high tom, 14 mid tom, 15 low tom. Hats, cymbals, kick and snare are untouched.
- Everyone on the team listens and writes down: is the beat easy to follow, is it interesting, would it work in a fitness class. The instructor's own judgement is valuable and should be written up as such.
- If time allows, ask 5 other people (ideally people who do group fitness) to rate each clip 1-5 on: "How easy is the beat to follow?", "How musically interesting is it?", "Would this work for a fitness class?". Clips unlabeled and in random order. Report means with a clear small-sample caveat; no significance claims; collect no personal data.

## 8. Audio rendering

Write patterns as MIDI with `pretty_midi` (drum track, the pitches above, fixed velocity). Render to audio with `pretty_midi.fluidsynth` and a soundfont if available; otherwise a simple numpy drum synthesizer (sine-sweep kick, noise-burst snare/hats) so the notebook never depends on a soundfont. Play clips in the notebook with `IPython.display.Audio` and save MIDI and WAV into `output/` so results can be heard without re-running anything.

## 9. Repo layout

```
project/
├── notebook.ipynb         # the deliverable: explanation, audio, results, reflection
├── README.md              # how to run; data source and licence
├── requirements.txt       # numpy, pandas, pretty_midi, matplotlib, soundfile, scipy, torch, pytest
├── src/
│   ├── config.py          # constants: classes, pitches, skeleton, BPM, seeds, paths
│   ├── data.py            # GMD download, quantize, map pitches, split, bar arrays, windows
│   ├── model.py           # LSTM
│   ├── train.py           # training script
│   ├── generate.py        # rules (A), free (B), constrained (C), listening fill
│   ├── metrics.py         # skeleton accuracy, exact-bar rate, diversity, density
│   ├── audio.py           # MIDI writing + rendering
│   └── plots.py           # grids, loss curve, results figure
├── tests/test_core.py     # small checks (section 13)
├── data/                  # GMD (git-ignored)
├── checkpoints/           # trained model (small, committed)
└── output/                # midi/, audio/, results/
```

## 10. Work order

1. **Setup and data.** Repo, requirements, GMD download, official split, quantization and pitch mapping, d_GMD, example plots. Check audio rendering works (play a skeleton loop).
2. **Rule baseline and metrics.** Skeleton, rule decoration (A) tuned once to d_GMD, the metrics, audio. Done when A's loops play and the metrics behave correctly on hand-built patterns.
3. **Train the LSTM.** Check window counts, train, plot the loss curve, report validation loss (test loss once at the end).
4. **Generate B and C, evaluate.** 30 loops each, compute the table, render clips with the listening fill, listen. Optional: temperature clips, listener ratings.
5. **Notebook.** Assemble the sections below, run top to bottom from a clean state.

Minimum viable: steps 1-3 and 4 without the optional parts, then 5.

## 11. Notebook outline (7 sections)

1. **The idea:** group fitness, why predictable pulse plus variation matters, the research question. Briefly mention resources reviewed and why they are not used in the core (Drums RNN: old Magenta/TensorFlow dependencies, extra environment; Deep Drum: JavaScript demo built on Drums RNN; Metacreation: research projects, dataset ideas; AudioCraft/MusicGen: audio-domain, heavy compute, little control over the beat).
2. **The data:** GMD, quantization to the 16-step × 9-drum representation, what it discards.
3. **Rule baseline (A):** how it works, play a loop.
4. **Learned model:** simple diagram, training curve, play free (B) output.
5. **Adding constraints (C):** the model sometimes breaks the pulse; instead of asking the model to control decisions that must be reliable, we constrain them. Play the result. This is the conceptual centre of the notebook.
6. **Experiment:** A vs B vs C table, audio examples (with the common fill), temperature clips, listening notes.
7. **Reflection:** did learning add musical variation (A vs C)? did constraints make learned generation suitable for fitness (B vs C)? Design choice: the model spends capacity learning kick and snare that we then fix; a cleaner design would predict only the decoration, conditioned on the skeleton. We chose the simpler architecture deliberately, because it lets free and constrained generation come from exactly the same model and isolates the effect of constraints. Other limits: binary 16th-note grid (no velocity, no microtiming), GMD is general human drumming and not dance or fitness music, fills and phrase structure are rule-based, small listening sample. Then future work (below).

## 12. Future work (put in the notebook's last section; do not build now)

- Condition generation on a density or intensity level; learn fills using the dataset's `beat_type` label.
- A decoration-only model conditioned on the skeleton, compared with the current approach.
- A formal temperature study; velocity and microtiming.
- Compare with a pretrained model, Drums RNN. If added: run it in a separate environment, prime it with a four-on-the-floor bar, and note that its `--num_steps` includes the primer (confirm with `--help`; e.g. a 2-bar primer plus 8 generated bars needs 160 steps), then compare with the same three metrics on the same bars.
- A full class arc (warm-up, build, peak, cooldown) with changing BPM and temperature.
- Richer evaluation: novelty against training bars, pulse-clarity measures, leave-one-drummer-out, larger listening test.

## 13. Tests (small, in `tests/test_core.py`)

- Pitch mapping: a few known GMD pitches map to the expected channels.
- Quantization: synthetic notes at known times land on the expected steps.
- Constrained generation: for every seed, kick/snare channels exactly equal the skeleton.
- Same seed gives identical output.
- Skeleton accuracy: a perfect skeleton loop gives 100%; a loop with extra kicks scores lower even though all required hits are present; exact-bar rate drops accordingly.
- Diversity: two identical loops give 0; density counts channels 2-8 only.
- Listening fill: changes only channels 4-6 on steps 12-15 of bar 8.

## 14. Risks

| Risk | Mitigation |
|---|---|
| Pitch mapping or quantization mistakes | Follow the dataset page's mapping table; unit tests; inspect examples |
| Few 8-bar windows after filtering | Check counts first; fall back to 4-bar windows and say so |
| Free model produces near-empty or messy output | Report it honestly (that is a result); check training length and T first |
| Model overfits (small dataset) | Small model, early stopping on the official validation split |
| No soundfont available | numpy synth fallback; committed WAV files |
| Scope creep | Section 12 is explicitly out of scope |

## 15. Working agreements

- Work in the order of section 10; after each step run the tests and a short demo and summarize results.
- Keep functions small and typed; docstrings state shapes and units.
- Fix seeds; do not add dependencies beyond `requirements.txt` without asking.
- Do not tune the comparison to make C look good; report negative results.
- Keep code in `src/`; the notebook explains and shows results.

## 16. Definition of done

- `notebook.ipynb` runs top to bottom in a clean environment, is annotated, has audible examples for A, B and C, the results table, and the reflection.
- `pytest` passes.
- `output/` contains the generated MIDI and WAV files; `checkpoints/` contains the trained model; README explains setup, data source and licence.

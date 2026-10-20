"""Rule baseline (A), free LSTM (B), constrained LSTM (C, D), and the listening-only fill."""
from dataclasses import dataclass

import numpy as np
import torch

from src.config import FILL, N_BARS, N_DRUMS, SKELETON_CHANNELS, STEPS_PER_BAR, TEMPERATURE, skeleton_loop
from src.model import N_COMBOS, DrumLSTM, from_combo, step_input

# --- A: rules -------------------------------------------------------------------------------
# Hand-written chance of a hit per drum and position in the bar (16 steps), applied independently
# to every cell on top of the skeleton. Set once so the expected decoration density is close to
# d_GMD (about 10 hits per bar, data_analysis/README.md), then frozen. Not fitted to anything else.
EIGHTHS = np.arange(0, STEPS_PER_BAR, 2)          # 1 & 2 & 3 & 4 &
SIXTEENTH_OFFBEATS = np.arange(1, STEPS_PER_BAR, 2)  # e and a
BEATS = np.arange(0, STEPS_PER_BAR, 4)            # 1 2 3 4
BAR_END = np.arange(13, STEPS_PER_BAR)            # last three 16ths, where tom pickups go


def rule_probabilities() -> np.ndarray:
    """Chance of a hit per (step, drum) for one bar, shape (16, 9); kick/snare (0-1) are 0 here."""
    p = np.zeros((STEPS_PER_BAR, N_DRUMS))
    p[EIGHTHS, 2] = 0.9               # closed hat: 8th notes, the classic hat pattern
    p[SIXTEENTH_OFFBEATS, 2] = 0.1    # closed hat: occasional 16th in between
    p[14, 3] = 0.3                    # open hat: on the "&" of 4, leading into the next bar
    p[BAR_END, 4:7] = 0.05            # low/mid/high tom: rare pickups at the end of the bar
    p[0, 7] = 0.25                    # crash: sometimes on the first beat of a bar
    p[BEATS, 8] = 0.3                 # ride: sometimes on the beat
    return p


def rule_loop(seed: int, n_bars: int = N_BARS) -> np.ndarray:
    """System A: skeleton plus independently sampled decoration. Returns (n_bars * 16, 9) int8."""
    rng = np.random.default_rng(seed)
    p = np.tile(rule_probabilities(), (n_bars, 1))
    loop = (rng.random(p.shape) < p).astype(np.int8)
    loop[loop[:, 3] == 1, 2] = 0      # a hi-hat is either open or closed, not both
    loop[:, :2] = skeleton_loop(n_bars)[:, :2]
    return loop


# --- B, C, D: the LSTM --------------------------------------------------------------------
@dataclass
class Generation:
    """One generated loop plus what the model said at every step (used by the animation).

    loop: (steps, 9) int8. probs: (steps, K) the model's probabilities before the constraint
    (K = 512 combinations or 9 drums). allowed: (steps, K) bool, what the constraint permits.
    hidden: (steps, layers, units) the LSTM output h of every layer at each step, if recorded.
    cell: (steps, layers, units) the LSTM cell state c (its internal memory) at each step, if recorded.
    """
    loop: np.ndarray
    probs: np.ndarray
    allowed: np.ndarray
    hidden: np.ndarray | None = None
    cell: np.ndarray | None = None


@torch.no_grad()
def lstm_loop(model: DrumLSTM, seed: int, constrained: bool, n_bars: int = N_BARS,
              temperature: float = TEMPERATURE, record_hidden: bool = False) -> Generation:
    """Generate one loop step by step, feeding each step back as the next input.

    Free (B): the model chooses every drum. Constrained (C, D): kick and snare follow the skeleton;
    the model chooses only among options that match it (categorical: the combinations with the
    skeleton's kick and snare, renormalised; bernoulli: kick and snare are overwritten).
    The hidden state runs across all bars; it is never reset. Sampling uses numpy with this seed.
    """
    rng = np.random.default_rng(seed)
    skel = skeleton_loop(n_bars)
    dev = next(model.parameters()).device
    prev, state = torch.zeros(1, N_DRUMS, device=dev), None
    loop, all_probs, all_allowed, hidden, cell = [], [], [], [], []
    combos = from_combo(torch.arange(N_COMBOS)).numpy()  # (512, 9)
    for t in range(n_bars * STEPS_PER_BAR):
        logits, state = model(step_input(prev, t), state)
        if record_hidden:
            hidden.append(state[0][:, 0].cpu().numpy())  # (layers, units)
            cell.append(state[1][:, 0].cpu().numpy())
        logits = logits[0, 0].double() / temperature
        if model.output == "categorical":
            probs = torch.softmax(logits, -1).cpu().numpy()
            allowed = np.ones(N_COMBOS, bool)
            if constrained:
                allowed = (combos[:, list(SKELETON_CHANNELS)] == skel[t, list(SKELETON_CHANNELS)]).all(1)
            p = np.where(allowed, probs, 0)
            step = combos[rng.choice(N_COMBOS, p=p / p.sum())]
        else:
            probs = torch.sigmoid(logits).cpu().numpy()
            allowed = np.ones(N_DRUMS, bool)
            step = (rng.random(N_DRUMS) < probs).astype(float)
            if constrained:
                allowed[list(SKELETON_CHANNELS)] = False
                step[list(SKELETON_CHANNELS)] = skel[t, list(SKELETON_CHANNELS)]
        loop.append(step)
        all_probs.append(probs)
        all_allowed.append(allowed)
        prev = torch.tensor(step, dtype=torch.float32, device=dev)[None]
    return Generation(np.array(loop, dtype=np.int8), np.array(all_probs), np.array(all_allowed),
                      np.array(hidden) if record_hidden else None, np.array(cell) if record_hidden else None)


# --- listening fill -------------------------------------------------------------------------
def add_fill(loop: np.ndarray) -> np.ndarray:
    """Overwrite the toms (channels 4-6) on steps 12-15 of the last bar with the fixed fill.

    Listening clips only; same for all systems. Returns a copy, (steps, 9).
    """
    out = loop.copy()
    start = len(loop) - STEPS_PER_BAR
    out[start + 12:, 4:7] = 0
    for step, ch in FILL.items():
        out[start + step, ch] = 1
    return out

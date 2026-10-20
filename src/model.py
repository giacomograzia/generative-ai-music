"""Small LSTM: input = previous 9-dim step + one-hot position (16).

Two output types:
- "bernoulli": 9 logits, one independent hit probability per drum.
- "categorical": 512 logits, one choice among all drum combinations per step (as Magenta's Drums RNN).
Both are trained with the negative log-likelihood of the real next step, so their losses per step compare.
"""
import torch
from torch import nn

from src.config import N_DRUMS, STEPS_PER_BAR

INPUT_SIZE = N_DRUMS + STEPS_PER_BAR
N_COMBOS = 2 ** N_DRUMS
BITS = 2 ** torch.arange(N_DRUMS)  # drum k <-> bit k of the combination index


def device() -> torch.device:
    """The GPU if available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_combo(steps: torch.Tensor) -> torch.Tensor:
    """(..., 9) 0/1 drums -> (...) combination index in [0, 512)."""
    return (steps.long() * BITS.to(steps.device)).sum(-1)


def from_combo(index: torch.Tensor) -> torch.Tensor:
    """(...) combination index -> (..., 9) 0/1 drums (float)."""
    return ((index[..., None] & BITS.to(index.device)) > 0).float()


def make_inputs(windows: torch.Tensor) -> torch.Tensor:
    """Teacher-forcing inputs for windows (batch, steps, 9) that start on a bar line.

    Step t gets the drums of step t-1 (zeros at t=0) and the one-hot position t % 16.
    Returns (batch, steps, 25) float.
    """
    batch, steps, _ = windows.shape
    prev = torch.cat([torch.zeros_like(windows[:, :1]), windows[:, :-1]], dim=1).float()
    pos = nn.functional.one_hot(torch.arange(steps, device=windows.device) % STEPS_PER_BAR, STEPS_PER_BAR)
    return torch.cat([prev, pos.float().expand(batch, -1, -1)], dim=-1)


def step_input(prev: torch.Tensor, step: int) -> torch.Tensor:
    """Input for a single generation step: prev (batch, 9) drums of the previous step. Returns (batch, 1, 25)."""
    pos = torch.zeros(len(prev), STEPS_PER_BAR, device=prev.device)
    pos[:, step % STEPS_PER_BAR] = 1
    return torch.cat([prev.float(), pos], dim=-1)[:, None]


class DrumLSTM(nn.Module):
    """LSTM over 16th-note steps with a Bernoulli (9) or categorical (512) output per step."""

    def __init__(self, output: str = "bernoulli", hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        assert output in ("bernoulli", "categorical")
        self.config = dict(output=output, hidden=hidden, layers=layers, dropout=dropout)
        self.output = output
        self.lstm = nn.LSTM(INPUT_SIZE, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, N_DRUMS if output == "bernoulli" else N_COMBOS)

    def forward(self, x: torch.Tensor, state=None):
        """x: (batch, steps, 25). Returns (logits (batch, steps, 9 or 512), state)."""
        h, state = self.lstm(x, state)
        return self.out(h), state

    def nll(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of target steps (batch, steps, 9), summed over drums: (batch, steps) nats."""
        if self.output == "bernoulli":
            return nn.functional.binary_cross_entropy_with_logits(logits, target.float(), reduction="none").sum(-1)
        return nn.functional.cross_entropy(logits.transpose(1, 2), to_combo(target), reduction="none")

"""Training: model selection, the chosen model over several seeds, and four-on-the-floor fine-tuning.

Teacher forcing, Adam, gradient clipping, early stopping on validation loss. Run from the project root:
    .venv/bin/python -m src.train
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic CUDA kernels

import json
import random

import numpy as np
import pandas as pd
import torch

from src.config import CHECKPOINT_DIR, N_BARS, OUTPUT_DIR, TRAIN_SEEDS
from src.data import four_on_the_floor, load_bars, make_windows
from src.model import DrumLSTM, device, make_inputs

RESULTS = OUTPUT_DIR / "results"
STRIDE = 2           # bars between training windows
BATCH = 64
LR = 1e-3
FINETUNE_LR = 1e-4
CLIP = 3.0           # gradient-norm clipping, as Drums RNN
MAX_EPOCHS = 200
PATIENCE = 15        # epochs without validation improvement before stopping

# Model selection grid: two output types x three sizes, chosen by validation loss.
OUTPUTS = ("bernoulli", "categorical")
SIZES = {
    "small 2x128": dict(hidden=128, layers=2, dropout=0.2),
    "medium 2x256": dict(hidden=256, layers=2, dropout=0.5),
    "Drums RNN 3x256": dict(hidden=256, layers=3, dropout=0.5),
}


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and GPU) and use deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def split_windows(split: str, fotf_only: bool = False) -> torch.Tensor:
    """8-bar windows of one official split as a (n, 128, 9) uint8 tensor on the training device.

    fotf_only keeps only windows in which every bar has a kick on all four beats.
    """
    info, bars = load_bars()
    windows = make_windows([bars[i] for i in info.index[info.split == split]], N_BARS, STRIDE)
    if fotf_only:
        windows = windows[four_on_the_floor(windows)]
    return torch.from_numpy(windows).to(device())


def mean_loss(model: DrumLSTM, windows: torch.Tensor) -> float:
    """Mean negative log-likelihood per step (nats) over all windows, without gradients."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(windows), 512):
            w = windows[i:i + 512]
            logits, _ = model(make_inputs(w))
            total += model.nll(logits, w).sum().item()
    return total / (windows.shape[0] * windows.shape[1])


def fit(model: DrumLSTM, train_w: torch.Tensor, val_w: torch.Tensor, seed: int, lr: float) -> dict:
    """Train in place with early stopping; keeps the best weights (by validation loss). Returns the loss history."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    history = {"train": [], "val": []}
    best, best_state, waited = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for idx in torch.randperm(len(train_w), generator=gen).split(BATCH):
            w = train_w[idx.to(train_w.device)]
            logits, _ = model(make_inputs(w))
            loss = model.nll(logits, w).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        history["train"].append(mean_loss(model, train_w))
        history["val"].append(mean_loss(model, val_w))
        if history["val"][-1] < best:
            best, waited = history["val"][-1], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    history["best_epoch"] = int(np.argmin(history["val"])) + 1
    return history


def new_model(config: dict, seed: int) -> DrumLSTM:
    """Freshly initialised model; the seed fixes the initial weights."""
    set_seed(seed)
    return DrumLSTM(**config).to(device())


def checkpoint_path(kind: str, seed: int):
    """kind 'base' (trained on all GMD) or 'fotf' (fine-tuned on four-on-the-floor bars)."""
    return CHECKPOINT_DIR / f"lstm_{kind}_seed{seed}.pt"


def save(model: DrumLSTM, history: dict, kind: str, seed: int) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save({"config": model.config, "model": model.state_dict(), "history": history}, checkpoint_path(kind, seed))


def load_model(kind: str = "base", seed: int = 0) -> DrumLSTM:
    """Load a trained model on the training device, in eval mode."""
    ckpt = torch.load(checkpoint_path(kind, seed), map_location=device())
    model = DrumLSTM(**ckpt["config"]).to(device())
    model.load_state_dict(ckpt["model"])
    return model.eval()


def select_model(train_w: torch.Tensor, val_w: torch.Tensor) -> tuple[dict, pd.DataFrame]:
    """Train every output type x size once (first seed); returns the best config and the comparison table."""
    rows = []
    for output in OUTPUTS:
        for size, params in SIZES.items():
            config = dict(output=output, **params)
            model = new_model(config, TRAIN_SEEDS[0])
            history = fit(model, train_w, val_w, TRAIN_SEEDS[0], LR)
            rows.append({"output": output, "size": size, "parameters": sum(p.numel() for p in model.parameters()),
                         "best epoch": history["best_epoch"], "validation loss": min(history["val"]),
                         "config": json.dumps(config)})
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows).sort_values("validation loss")
    return json.loads(table.iloc[0]["config"]), table


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from src.plot_style import apply_style
    from src.plots import plot_loss_curve

    RESULTS.mkdir(parents=True, exist_ok=True)
    train_w, val_w, test_w = split_windows("train"), split_windows("validation"), split_windows("test")
    fotf_train, fotf_val = split_windows("train", fotf_only=True), split_windows("validation", fotf_only=True)

    config, table = select_model(train_w, val_w)
    table.drop(columns="config").to_csv(RESULTS / "model_selection.csv", index=False)
    print(f"chosen: {config}", flush=True)

    rows = []
    for seed in TRAIN_SEEDS:
        model = new_model(config, seed)
        history = fit(model, train_w, val_w, seed, LR)
        # Test loss is computed once per trained model and only reported; nothing is tuned on it.
        history["test"] = mean_loss(model, test_w)
        save(model, history, "base", seed)
        fotf_before = mean_loss(model, fotf_val)
        ft_history = fit(model, fotf_train, fotf_val, seed, FINETUNE_LR)
        save(model, ft_history, "fotf", seed)
        rows.append({"seed": seed, "best epoch": history["best_epoch"], "validation loss": min(history["val"]),
                     "test loss": history["test"], "fotf validation loss before fine-tuning": fotf_before,
                     "fotf validation loss after fine-tuning": min(ft_history["val"]),
                     "fine-tuning epochs": ft_history["best_epoch"]})
        print(rows[-1], flush=True)
        if seed == TRAIN_SEEDS[0]:
            apply_style()
            plot_loss_curve(history).savefig(RESULTS / "loss_curve.png", bbox_inches="tight")
    pd.DataFrame(rows).to_csv(RESULTS / "training.csv", index=False)


if __name__ == "__main__":
    main()

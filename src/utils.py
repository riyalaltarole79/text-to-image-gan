"""
utils.py
========
Shared utilities: seeding, metrics, logging, visualisation.
"""

import csv
import os
import random
import logging
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ── Running average meter ──────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ── Early stopping ─────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float("inf")

    def __call__(self, metric: float) -> bool:
        if metric < self.best - self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ── Image utilities ────────────────────────────────────────────────────────

def denorm(tensor: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]"""
    return (tensor + 1.0) / 2.0


def save_sample_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 4,
    captions: List[str] = None,
):
    """Save a grid of images; optionally annotate with captions."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    imgs = denorm(images.detach().cpu()).clamp(0, 1)

    if captions:
        # Use matplotlib for captioned output
        n    = imgs.size(0)
        ncol = nrow
        nrow_actual = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow_actual, ncol, figsize=(ncol * 3, nrow_actual * 3.5))
        axes = np.array(axes).ravel()
        for i, ax in enumerate(axes):
            if i < n:
                img_np = imgs[i].permute(1, 2, 0).numpy()
                ax.imshow(img_np)
                if captions and i < len(captions):
                    ax.set_title(captions[i], fontsize=7, wrap=True)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        save_image(imgs, path, nrow=nrow, normalize=False)


def plot_loss_curves(csv_path: str, out_path: str):
    """Read training CSV and plot G / D loss curves."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    epochs  = [r["epoch"] for r in rows]
    d_loss  = [r.get("d_loss", 0) for r in rows]
    g_loss  = [r.get("g_loss", 0) for r in rows]
    vd_loss = [r.get("val_d_loss", 0) for r in rows]
    vg_loss = [r.get("val_g_loss", 0) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, d_loss,  label="Train D")
    axes[0].plot(epochs, vd_loss, label="Val D", linestyle="--")
    axes[0].set_title("Discriminator Loss"); axes[0].legend()
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, g_loss,  label="Train G")
    axes[1].plot(epochs, vg_loss, label="Val G", linestyle="--")
    axes[1].set_title("Generator Loss"); axes[1].legend()
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Loss curves saved → {out_path}")


# ── CSV logging ────────────────────────────────────────────────────────────

def log_to_csv(path: str, metrics: Dict[str, Any], write_header: bool = False):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    mode = "w" if write_header else "a"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)


# ── Model helpers ──────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, name: str = "Model"):
    total = count_parameters(model)
    logger.info(f"{name}: {total:,} trainable parameters")


# ── Gradient norm ──────────────────────────────────────────────────────────

def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5

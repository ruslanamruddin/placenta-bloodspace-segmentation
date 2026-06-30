"""Utility functions: seed setting, logging, figure utilities, data splitting."""

import os
import json
import random
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(split_path):
    """Load the immutable data split."""
    with open(split_path) as f:
        return json.load(f)


def save_json(data, path):
    """Save dict as formatted JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_project_root():
    """Return project root (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------- Figure style ----------

# Color palette — 5-color scheme (most → least dominant)
# #003d5c → #594e90 → #bc4c96 → #ff5f66 → #ffa600
COLORS = {
    # Tissue classes (ordered by bar height / visual area)
    "MBS": "#003d5c",           # Tallest bars (highest Dice)
    "FBS": "#594e90",           # Medium bars
    "Background": "#bc4c96",    # Shortest bars (lowest Dice)
    # Encoder families (ordered by count of encoders)
    "ImageNet CNN": "#003d5c",  # 10 encoders — most prominent
    "ConvNeXt": "#594e90",      # 2 encoders
    "Histology ViT": "#bc4c96", # 4 encoders
}

# Named access to the full palette
C1 = "#003d5c"  # primary / most used
C2 = "#594e90"  # secondary
C3 = "#bc4c96"  # tertiary
C4 = "#ff5f66"  # accent
C5 = "#ffa600"  # least used / smallest elements


def setup_figure_style():
    """Set publication-quality figure defaults."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def save_figure(fig, path):
    """Save figure as both PNG and PDF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    pdf_path = path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


class Timer:
    """Context manager for timing."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        self.minutes = self.elapsed / 60

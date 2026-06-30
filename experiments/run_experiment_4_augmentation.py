#!/usr/bin/env python3
"""Experiment 4: Data Augmentation.

Compare 7 augmentation strategies using best encoder + decoder + loss.
Strategies B, C, G: all 3 seeds; A, D, E, F: seed 42 only.
Total: 3×3 + 4×1 = 13 training runs.
"""

import os
import sys
import json
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import save_json, setup_figure_style, save_figure, COLORS, Timer
from src.dataset import PlacentaDataset
from src.models import create_model, count_parameters
from src.losses import create_loss
from src.augmentations import get_augmentation
from src.train import train_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "data", "cd31_images")
MASK_DIR = os.path.join(BASE_DIR, "data", "masks")
SPLIT_PATH = os.path.join(BASE_DIR, "results", "experiment_1_data_split", "split_info.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "experiment_4_augmentation")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "experiment_4")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

ENCODER = "tu-convnext_small"
DECODER = "Unet"
LOSS = "ce_dice"

# All strategies: 3 seeds each
STRATEGY_SEEDS = {
    "A": [42, 123, 456],
    "B": [42, 123, 456],
    "C": [42, 123, 456],
    "D": [42, 123, 456],
    "E": [42, 123, 456],
    "F": [42, 123, 456],
    "G": [42, 123, 456],
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_single(strategy, seed, split):
    run_name = f"aug{strategy}_seed{seed}"
    results_path = os.path.join(CKPT_DIR, f"{run_name}_results.json")
    if os.path.exists(results_path):
        print(f"  Skipping {run_name} (already completed)")
        with open(results_path) as f:
            return json.load(f)

    print(f"\n{'='*50}")
    print(f"Training: {run_name}")
    print(f"{'='*50}")

    augmentation = get_augmentation(strategy)
    model = create_model(ENCODER, decoder_name=DECODER, num_classes=3,
                         freeze_encoder=False, img_size=512)

    train_dataset = PlacentaDataset(IMG_DIR, MASK_DIR, split["train_images"],
                                     augmentation=augmentation, input_size=512)
    val_dataset = PlacentaDataset(IMG_DIR, MASK_DIR, split["val_images"], input_size=512)
    test_dataset = PlacentaDataset(IMG_DIR, MASK_DIR, split["test_images"], input_size=512)

    config = {
        "model": model,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "loss_fn": create_loss(LOSS),
        "seed": seed,
        "max_epochs": 100,
        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "checkpoint_dir": CKPT_DIR,
        "run_name": run_name,
        "device": DEVICE,
    }

    try:
        with Timer() as t:
            results = train_model(config)
        print(f"  Done in {t.minutes:.1f} min | Test Dice: {results['test_metrics'].get('mean_dice', 'N/A')}")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        del model
        torch.cuda.empty_cache()
        return None

    del model
    torch.cuda.empty_cache()
    return results


def main():
    print("=" * 60)
    print("EXPERIMENT 4: Augmentation Comparison")
    print(f"Config: {ENCODER} + {DECODER} + {LOSS}")
    print("=" * 60)

    split = json.load(open(SPLIT_PATH))

    all_rows = []
    for strategy, seeds in STRATEGY_SEEDS.items():
        for seed in seeds:
            results = run_single(strategy, seed, split)
            if results is None:
                continue
            test = results.get("test_metrics", {})
            row = {
                "strategy": strategy,
                "seed": results["seed"],
                "mean_dice": test.get("mean_dice", np.nan),
                "fbs_dice": test.get("fbs_dice", np.nan),
                "mbs_dice": test.get("mbs_dice", np.nan),
                "bg_dice": test.get("bg_dice", np.nan),
                "training_min": results.get("training_min", np.nan),
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "augmentation_comparison.csv"), index=False)

    # Summary
    summary_rows = []
    for strategy in df["strategy"].unique():
        sdf = df[df["strategy"] == strategy]
        row = {"strategy": strategy}
        for col in ["mean_dice", "fbs_dice", "mbs_dice", "bg_dice", "training_min"]:
            row[f"{col}_mean"] = round(sdf[col].mean(), 4)
            row[f"{col}_sd"] = round(sdf[col].std(), 4) if len(sdf) > 1 else 0.0
        row["n_seeds"] = len(sdf)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_dice_mean", ascending=False)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "augmentation_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("AUGMENTATION SUMMARY")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        sd_str = f"±{row['mean_dice_sd']:.4f}" if row["n_seeds"] > 1 else " (1 seed)"
        print(f"  Strategy {row['strategy']} | Dice={row['mean_dice_mean']:.4f}{sd_str} | "
              f"FBS={row['fbs_dice_mean']:.4f} | MBS={row['mbs_dice_mean']:.4f} | BG={row['bg_dice_mean']:.4f}")

    # Best augmentation
    best = summary_df.iloc[0]
    baseline = summary_df[summary_df["strategy"] == "A"].iloc[0]
    improvement = best["mean_dice_mean"] - baseline["mean_dice_mean"]
    meaningful = improvement > 0.005

    best_aug_json = {
        "best_augmentation": best["strategy"],
        "mean_dice": float(best["mean_dice_mean"]),
        "mean_dice_sd": float(best["mean_dice_sd"]),
        "baseline_dice": float(baseline["mean_dice_mean"]),
        "improvement_over_baseline": float(round(improvement, 4)),
        "meaningful_improvement": bool(meaningful),
        "reason": (f"Strategy {best['strategy']} achieves {best['mean_dice_mean']:.4f} "
                   f"(+{improvement:.4f} over baseline A={baseline['mean_dice_mean']:.4f})"),
    }
    if not meaningful:
        best_aug_json["best_augmentation"] = "A"
        best_aug_json["reason"] = f"No strategy meaningfully improves over baseline A ({baseline['mean_dice_mean']:.4f})"

    save_json(best_aug_json, os.path.join(RESULTS_DIR, "best_augmentation.json"))

    # Figures
    setup_figure_style()

    # Bar plot with per-class breakdown
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(summary_df)
    width = 0.2
    x_arr = np.arange(n)
    ax.bar(x_arr - 1.5*width, summary_df["mean_dice_mean"], width, label="Mean",
           yerr=summary_df["mean_dice_sd"], color="#2196F3", capsize=3)
    ax.bar(x_arr - 0.5*width, summary_df["fbs_dice_mean"], width, label="FBS", color=COLORS["FBS"])
    ax.bar(x_arr + 0.5*width, summary_df["mbs_dice_mean"], width, label="MBS", color=COLORS["MBS"])
    ax.bar(x_arr + 1.5*width, summary_df["bg_dice_mean"], width, label="BG", color=COLORS["Background"])
    ax.set_xticks(x_arr)
    ax.set_xticklabels([f"Strategy {s}" for s in summary_df["strategy"]])
    ax.set_ylabel("Dice Score")
    ax.set_title("Augmentation Strategy Comparison")
    ax.legend()
    ax.set_ylim(0.6, 0.9)
    save_figure(fig, os.path.join(FIGURES_DIR, "augmentation_barplot.png"))

    # Per-class grouped bars
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    ax.bar(x_arr - width, summary_df["fbs_dice_mean"], width, label="FBS", color=COLORS["FBS"])
    ax.bar(x_arr, summary_df["mbs_dice_mean"], width, label="MBS", color=COLORS["MBS"])
    ax.bar(x_arr + width, summary_df["bg_dice_mean"], width, label="Background", color=COLORS["Background"])
    ax.set_xticks(x_arr)
    ax.set_xticklabels([f"Strategy {s}" for s in summary_df["strategy"]])
    ax.set_ylabel("Dice Score")
    ax.set_title("Per-Class Dice by Augmentation Strategy")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    save_figure(fig, os.path.join(FIGURES_DIR, "augmentation_per_class.png"))

    carry = best_aug_json["best_augmentation"]
    print(f"\nCarry forward: Strategy {carry}")
    print("Experiment 4 COMPLETE")


if __name__ == "__main__":
    main()

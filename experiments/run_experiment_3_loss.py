#!/usr/bin/env python3
"""Experiment 3: Loss Function Comparison.

Compare 9 loss functions using best encoder (efficientnet-b3) + best decoder (UnetPlusPlus), frozen.
3 seeds each = 27 training runs.
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
from src.losses import create_loss, LOSS_REGISTRY
from src.augmentations import get_augmentation
from src.train import train_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "data", "cd31_images")
MASK_DIR = os.path.join(BASE_DIR, "data", "masks")
SPLIT_PATH = os.path.join(BASE_DIR, "results", "experiment_1_data_split", "split_info.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "experiment_3_loss")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "experiment_3")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# Best from Experiments 2a and 2b
ENCODER = "tu-convnext_small"
DECODER = "Unet"

LOSSES = ["ce", "dice", "ce_dice", "focal_dice", "lovasz",
          "tversky_recall", "tversky_precision", "unified_focal", "combo"]
SEEDS = [42, 123, 456]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_single(loss_name, seed, split, augmentation):
    run_name = f"{loss_name}_seed{seed}"
    results_path = os.path.join(CKPT_DIR, f"{run_name}_results.json")
    if os.path.exists(results_path):
        print(f"  Skipping {run_name} (already completed)")
        with open(results_path) as f:
            return json.load(f)

    print(f"\n{'='*50}")
    print(f"Training: {run_name} ({ENCODER} + {DECODER})")
    print(f"{'='*50}")

    model = create_model(ENCODER, decoder_name=DECODER, num_classes=3,
                         freeze_encoder=False, img_size=512)
    total_p, train_p = count_parameters(model)
    print(f"  Params: {total_p}M total, {train_p}M trainable")

    train_dataset = PlacentaDataset(IMG_DIR, MASK_DIR, split["train_images"],
                                     augmentation=augmentation, input_size=512)
    val_dataset = PlacentaDataset(IMG_DIR, MASK_DIR, split["val_images"], input_size=512)
    test_dataset = PlacentaDataset(IMG_DIR, MASK_DIR, split["test_images"], input_size=512)

    config = {
        "model": model,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "loss_fn": create_loss(loss_name),
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
    print("EXPERIMENT 3: Loss Function Comparison")
    print(f"Encoder: {ENCODER}, Decoder: {DECODER}")
    print("=" * 60)

    split = json.load(open(SPLIT_PATH))
    augmentation = get_augmentation("A")

    all_rows = []
    for loss_name in LOSSES:
        for seed in SEEDS:
            results = run_single(loss_name, seed, split, augmentation)
            if results is None:
                continue
            test = results.get("test_metrics", {})
            row = {
                "loss_name": loss_name,
                "seed": results["seed"],
                "mean_dice": test.get("mean_dice", np.nan),
                "fbs_dice": test.get("fbs_dice", np.nan),
                "mbs_dice": test.get("mbs_dice", np.nan),
                "bg_dice": test.get("bg_dice", np.nan),
                "fbs_precision": test.get("fbs_precision", np.nan),
                "fbs_recall": test.get("fbs_recall", np.nan),
                "mbs_precision": test.get("mbs_precision", np.nan),
                "mbs_recall": test.get("mbs_recall", np.nan),
                "mean_precision": test.get("mean_precision", np.nan),
                "mean_recall": test.get("mean_recall", np.nan),
                "training_min": results.get("training_min", np.nan),
                "best_epoch": results.get("best_epoch", np.nan),
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "loss_comparison.csv"), index=False)

    # Summary
    summary_rows = []
    for loss_name in df["loss_name"].unique():
        ldf = df[df["loss_name"] == loss_name]
        row = {"loss_name": loss_name}
        for col in ["mean_dice", "fbs_dice", "mbs_dice", "bg_dice",
                     "mean_precision", "mean_recall", "training_min"]:
            row[f"{col}_mean"] = round(ldf[col].mean(), 4)
            row[f"{col}_sd"] = round(ldf[col].std(), 4)
        row["n_seeds"] = len(ldf)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_dice_mean", ascending=False)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "loss_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("LOSS FUNCTION SUMMARY")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        print(f"  {row['loss_name']:20s} | Dice={row['mean_dice_mean']:.4f}±{row['mean_dice_sd']:.4f} | "
              f"Prec={row.get('mean_precision_mean', 0):.4f} | Rec={row.get('mean_recall_mean', 0):.4f}")

    # Best loss
    best = summary_df.iloc[0]
    best_loss_json = {
        "best_loss": best["loss_name"],
        "mean_dice": float(best["mean_dice_mean"]),
        "mean_dice_sd": float(best["mean_dice_sd"]),
        "reason": f"Highest mean Dice ({best['mean_dice_mean']:.4f})",
    }
    if len(summary_df) > 1:
        runner = summary_df.iloc[1]
        best_loss_json["runner_up"] = runner["loss_name"]
        best_loss_json["runner_up_dice"] = float(runner["mean_dice_mean"])
    save_json(best_loss_json, os.path.join(RESULTS_DIR, "best_loss.json"))

    # Figures
    setup_figure_style()

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(summary_df))
    ax.bar(x, summary_df["mean_dice_mean"], yerr=summary_df["mean_dice_sd"],
           color="#2196F3", capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df["loss_name"], rotation=30, ha="right")
    ax.set_ylabel("Mean Dice Score")
    ax.set_title(f"Loss Function Comparison ({ENCODER} + {DECODER})")
    ax.set_ylim(0.6, 0.85)
    save_figure(fig, os.path.join(FIGURES_DIR, "loss_barplot.png"))

    # Precision-recall scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in summary_df.iterrows():
        size = row["mean_dice_mean"] * 500
        ax.scatter(row["mean_recall_mean"], row["mean_precision_mean"],
                   s=size, alpha=0.7, edgecolors="black", linewidth=0.5)
        ax.annotate(row["loss_name"],
                    (row["mean_recall_mean"], row["mean_precision_mean"]),
                    fontsize=8, ha="center", va="bottom", xytext=(0, 8),
                    textcoords="offset points")
    ax.set_xlabel("Mean Recall")
    ax.set_ylabel("Mean Precision")
    ax.set_title("Precision vs Recall by Loss Function (size = Dice)")
    ax.plot([0.6, 1], [0.6, 1], "k--", alpha=0.3, label="P=R line")
    ax.legend()
    save_figure(fig, os.path.join(FIGURES_DIR, "loss_precision_recall.png"))

    # Per-class Dice
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(summary_df)
    width = 0.25
    x_arr = np.arange(n)
    ax.bar(x_arr - width, summary_df["fbs_dice_mean"], width, label="FBS", color=COLORS["FBS"])
    ax.bar(x_arr, summary_df["mbs_dice_mean"], width, label="MBS", color=COLORS["MBS"])
    ax.bar(x_arr + width, summary_df["bg_dice_mean"], width, label="Background", color=COLORS["Background"])
    ax.set_xticks(x_arr)
    ax.set_xticklabels(summary_df["loss_name"], rotation=30, ha="right")
    ax.set_ylabel("Dice Score")
    ax.set_title("Per-Class Dice by Loss Function")
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    save_figure(fig, os.path.join(FIGURES_DIR, "loss_per_class.png"))

    print(f"\nBest loss: {best['loss_name']} (Dice={best['mean_dice_mean']:.4f})")
    print("Experiment 3 COMPLETE")


if __name__ == "__main__":
    main()

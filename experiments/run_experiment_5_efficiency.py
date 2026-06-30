#!/usr/bin/env python3
"""Experiment 5: Data Efficiency.

Train on progressively smaller subsets to characterize the annotation-performance relationship.
8 sizes × 3 seeds = 24 training runs.
"""

import os
import sys
import json
import random
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
from src.models import create_model
from src.losses import create_loss
from src.augmentations import get_augmentation
from src.train import train_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "data", "cd31_images")
MASK_DIR = os.path.join(BASE_DIR, "data", "masks")
SPLIT_PATH = os.path.join(BASE_DIR, "results", "experiment_1_data_split", "split_info.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "experiment_5_data_efficiency")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "experiment_5")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

ENCODER = "tu-convnext_small"
DECODER = "Unet"
LOSS = "ce_dice"
AUGMENTATION = "A"

SUBSET_SIZES = [1, 2, 3, 4, 5, 7, 10, 13]  # 13 = full training set (15 placentas minus 2 val)
SEEDS = [42, 123, 456]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_placenta_images(placenta_id, all_images):
    """Get all images for a placenta ID."""
    start = (placenta_id - 1) * 5 + 1
    end = start + 5
    expected = [f"{i}.png" for i in range(start, end)]
    return [f for f in expected if f in all_images]


def select_subset(train_placentas, n_placentas, seed):
    """Select n_placentas from train_placentas using seed."""
    rng = random.Random(seed)
    if n_placentas >= len(train_placentas):
        return sorted(train_placentas)
    selected = rng.sample(train_placentas, n_placentas)
    return sorted(selected)


def main():
    print("=" * 60)
    print("EXPERIMENT 5: Data Efficiency")
    print(f"Config: {ENCODER} + {DECODER} + {LOSS} + Aug {AUGMENTATION}")
    print("=" * 60)

    split = json.load(open(SPLIT_PATH))
    augmentation = get_augmentation(AUGMENTATION)
    all_images_set = set(split["train_images"] + split["val_images"] + split["test_images"])

    all_rows = []
    for n_placentas in SUBSET_SIZES:
        for seed in SEEDS:
            run_name = f"n{n_placentas}_seed{seed}"
            results_path = os.path.join(CKPT_DIR, f"{run_name}_results.json")

            if os.path.exists(results_path):
                print(f"  Skipping {run_name} (already completed)")
                with open(results_path) as f:
                    results = json.load(f)
            else:
                print(f"\n{'='*50}")
                print(f"Training: {run_name} ({n_placentas} placentas)")
                print(f"{'='*50}")

                # Select subset
                selected = select_subset(split["train_placentas"], n_placentas, seed)
                train_images = []
                for pid in selected:
                    train_images.extend(get_placenta_images(pid, all_images_set))

                print(f"  Placentas: {selected}")
                print(f"  Images: {len(train_images)}")

                model = create_model(ENCODER, decoder_name=DECODER, num_classes=3,
                                     freeze_encoder=False, img_size=512)

                train_dataset = PlacentaDataset(IMG_DIR, MASK_DIR, train_images,
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
                    "batch_size": min(8, len(train_images)),
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
                    results["placentas_used"] = selected
                    results["n_train_images"] = len(train_images)
                    # Re-save with extra info
                    with open(results_path, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"  Done in {t.minutes:.1f} min | Test Dice: {results['test_metrics'].get('mean_dice', 'N/A')}")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    traceback.print_exc()
                    del model
                    torch.cuda.empty_cache()
                    continue

                del model
                torch.cuda.empty_cache()

            test = results.get("test_metrics", {})
            row = {
                "n_placentas": n_placentas,
                "n_images": n_placentas * 5,
                "seed": results.get("seed", seed),
                "placentas_used": str(results.get("placentas_used", "")),
                "mean_dice": test.get("mean_dice", np.nan),
                "fbs_dice": test.get("fbs_dice", np.nan),
                "mbs_dice": test.get("mbs_dice", np.nan),
                "bg_dice": test.get("bg_dice", np.nan),
                "training_min": results.get("training_min", np.nan),
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "data_efficiency.csv"), index=False)

    # Summary with marginal gain
    summary_rows = []
    prev_dice = None
    for n_p in SUBSET_SIZES:
        sdf = df[df["n_placentas"] == n_p]
        row = {
            "n_placentas": n_p,
            "n_images": n_p * 5,
            "mean_dice_mean": round(sdf["mean_dice"].mean(), 4),
            "mean_dice_sd": round(sdf["mean_dice"].std(), 4),
            "fbs_dice_mean": round(sdf["fbs_dice"].mean(), 4),
            "mbs_dice_mean": round(sdf["mbs_dice"].mean(), 4),
            "bg_dice_mean": round(sdf["bg_dice"].mean(), 4),
        }
        if prev_dice is not None:
            row["marginal_gain"] = round(row["mean_dice_mean"] - prev_dice, 4)
        else:
            row["marginal_gain"] = 0.0
        prev_dice = row["mean_dice_mean"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "data_efficiency_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("DATA EFFICIENCY SUMMARY")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        gain_str = f"+{row['marginal_gain']:.4f}" if row["marginal_gain"] > 0 else f"{row['marginal_gain']:.4f}"
        print(f"  {int(row['n_placentas']):2d} placentas ({int(row['n_images']):2d} imgs) | "
              f"Dice={row['mean_dice_mean']:.4f}±{row['mean_dice_sd']:.4f} | gain={gain_str}")

    # Find plateau and 96% threshold
    full_dice = summary_df.iloc[-1]["mean_dice_mean"]
    threshold_96 = full_dice * 0.96
    plateau_row = None
    for _, row in summary_df.iterrows():
        if row["mean_dice_mean"] >= threshold_96:
            plateau_row = row
            break

    analysis = {
        "full_training_dice": float(full_dice),
        "threshold_96_pct": float(round(threshold_96, 4)),
        "plateau_n_placentas": int(plateau_row["n_placentas"]) if plateau_row is not None else None,
        "plateau_n_images": int(plateau_row["n_images"]) if plateau_row is not None else None,
        "plateau_dice": float(plateau_row["mean_dice_mean"]) if plateau_row is not None else None,
    }
    save_json(analysis, os.path.join(RESULTS_DIR, "data_efficiency_analysis.json"))

    # Figures
    setup_figure_style()

    # Learning curve with marginal gain bars
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    ax1.errorbar(summary_df["n_placentas"], summary_df["mean_dice_mean"],
                 yerr=summary_df["mean_dice_sd"], marker="o", capsize=5,
                 color="#2196F3", linewidth=2, markersize=8)
    ax1.axhline(y=threshold_96, color="red", linestyle="--", alpha=0.5, label=f"96% threshold ({threshold_96:.3f})")
    ax1.set_ylabel("Mean Dice Score")
    ax1.set_title("Data Efficiency: Dice vs Training Set Size")
    ax1.legend()
    ax1.set_ylim(0.5, 0.85)
    ax1.grid(True, alpha=0.3)

    ax2.bar(summary_df["n_placentas"], summary_df["marginal_gain"],
            color="#4CAF50", edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Number of Training Placentas")
    ax2.set_ylabel("Marginal Gain")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURES_DIR, "learning_curve.png"))

    # Per-class learning curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary_df["n_placentas"], summary_df["fbs_dice_mean"], "o-",
            color=COLORS["FBS"], label="FBS", linewidth=2)
    ax.plot(summary_df["n_placentas"], summary_df["mbs_dice_mean"], "o-",
            color=COLORS["MBS"], label="MBS", linewidth=2)
    ax.plot(summary_df["n_placentas"], summary_df["bg_dice_mean"], "o-",
            color=COLORS["Background"], label="Background", linewidth=2)
    ax.set_xlabel("Number of Training Placentas")
    ax.set_ylabel("Dice Score")
    ax.set_title("Per-Class Dice vs Training Set Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, os.path.join(FIGURES_DIR, "learning_curve_per_class.png"))

    print("\nExperiment 5 COMPLETE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Experiment 2a: Encoder Comparison.

Compare 15 encoders with U-Net decoder, all frozen (decoder-only training).
3 seeds each = up to 45 training runs.
After identifying best encoder, run frozen vs unfrozen comparison.
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

from src.utils import set_seed, save_json, setup_figure_style, save_figure, COLORS, Timer
from src.dataset import PlacentaDataset
from src.models import create_model, count_parameters, ENCODER_REGISTRY
from src.losses import create_loss
from src.augmentations import get_augmentation
from src.train import train_model

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "data", "cd31_images")
MASK_DIR = os.path.join(BASE_DIR, "data", "masks")
SPLIT_PATH = os.path.join(BASE_DIR, "results", "experiment_1_data_split", "split_info.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "experiment_2a_encoder")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "experiment_2a")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# Encoder list (in order from CLAUDE.md)
ENCODERS = [
    "resnet34", "resnet50", "resnet50_random",
    "efficientnet-b3", "efficientnet-b5",
    "tu-efficientnetv2_s", "tu-efficientnetv2_m", "tu-efficientnetv2_l",
    "tu-convnext_tiny", "tu-convnext_small",
    "phikon-v2", "uni", "conch", "h-optimus-0", "virchow2",
]

SEEDS = [42, 123, 456]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_single(encoder_name, seed, split, augmentation, freeze=True):
    """Run a single encoder training."""
    run_name = f"{encoder_name}_seed{seed}" + ("_unfrozen" if not freeze else "")

    # Check if already completed
    results_path = os.path.join(CKPT_DIR, f"{run_name}_results.json")
    if os.path.exists(results_path):
        print(f"  Skipping {run_name} (already completed)")
        with open(results_path) as f:
            return json.load(f)

    print(f"\n{'='*50}")
    print(f"Training: {run_name}")
    print(f"{'='*50}")

    try:
        model = create_model(encoder_name, decoder_name="Unet", num_classes=3,
                             freeze_encoder=False, img_size=512)
    except Exception as e:
        print(f"  FAILED to create model {encoder_name}: {e}")
        traceback.print_exc()
        return None

    total_p, train_p = count_parameters(model)
    print(f"  Total params: {total_p}M, Trainable: {train_p}M")

    # Datasets
    train_dataset = PlacentaDataset(IMG_DIR, MASK_DIR, split["train_images"],
                                     augmentation=augmentation, input_size=512)
    val_dataset = PlacentaDataset(IMG_DIR, MASK_DIR, split["val_images"], input_size=512)
    test_dataset = PlacentaDataset(IMG_DIR, MASK_DIR, split["test_images"], input_size=512)

    config = {
        "model": model,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "loss_fn": create_loss("ce_dice"),
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
        print(f"  FAILED during training {run_name}: {e}")
        traceback.print_exc()
        del model
        torch.cuda.empty_cache()
        return None

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    return results


def create_results_row(encoder_name, results, freeze=True):
    """Create a row for the results CSV."""
    info = ENCODER_REGISTRY[encoder_name]
    test = results.get("test_metrics", {})
    return {
        "encoder": encoder_name,
        "family": info["family"],
        "pretrained": info.get("pretrained", "self-supervised"),
        "frozen": freeze,
        "seed": results["seed"],
        "mean_dice": test.get("mean_dice", np.nan),
        "fbs_dice": test.get("fbs_dice", np.nan),
        "mbs_dice": test.get("mbs_dice", np.nan),
        "bg_dice": test.get("bg_dice", np.nan),
        "fbs_iou": test.get("fbs_iou", np.nan),
        "mbs_iou": test.get("mbs_iou", np.nan),
        "bg_iou": test.get("bg_iou", np.nan),
        "fbs_precision": test.get("fbs_precision", np.nan),
        "fbs_recall": test.get("fbs_recall", np.nan),
        "mbs_precision": test.get("mbs_precision", np.nan),
        "mbs_recall": test.get("mbs_recall", np.nan),
        "total_params_M": results.get("total_params_M", np.nan),
        "trainable_params_M": results.get("trainable_params_M", np.nan),
        "inference_ms": results.get("inference_ms", np.nan),
        "gpu_memory_MB": results.get("gpu_memory_MB", np.nan),
        "training_min": results.get("training_min", np.nan),
        "best_epoch": results.get("best_epoch", np.nan),
    }


def create_figures(df, summary_df):
    """Create all Experiment 2a figures."""
    setup_figure_style()

    # 1. Bar plot: Mean Dice per encoder, grouped by family
    fig, ax = plt.subplots(figsize=(12, 5))
    # Sort by family then dice
    plot_df = summary_df.sort_values(["family", "mean_dice_mean"], ascending=[True, False])
    x = range(len(plot_df))
    colors = [COLORS.get(f, "#999999") for f in plot_df["family"]]
    bars = ax.bar(x, plot_df["mean_dice_mean"], yerr=plot_df["mean_dice_sd"],
                  color=colors, capsize=3, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["encoder"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Dice Score")
    ax.set_title("Encoder Comparison — Mean Dice (Frozen, U-Net Decoder)")

    # Legend for families
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=COLORS[f], label=f) for f in ["ImageNet CNN", "ConvNeXt", "Histology ViT"]]
    ax.legend(handles=legend_patches, loc="lower right")
    ax.set_ylim(0, 1)
    save_figure(fig, os.path.join(FIGURES_DIR, "encoder_barplot.png"))

    # 2. Bubble chart: Dice vs inference time, bubble size = params, color = family
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in summary_df.iterrows():
        color = COLORS.get(row["family"], "#999999")
        ax.scatter(row["inference_ms_mean"], row["mean_dice_mean"],
                   s=row["total_params_M_mean"] * 2,
                   c=color, alpha=0.7, edgecolors="black", linewidth=0.5)
        ax.annotate(row["encoder"], (row["inference_ms_mean"], row["mean_dice_mean"]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")
    ax.set_xlabel("Inference Time (ms/tile)")
    ax.set_ylabel("Mean Dice Score")
    ax.set_title("Dice vs Inference Time (bubble size = parameters)")
    ax.legend(handles=legend_patches, loc="lower right")
    save_figure(fig, os.path.join(FIGURES_DIR, "encoder_bubble_chart.png"))

    # 3. Per-class Dice
    fig, ax = plt.subplots(figsize=(12, 5))
    n = len(plot_df)
    width = 0.25
    x_arr = np.arange(n)
    ax.bar(x_arr - width, plot_df["fbs_dice_mean"], width, label="FBS", color=COLORS["FBS"])
    ax.bar(x_arr, plot_df["mbs_dice_mean"], width, label="MBS", color=COLORS["MBS"])
    ax.bar(x_arr + width, plot_df["bg_dice_mean"], width, label="Background", color=COLORS["Background"])
    ax.set_xticks(x_arr)
    ax.set_xticklabels(plot_df["encoder"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Dice Score")
    ax.set_title("Per-Class Dice by Encoder")
    ax.legend()
    ax.set_ylim(0, 1)
    save_figure(fig, os.path.join(FIGURES_DIR, "encoder_per_class.png"))


def main():
    print("=" * 60)
    print("EXPERIMENT 2a: Encoder Comparison")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load split
    split = json.load(open(SPLIT_PATH))
    augmentation = get_augmentation("A")

    # Run all encoders
    all_rows = []
    skipped = []

    for encoder_name in ENCODERS:
        for seed in SEEDS:
            results = run_single(encoder_name, seed, split, augmentation, freeze=True)
            if results is None:
                skipped.append(encoder_name)
                continue
            row = create_results_row(encoder_name, results, freeze=True)
            all_rows.append(row)

    # Save detailed results
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "encoder_comparison.csv"), index=False)

    # Create summary (aggregate across seeds)
    frozen_df = df[df["frozen"] == True]
    agg_cols = ["mean_dice", "fbs_dice", "mbs_dice", "bg_dice",
                "fbs_iou", "mbs_iou", "bg_iou",
                "inference_ms", "gpu_memory_MB", "training_min", "total_params_M"]
    summary_rows = []
    for encoder_name in frozen_df["encoder"].unique():
        enc_df = frozen_df[frozen_df["encoder"] == encoder_name]
        row = {"encoder": encoder_name, "family": enc_df["family"].iloc[0]}
        for col in agg_cols:
            row[f"{col}_mean"] = round(enc_df[col].mean(), 4)
            row[f"{col}_sd"] = round(enc_df[col].std(), 4)
        row["n_seeds"] = len(enc_df)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_dice_mean", ascending=False)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "encoder_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("ENCODER SUMMARY (sorted by Mean Dice)")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        print(f"  {row['encoder']:25s} | Dice={row['mean_dice_mean']:.4f}±{row['mean_dice_sd']:.4f} | "
              f"Inf={row.get('inference_ms_mean', 0):.1f}ms | Params={row.get('total_params_M_mean', 0):.1f}M")

    # Select best encoder
    best = summary_df.iloc[0]

    # Check if a foundation model has >5x inference time of best ConvNeXt
    convnext_df = summary_df[summary_df["family"] == "ConvNeXt"]
    best_encoder_name = best["encoder"]
    reason = f"Highest mean Dice ({best['mean_dice_mean']:.4f})"

    if best["family"] == "Histology ViT" and len(convnext_df) > 0:
        best_convnext = convnext_df.iloc[0]
        if best["inference_ms_mean"] > 5 * best_convnext["inference_ms_mean"]:
            runner_up = best
            best = best_convnext
            best_encoder_name = best["encoder"]
            reason = (f"Best ConvNeXt selected over {runner_up['encoder']} "
                      f"(Dice {runner_up['mean_dice_mean']:.4f}) due to "
                      f"{runner_up['inference_ms_mean']:.0f}ms > 5x {best['inference_ms_mean']:.0f}ms inference")

    best_encoder_json = {
        "best_encoder": best_encoder_name,
        "mean_dice": float(best["mean_dice_mean"]),
        "mean_dice_sd": float(best["mean_dice_sd"]),
        "reason": reason,
        "family": best["family"],
        "inference_ms": float(best.get("inference_ms_mean", 0)),
        "total_params_M": float(best.get("total_params_M_mean", 0)),
    }

    # Add runner-up info
    if len(summary_df) > 1:
        runner = summary_df.iloc[1]
        best_encoder_json["runner_up"] = runner["encoder"]
        best_encoder_json["runner_up_dice"] = float(runner["mean_dice_mean"])

    if skipped:
        best_encoder_json["skipped_encoders"] = list(set(skipped))
        best_encoder_json["skip_reason"] = "Failed to load model"

    save_json(best_encoder_json, os.path.join(RESULTS_DIR, "best_encoder.json"))
    print(f"\nBest encoder: {best_encoder_name} (Dice={best['mean_dice_mean']:.4f})")

    # Create figures
    create_figures(df, summary_df)

    print(f"\nResults saved to {RESULTS_DIR}")
    print("Experiment 2a COMPLETE")


if __name__ == "__main__":
    main()

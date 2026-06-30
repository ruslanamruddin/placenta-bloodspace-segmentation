#!/usr/bin/env python3
"""Experiment 2b: Decoder Comparison.

Compare 5 decoders using ResNet34 encoder (fixed), frozen.
3 seeds each = 15 training runs.
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
from src.models import create_model, count_parameters, DECODER_REGISTRY
from src.losses import create_loss
from src.augmentations import get_augmentation
from src.train import train_model

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "data", "cd31_images")
MASK_DIR = os.path.join(BASE_DIR, "data", "masks")
SPLIT_PATH = os.path.join(BASE_DIR, "results", "experiment_1_data_split", "split_info.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "experiment_2b_decoder")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "experiment_2b")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DECODERS = ["Unet", "UnetPlusPlus", "MAnet", "FPN", "DeepLabV3Plus"]
ENCODER = "resnet34"  # Fixed encoder for decoder comparison
SEEDS = [42, 123, 456]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_single(decoder_name, seed, split, augmentation):
    """Run a single decoder training."""
    run_name = f"{decoder_name}_seed{seed}"

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
        model = create_model(ENCODER, decoder_name=decoder_name, num_classes=3,
                             freeze_encoder=False, img_size=512)
    except Exception as e:
        print(f"  FAILED to create model: {e}")
        return None

    total_p, train_p = count_parameters(model)
    print(f"  Total params: {total_p}M, Trainable: {train_p}M")

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
        print(f"  FAILED during training: {e}")
        traceback.print_exc()
        del model
        torch.cuda.empty_cache()
        return None

    del model
    torch.cuda.empty_cache()
    return results


def main():
    print("=" * 60)
    print("EXPERIMENT 2b: Decoder Comparison")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Encoder: {ENCODER} (frozen)")

    split = json.load(open(SPLIT_PATH))
    augmentation = get_augmentation("A")

    all_rows = []
    for decoder_name in DECODERS:
        for seed in SEEDS:
            results = run_single(decoder_name, seed, split, augmentation)
            if results is None:
                continue
            test = results.get("test_metrics", {})
            row = {
                "decoder": decoder_name,
                "encoder": ENCODER,
                "seed": results["seed"],
                "mean_dice": test.get("mean_dice", np.nan),
                "fbs_dice": test.get("fbs_dice", np.nan),
                "mbs_dice": test.get("mbs_dice", np.nan),
                "bg_dice": test.get("bg_dice", np.nan),
                "fbs_iou": test.get("fbs_iou", np.nan),
                "mbs_iou": test.get("mbs_iou", np.nan),
                "bg_iou": test.get("bg_iou", np.nan),
                "total_params_M": results.get("total_params_M", np.nan),
                "trainable_params_M": results.get("trainable_params_M", np.nan),
                "inference_ms": results.get("inference_ms", np.nan),
                "gpu_memory_MB": results.get("gpu_memory_MB", np.nan),
                "training_min": results.get("training_min", np.nan),
                "best_epoch": results.get("best_epoch", np.nan),
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "decoder_comparison.csv"), index=False)

    # Summary
    summary_rows = []
    for decoder_name in df["decoder"].unique():
        dec_df = df[df["decoder"] == decoder_name]
        row = {"decoder": decoder_name}
        for col in ["mean_dice", "fbs_dice", "mbs_dice", "bg_dice", "inference_ms",
                     "gpu_memory_MB", "training_min", "total_params_M", "trainable_params_M"]:
            row[f"{col}_mean"] = round(dec_df[col].mean(), 4)
            row[f"{col}_sd"] = round(dec_df[col].std(), 4)
        row["n_seeds"] = len(dec_df)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_dice_mean", ascending=False)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "decoder_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("DECODER SUMMARY (sorted by Mean Dice)")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        print(f"  {row['decoder']:20s} | Dice={row['mean_dice_mean']:.4f}±{row['mean_dice_sd']:.4f} | "
              f"Inf={row.get('inference_ms_mean', 0):.1f}ms | Params={row.get('total_params_M_mean', 0):.1f}M")

    # Best decoder
    best = summary_df.iloc[0]
    best_decoder_json = {
        "best_decoder": best["decoder"],
        "mean_dice": float(best["mean_dice_mean"]),
        "mean_dice_sd": float(best["mean_dice_sd"]),
        "reason": f"Highest mean Dice ({best['mean_dice_mean']:.4f}) with {ENCODER} encoder",
        "inference_ms": float(best.get("inference_ms_mean", 0)),
        "total_params_M": float(best.get("total_params_M_mean", 0)),
    }
    if len(summary_df) > 1:
        runner = summary_df.iloc[1]
        best_decoder_json["runner_up"] = runner["decoder"]
        best_decoder_json["runner_up_dice"] = float(runner["mean_dice_mean"])

    save_json(best_decoder_json, os.path.join(RESULTS_DIR, "best_decoder.json"))

    # Figures
    setup_figure_style()

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(summary_df))
    ax.bar(x, summary_df["mean_dice_mean"], yerr=summary_df["mean_dice_sd"],
           color="#4CAF50", capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df["decoder"], rotation=30, ha="right")
    ax.set_ylabel("Mean Dice Score")
    ax.set_title(f"Decoder Comparison (Encoder: {ENCODER}, Frozen)")
    ax.set_ylim(0.6, 0.85)
    save_figure(fig, os.path.join(FIGURES_DIR, "decoder_barplot.png"))

    # Table figure
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            row["decoder"],
            f"{row['mean_dice_mean']:.4f} ± {row['mean_dice_sd']:.4f}",
            f"{row.get('total_params_M_mean', 0):.1f}",
            f"{row.get('inference_ms_mean', 0):.1f}",
        ])
    table = ax.table(cellText=table_data,
                     colLabels=["Decoder", "Mean Dice", "Params (M)", "Inf. (ms)"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    save_figure(fig, os.path.join(FIGURES_DIR, "decoder_table.png"))

    print(f"\nBest decoder: {best['decoder']} (Dice={best['mean_dice_mean']:.4f})")
    print(f"Results saved to {RESULTS_DIR}")
    print("Experiment 2b COMPLETE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Experiment 6: Downstream Validation.

Compare model-derived morphometric measurements against expert annotations
on the 25 held-out test images.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import save_json, setup_figure_style, save_figure, COLORS
from src.dataset import PlacentaDataset
from src.models import create_model
from src.losses import create_loss
from src.augmentations import get_augmentation
from src.train import train_model
from src.evaluate import evaluate_model, compute_metrics
from src.morphometrics import compute_morphometrics
from src.agreement import compute_agreement

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "data", "cd31_images")
MASK_DIR = os.path.join(BASE_DIR, "data", "masks")
SPLIT_PATH = os.path.join(BASE_DIR, "results", "experiment_1_data_split", "split_info.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "experiment_6_validation")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "experiment_6")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

ENCODER = "tu-convnext_small"
DECODER = "Unet"
LOSS = "ce_dice"
AUGMENTATION = "A"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METRIC_NAMES = [
    "mbs_pct_area", "fbs_pct_area",
    "mbs_pct_perimeter", "fbs_pct_perimeter",
    "fbs_mbs_area_ratio", "fbs_pa_ratio",
]
METRIC_LABELS = {
    "mbs_pct_area": "MBS % Area",
    "fbs_pct_area": "FBS % Area",
    "mbs_pct_perimeter": "MBS % Perimeter",
    "fbs_pct_perimeter": "FBS % Perimeter",
    "fbs_mbs_area_ratio": "FBS:MBS Area Ratio",
    "fbs_pa_ratio": "FBS P:A Ratio",
}


def train_final_model(split):
    """Train the final model on all training data."""
    run_name = "final_model"
    ckpt_path = os.path.join(CKPT_DIR, f"{run_name}_best.pth")

    if os.path.exists(ckpt_path):
        print("  Loading existing final model checkpoint")
        model = create_model(ENCODER, decoder_name=DECODER, num_classes=3,
                             freeze_encoder=False, img_size=512)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        return model

    print("  Training final model on all 13 training placentas...")
    augmentation = get_augmentation(AUGMENTATION)
    model = create_model(ENCODER, decoder_name=DECODER, num_classes=3,
                         freeze_encoder=False, img_size=512)

    train_dataset = PlacentaDataset(IMG_DIR, MASK_DIR, split["train_images"],
                                     augmentation=augmentation, input_size=512)
    val_dataset = PlacentaDataset(IMG_DIR, MASK_DIR, split["val_images"], input_size=512)

    config = {
        "model": model,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "loss_fn": create_loss(LOSS),
        "seed": SEED,
        "max_epochs": 100,
        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "checkpoint_dir": CKPT_DIR,
        "run_name": run_name,
        "device": DEVICE,
    }

    results = train_model(config)
    print(f"  Final model trained. Best val Dice: {results['best_val_dice']:.4f}")

    # Reload best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    return model


@torch.no_grad()
def generate_predictions(model, split):
    """Generate predictions for all test images."""
    model.eval()
    model = model.to(DEVICE)

    test_dataset = PlacentaDataset(IMG_DIR, MASK_DIR, split["test_images"], input_size=512)
    predictions = {}

    for i in range(len(test_dataset)):
        image, mask, fname = test_dataset[i]
        logits = model(image.unsqueeze(0).to(DEVICE))
        pred = logits.argmax(dim=1).squeeze().cpu().numpy()

        # Save prediction
        pred_img = Image.fromarray(pred.astype(np.uint8))
        pred_img.save(os.path.join(PRED_DIR, fname))

        predictions[fname] = pred

    return predictions


def get_placenta_id(fname):
    """Get placenta ID from filename."""
    img_id = int(fname.replace(".png", ""))
    return (img_id - 1) // 5 + 1


def save_qualitative(split, predictions, per_image_dice):
    """Save qualitative comparison images (best, median, worst)."""
    setup_figure_style()

    # Sort by dice
    sorted_images = sorted(per_image_dice.items(), key=lambda x: x[1])
    worst_fname = sorted_images[0][0]
    best_fname = sorted_images[-1][0]
    median_idx = len(sorted_images) // 2
    median_fname = sorted_images[median_idx][0]

    # Color map: 0=Background, 1=FBS, 2=MBS
    def colorize_mask(mask):
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[mask == 0] = [220, 220, 220]  # Background gray
        rgb[mask == 1] = [123, 45, 142]   # FBS purple
        rgb[mask == 2] = [233, 30, 99]    # MBS pink
        return rgb

    for label, fname in [("best", best_fname), ("median", median_fname), ("worst", worst_fname)]:
        # Load original image (resized to 512)
        img = np.array(Image.open(os.path.join(IMG_DIR, fname)).convert("RGB").resize((512, 512)))
        expert = np.array(Image.open(os.path.join(MASK_DIR, fname)).resize((512, 512), Image.NEAREST))
        pred = predictions[fname]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title(f"Original ({fname})")
        axes[0].axis("off")

        axes[1].imshow(colorize_mask(expert))
        axes[1].set_title("Expert Mask")
        axes[1].axis("off")

        axes[2].imshow(colorize_mask(pred))
        dice = per_image_dice[fname]
        axes[2].set_title(f"Model Prediction (Dice={dice:.3f})")
        axes[2].axis("off")

        plt.suptitle(f"{label.capitalize()} Case", fontsize=14, fontweight="bold")
        save_figure(fig, os.path.join(FIGURES_DIR, f"qualitative_{label}.png"))


def main():
    print("=" * 60)
    print("EXPERIMENT 6: Downstream Validation")
    print(f"Config: {ENCODER} + {DECODER} + {LOSS} + Aug {AUGMENTATION}")
    print("=" * 60)

    split = json.load(open(SPLIT_PATH))

    # Step 1: Train/load final model
    print("\n--- Step 1: Final Model ---")
    model = train_final_model(split)

    # Step 2: Generate predictions
    print("\n--- Step 2: Generate Predictions ---")
    predictions = generate_predictions(model, split)
    print(f"  Generated {len(predictions)} predictions")

    # Compute per-image Dice
    per_image_dice = {}
    for fname in split["test_images"]:
        expert = np.array(Image.open(os.path.join(MASK_DIR, fname)).resize((512, 512), Image.NEAREST))
        pred = predictions[fname]
        pred_t = torch.from_numpy(pred).unsqueeze(0)
        expert_t = torch.from_numpy(expert).unsqueeze(0)
        metrics = compute_metrics(pred_t, expert_t)
        per_image_dice[fname] = metrics["mean_dice"]

    # Step 3: Compute morphometric metrics
    print("\n--- Step 3: Morphometric Metrics ---")
    rows = []
    for fname in split["test_images"]:
        expert_mask = np.array(Image.open(os.path.join(MASK_DIR, fname)).resize((512, 512), Image.NEAREST))
        pred_mask = predictions[fname]
        placenta_id = get_placenta_id(fname)

        expert_metrics = compute_morphometrics(expert_mask)
        model_metrics = compute_morphometrics(pred_mask)

        for metric_name in METRIC_NAMES:
            rows.append({
                "image": fname,
                "placenta_id": placenta_id,
                "metric": metric_name,
                "model_value": round(model_metrics[metric_name], 4),
                "expert_value": round(expert_metrics[metric_name], 4),
            })

    per_image_df = pd.DataFrame(rows)
    per_image_df.to_csv(os.path.join(RESULTS_DIR, "per_image_metrics.csv"), index=False)

    # Placenta-level comparison
    placenta_df = per_image_df.groupby(["placenta_id", "metric"]).agg(
        model_mean=("model_value", "mean"),
        expert_mean=("expert_value", "mean"),
    ).reset_index()
    placenta_df.to_csv(os.path.join(RESULTS_DIR, "placenta_level_comparison.csv"), index=False)

    # Step 4: Statistical agreement analysis
    print("\n--- Step 4: Agreement Analysis ---")
    agreement_rows = []
    for metric_name in METRIC_NAMES:
        mdf = per_image_df[per_image_df["metric"] == metric_name]
        model_vals = mdf["model_value"].values
        expert_vals = mdf["expert_value"].values
        image_ids = mdf["image"].values
        placenta_ids = mdf["placenta_id"].values

        agreement = compute_agreement(model_vals, expert_vals, image_ids, placenta_ids)
        agreement["metric"] = metric_name
        agreement_rows.append(agreement)

        label = METRIC_LABELS[metric_name]
        print(f"  {label}: r={agreement['pearson_r']:.3f}, ICC={agreement['icc']:.3f}, "
              f"CCC={agreement['lins_ccc']:.3f}, bias={agreement['bias']:.3f}")

    agreement_df = pd.DataFrame(agreement_rows)
    cols = ["metric", "pearson_r", "pearson_p", "icc", "icc_ci_lower", "icc_ci_upper",
            "lins_ccc", "bias", "loa_lower", "loa_upper", "paired_t_stat", "paired_t_p", "mean_pct_diff"]
    agreement_df = agreement_df[cols]
    agreement_df.to_csv(os.path.join(RESULTS_DIR, "agreement_summary.csv"), index=False)

    # Save validation summary JSON
    summary = {}
    for _, row in agreement_df.iterrows():
        summary[row["metric"]] = {k: v for k, v in row.items() if k != "metric"}
    save_json(summary, os.path.join(RESULTS_DIR, "validation_summary.json"))

    # Step 5: Figures
    print("\n--- Step 5: Figures ---")
    setup_figure_style()

    # Correlation plots (2×3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, metric_name in enumerate(METRIC_NAMES):
        ax = axes[i // 3, i % 3]
        mdf = per_image_df[per_image_df["metric"] == metric_name]
        ax.scatter(mdf["expert_value"], mdf["model_value"], alpha=0.7, edgecolors="black", linewidth=0.5)
        # Identity line
        all_vals = np.concatenate([mdf["expert_value"], mdf["model_value"]])
        lims = [min(all_vals) * 0.9, max(all_vals) * 1.1]
        ax.plot(lims, lims, "k--", alpha=0.3)
        ax.set_xlabel("Expert")
        ax.set_ylabel("Model")
        r_val = agreement_df[agreement_df["metric"] == metric_name]["pearson_r"].values[0]
        ax.set_title(f"{METRIC_LABELS[metric_name]}\nr={r_val:.3f}")
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURES_DIR, "correlation_plots.png"))

    # Bland-Altman plots (2×3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, metric_name in enumerate(METRIC_NAMES):
        ax = axes[i // 3, i % 3]
        mdf = per_image_df[per_image_df["metric"] == metric_name]
        model_vals = mdf["model_value"].values
        expert_vals = mdf["expert_value"].values
        diff = model_vals - expert_vals
        mean_both = (model_vals + expert_vals) / 2

        arow = agreement_df[agreement_df["metric"] == metric_name].iloc[0]

        ax.scatter(mean_both, diff, alpha=0.7, edgecolors="black", linewidth=0.5)
        ax.axhline(y=arow["bias"], color="red", linestyle="-", label=f"Bias={arow['bias']:.2f}")
        ax.axhline(y=arow["loa_upper"], color="blue", linestyle="--", alpha=0.5)
        ax.axhline(y=arow["loa_lower"], color="blue", linestyle="--", alpha=0.5)
        ax.set_xlabel("Mean (Model + Expert) / 2")
        ax.set_ylabel("Difference (Model - Expert)")
        ax.set_title(METRIC_LABELS[metric_name])
        ax.legend(fontsize=7)
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURES_DIR, "bland_altman_plots.png"))

    # Qualitative examples
    save_qualitative(split, predictions, per_image_dice)

    print("\n" + "=" * 60)
    print("EXPERIMENT 6 COMPLETE")
    print(f"Results saved to {RESULTS_DIR}")
    print("=" * 60)

    # Print overall summary
    mean_dice_test = np.mean(list(per_image_dice.values()))
    print(f"\nTest set mean Dice: {mean_dice_test:.4f}")
    print(f"Per-image Dice range: [{min(per_image_dice.values()):.4f}, {max(per_image_dice.values()):.4f}]")


if __name__ == "__main__":
    main()

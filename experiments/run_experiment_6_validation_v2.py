#!/usr/bin/env python3
"""Experiment 6 (v2): Downstream Validation at native resolution.

This is a revision of run_experiment_6_validation.py. The ORIGINAL script is
preserved unchanged; all v2 outputs go to results/experiment_6_validation_v2/.

What changed vs v1 (and why)
----------------------------
v1 computed every morphometric in the squashed 512x512 training frame:
  - the model prediction is natively 512x512, and
  - the expert mask (native 2048x1536) was DOWNSAMPLED to 512x512 (nearest)
    before measurement.
The source images are 2048x1536 (4:3). Resizing to 512x512 (1:1) is a
non-uniform squash (width x0.25, height x0.333). This is harmless for
area-proportion metrics (scale-invariant) but distorts every PERIMETER-based
metric in an orientation-dependent way, AND throwing the expert mask down to
512 discards the boundary detail that perimeter is most sensitive to.

v2 measures all morphometrics at NATIVE 2048x1536 resolution:
  - the 512x512 model prediction is upsampled (nearest) to 2048x1536, and
  - the expert mask is used at its native 2048x1536 (no downsampling).
At native resolution the acquisition pixels are square, so perimeter is
geometrically consistent and no anisotropic correction is required. v2 also
implements the full 6-metric spec, restoring the two %-perimeter metrics
(mbs_pct_perimeter, fbs_pct_perimeter) that v1 silently dropped.

To isolate the resolution change as the ONLY variable, v2 reuses the exact
same saved 512x512 predictions from v1 (results/.../predictions/). No
retraining and no re-inference. Class mapping is identical to v1:
FBS=0, MBS=1, Background=2 (confirmed against mask pixel frequencies).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import save_json, setup_figure_style, save_figure
from src.evaluate import compute_metrics
from src.agreement import compute_agreement
# NOTE: src.models (and its timm/smp deps) is imported lazily inside the
# regenerate fallback only — it is not needed when v1 predictions exist.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "data", "cd31_images")
MASK_DIR = os.path.join(BASE_DIR, "data", "masks")
SPLIT_PATH = os.path.join(BASE_DIR, "results", "experiment_1_data_split", "split_info.json")

V1_DIR = os.path.join(BASE_DIR, "results", "experiment_6_validation")
V1_PRED_DIR = os.path.join(V1_DIR, "predictions")

RESULTS_DIR = os.path.join(BASE_DIR, "results", "experiment_6_validation_v2")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions_native")
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints", "experiment_6", "final_model_best.pth")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

ENCODER = "tu-convnext_small"
DECODER = "Unet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Native acquisition resolution (width, height) for PIL.
NATIVE_W, NATIVE_H = 2048, 1536

# Class mapping (matches v1 / morphometrics.py / canonical CLAUDE.md).
FBS, MBS, BG = 0, 1, 2

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


def _perimeter(binary_mask):
    """Total perimeter of outer contours (RETR_EXTERNAL), in pixels."""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return float(sum(cv2.arcLength(c, closed=True) for c in contours))


def compute_morphometrics_native(mask):
    """Compute all 6 morphometric metrics from a (native-resolution) mask.

    Args:
        mask: (H, W) integer array, FBS=0, MBS=1, Background=2.

    Returns:
        dict with the 6 metrics in METRIC_NAMES.
    """
    fbs_mask = (mask == FBS).astype(np.uint8)
    mbs_mask = (mask == MBS).astype(np.uint8)

    fbs_area = int(fbs_mask.sum())
    mbs_area = int(mbs_mask.sum())
    total_area = fbs_area + mbs_area

    fbs_perim = _perimeter(fbs_mask)
    mbs_perim = _perimeter(mbs_mask)
    total_perim = fbs_perim + mbs_perim

    return {
        "mbs_pct_area": 100 * mbs_area / total_area if total_area > 0 else 0.0,
        "fbs_pct_area": 100 * fbs_area / total_area if total_area > 0 else 0.0,
        "mbs_pct_perimeter": 100 * mbs_perim / total_perim if total_perim > 0 else 0.0,
        "fbs_pct_perimeter": 100 * fbs_perim / total_perim if total_perim > 0 else 0.0,
        "fbs_mbs_area_ratio": fbs_area / mbs_area if mbs_area > 0 else 0.0,
        "fbs_pa_ratio": fbs_perim / fbs_area if fbs_area > 0 else 0.0,
    }


def get_placenta_id(fname):
    img_id = int(fname.replace(".png", ""))
    return (img_id - 1) // 5 + 1


def load_pred_512(fname):
    """Load the v1-saved 512x512 prediction if present; else regenerate it."""
    p = os.path.join(V1_PRED_DIR, fname)
    if os.path.exists(p):
        return np.array(Image.open(p))
    return None


@torch.no_grad()
def regenerate_pred_512(model, fname):
    """Regenerate a 512x512 prediction from the checkpoint (fallback only)."""
    from src.dataset import get_normalize_transform
    image = np.array(Image.open(os.path.join(IMG_DIR, fname)).convert("RGB"))
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    norm = get_normalize_transform()(image=image)["image"]
    logits = model(norm.unsqueeze(0).to(DEVICE))
    return logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)


def upsample_to_native(pred_512):
    """Nearest-neighbour upsample a 512x512 label map to native 2048x1536."""
    return np.array(
        Image.fromarray(pred_512.astype(np.uint8)).resize(
            (NATIVE_W, NATIVE_H), Image.NEAREST
        )
    )


def main():
    print("=" * 64)
    print("EXPERIMENT 6 (v2): Downstream Validation @ native 2048x1536")
    print("=" * 64)

    split = json.load(open(SPLIT_PATH))
    test_images = split["test_images"]

    # Lazy model load only if any v1 prediction is missing.
    model = None

    # ---- Build native-resolution prediction/expert pairs -----------------
    print("\n--- Loading predictions and upsampling to native ---")
    preds_native, experts_native = {}, {}
    for fname in test_images:
        pred_512 = load_pred_512(fname)
        if pred_512 is None:
            if model is None:
                print("  Some v1 predictions missing; loading checkpoint...")
                from src.models import create_model
                model = create_model(ENCODER, decoder_name=DECODER, num_classes=3,
                                     freeze_encoder=False, img_size=512).to(DEVICE)
                model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE,
                                                 weights_only=True))
                model.eval()
            pred_512 = regenerate_pred_512(model, fname)

        pred_native = upsample_to_native(pred_512)
        expert_native = np.array(Image.open(os.path.join(MASK_DIR, fname)))

        preds_native[fname] = pred_native
        experts_native[fname] = expert_native

        # Save the native prediction for the record.
        Image.fromarray(pred_native.astype(np.uint8)).save(
            os.path.join(PRED_DIR, fname)
        )

    # ---- Per-image Dice at native resolution -----------------------------
    per_image_dice = {}
    for fname in test_images:
        pred_t = torch.from_numpy(preds_native[fname]).unsqueeze(0)
        expert_t = torch.from_numpy(experts_native[fname]).unsqueeze(0)
        per_image_dice[fname] = compute_metrics(pred_t, expert_t)["mean_dice"]

    # ---- Morphometrics (6 metrics, native resolution) --------------------
    print("\n--- Computing morphometrics (6 metrics, native) ---")
    rows = []
    for fname in test_images:
        em = compute_morphometrics_native(experts_native[fname])
        mm = compute_morphometrics_native(preds_native[fname])
        pid = get_placenta_id(fname)
        for metric_name in METRIC_NAMES:
            rows.append({
                "image": fname,
                "placenta_id": pid,
                "metric": metric_name,
                "model_value": round(mm[metric_name], 4),
                "expert_value": round(em[metric_name], 4),
            })
    per_image_df = pd.DataFrame(rows)
    per_image_df.to_csv(os.path.join(RESULTS_DIR, "per_image_metrics.csv"), index=False)

    placenta_df = per_image_df.groupby(["placenta_id", "metric"]).agg(
        model_mean=("model_value", "mean"),
        expert_mean=("expert_value", "mean"),
    ).reset_index()
    placenta_df.to_csv(os.path.join(RESULTS_DIR, "placenta_level_comparison.csv"), index=False)

    # ---- Agreement analysis ----------------------------------------------
    print("\n--- Agreement analysis ---")
    agreement_rows = []
    for metric_name in METRIC_NAMES:
        mdf = per_image_df[per_image_df["metric"] == metric_name]
        agreement = compute_agreement(
            mdf["model_value"].values, mdf["expert_value"].values,
            mdf["image"].values, mdf["placenta_id"].values,
        )
        agreement["metric"] = metric_name
        agreement_rows.append(agreement)
        print(f"  {METRIC_LABELS[metric_name]:<20s}: r={agreement['pearson_r']:.3f}, "
              f"ICC={agreement['icc']:.3f}, CCC={agreement['lins_ccc']:.3f}, "
              f"bias={agreement['bias']:.3f}")

    cols = ["metric", "pearson_r", "pearson_p", "icc", "icc_ci_lower", "icc_ci_upper",
            "lins_ccc", "bias", "loa_lower", "loa_upper", "paired_t_stat",
            "paired_t_p", "mean_pct_diff"]
    agreement_df = pd.DataFrame(agreement_rows)[cols]
    agreement_df.to_csv(os.path.join(RESULTS_DIR, "agreement_summary.csv"), index=False)

    summary = {}
    for _, row in agreement_df.iterrows():
        summary[row["metric"]] = {k: v for k, v in row.items() if k != "metric"}
    summary["_meta"] = {
        "resolution": "native 2048x1536",
        "expert_mask": "native (no downsampling)",
        "model_pred": "512x512 upsampled to native via nearest-neighbour",
        "predictions_source": "reused from v1 (identical model outputs)",
        "n_metrics": len(METRIC_NAMES),
        "test_mean_dice_native": round(float(np.mean(list(per_image_dice.values()))), 4),
    }
    save_json(summary, os.path.join(RESULTS_DIR, "validation_summary.json"))

    # ---- Figures ----------------------------------------------------------
    print("\n--- Figures ---")
    setup_figure_style()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, metric_name in enumerate(METRIC_NAMES):
        ax = axes[i // 3, i % 3]
        mdf = per_image_df[per_image_df["metric"] == metric_name]
        ax.scatter(mdf["expert_value"], mdf["model_value"], alpha=0.7,
                   edgecolors="black", linewidth=0.5)
        all_vals = np.concatenate([mdf["expert_value"], mdf["model_value"]])
        lims = [min(all_vals) * 0.9, max(all_vals) * 1.1]
        ax.plot(lims, lims, "k--", alpha=0.3)
        ax.set_xlabel("Expert")
        ax.set_ylabel("Model")
        r_val = agreement_df[agreement_df["metric"] == metric_name]["pearson_r"].values[0]
        ax.set_title(f"{METRIC_LABELS[metric_name]}\nr={r_val:.3f}")
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURES_DIR, "correlation_plots.png"))

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
        ax.axhline(y=arow["bias"], color="red", linestyle="-",
                   label=f"Bias={arow['bias']:.2f}")
        ax.axhline(y=arow["loa_upper"], color="blue", linestyle="--", alpha=0.5)
        ax.axhline(y=arow["loa_lower"], color="blue", linestyle="--", alpha=0.5)
        ax.set_xlabel("Mean (Model + Expert) / 2")
        ax.set_ylabel("Difference (Model - Expert)")
        ax.set_title(METRIC_LABELS[metric_name])
        ax.legend(fontsize=7)
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURES_DIR, "bland_altman_plots.png"))

    # ---- v1 vs v2 comparison ---------------------------------------------
    print("\n--- v1 vs v2 comparison ---")
    build_comparison(agreement_df, per_image_df)

    mean_dice = np.mean(list(per_image_dice.values()))
    print("\n" + "=" * 64)
    print("EXPERIMENT 6 (v2) COMPLETE")
    print(f"Results: {RESULTS_DIR}")
    print(f"Test mean Dice (native): {mean_dice:.4f}")
    print("=" * 64)


def build_comparison(v2_agreement, v2_per_image):
    """Compare v1 (512 frame) vs v2 (native) on shared agreement metrics."""
    v1_path = os.path.join(V1_DIR, "agreement_summary.csv")
    if not os.path.exists(v1_path):
        print("  v1 agreement_summary.csv not found; skipping comparison.")
        return
    v1 = pd.read_csv(v1_path).set_index("metric")
    v2 = v2_agreement.set_index("metric")

    shared = [m for m in v1.index if m in v2.index]
    fields = ["pearson_r", "icc", "lins_ccc", "bias"]
    rows = []
    for m in shared:
        row = {"metric": m}
        for f in fields:
            row[f"{f}_v1"] = round(float(v1.loc[m, f]), 4)
            row[f"{f}_v2"] = round(float(v2.loc[m, f]), 4)
            row[f"{f}_delta"] = round(float(v2.loc[m, f]) - float(v1.loc[m, f]), 4)
        rows.append(row)
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(RESULTS_DIR, "comparison_v1_vs_v2.csv"), index=False)

    only_v2 = [m for m in v2.index if m not in v1.index]
    print(f"  Metrics in v1: {list(v1.index)}")
    print(f"  New in v2 (not measured by v1): {only_v2}")
    print("\n  Agreement: v1 (512 squashed)  ->  v2 (native 2048x1536)")
    for _, r in comp.iterrows():
        print(f"    {METRIC_LABELS.get(r['metric'], r['metric']):<20s} "
              f"r {r['pearson_r_v1']:+.3f}->{r['pearson_r_v2']:+.3f} "
              f"({r['pearson_r_delta']:+.3f})   "
              f"ICC {r['icc_v1']:+.3f}->{r['icc_v2']:+.3f} "
              f"({r['icc_delta']:+.3f})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quick-start inference: segment a CD31 image with the trained model.

Loads the final optimized model (ConvNeXt-Small + U-Net, trained with CE+Dice
and geometric augmentation) and produces, for one image or a folder of images:

  * ``<name>_mask.png``    -- the raw 3-class label map (0=FBS, 1=MBS, 2=Background)
  * ``<name>_overlay.png`` -- the input with FBS/MBS shaded for visual inspection
  * a printed morphometric summary (FBS/MBS % area, FBS:MBS ratio, FBS P:A ratio)

Example
-------
    python predict.py --image my_section.png --checkpoint final_model_best.pth
    python predict.py --image folder_of_pngs/ --checkpoint final_model_best.pth --outdir out/

The trained weights (``final_model_best.pth``, ~205 MB) are not bundled in this
repository. Download them from the Zenodo archive (DOI: <TO BE ADDED>) linked in
the README, or reproduce them by running the pipeline
(``python run_pipeline.py --only 6`` writes the checkpoint to
``checkpoints/experiment_6/final_model_best.pth``).
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import create_model
from src.dataset import get_normalize_transform
import albumentations as A

# Optimized configuration carried forward through the experiments.
ENCODER = "tu-convnext_small"
DECODER = "Unet"
INPUT_SIZE = 512

# Class index -> (name, RGB overlay colour). Background is left unshaded.
CLASSES = {
    0: ("FBS", (89, 78, 144)),    # fetal blood space   -> purple  (#594e90)
    1: ("MBS", (0, 61, 92)),      # maternal blood space-> deep blue(#003d5c)
    2: ("Background", None),
}


def load_model(checkpoint, device):
    model = create_model(ENCODER, decoder_name=DECODER, num_classes=3,
                         freeze_encoder=False, img_size=INPUT_SIZE)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


@torch.no_grad()
def predict_one(model, image_path, device):
    """Return (native_RGB array, 512x512 label map, label map upsampled to native)."""
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]

    resized = A.Resize(INPUT_SIZE, INPUT_SIZE)(image=image)["image"]
    tensor = get_normalize_transform()(image=resized)["image"].unsqueeze(0).to(device)
    pred = model(tensor).argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # Nearest-neighbour upsample the label map back to the acquisition resolution.
    pred_native = np.array(
        Image.fromarray(pred).resize((w, h), Image.NEAREST)
    )
    return image, pred, pred_native


def make_overlay(image, label_map, alpha=0.45):
    overlay = image.copy()
    for idx, (_, colour) in CLASSES.items():
        if colour is None:
            continue
        m = label_map == idx
        overlay[m] = (alpha * np.array(colour) + (1 - alpha) * image[m]).astype(np.uint8)
    return overlay


def morphometrics(label_map):
    """FBS/MBS area fractions and ratios from a label map (matches src/morphometrics.py)."""
    import cv2
    fbs = int((label_map == 0).sum())
    mbs = int((label_map == 1).sum())
    vascular = fbs + mbs
    out = {
        "FBS_pct_area": 100 * fbs / vascular if vascular else 0.0,
        "MBS_pct_area": 100 * mbs / vascular if vascular else 0.0,
        "FBS_MBS_area_ratio": fbs / mbs if mbs else float("nan"),
    }
    fbs_mask = (label_map == 0).astype(np.uint8)
    contours, _ = cv2.findContours(fbs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fbs_perim = sum(cv2.arcLength(c, True) for c in contours)
    out["FBS_perimeter_to_area"] = fbs_perim / fbs if fbs else float("nan")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="Input PNG/JPG image or a folder of images.")
    p.add_argument("--checkpoint", required=True, help="Path to final_model_best.pth.")
    p.add_argument("--outdir", default="predictions", help="Where to write masks/overlays.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}\n"
                 f"See this script's header for how to obtain or reproduce the weights.")

    if os.path.isdir(args.image):
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        images = sorted(os.path.join(args.image, f) for f in os.listdir(args.image)
                        if f.lower().endswith(exts))
    else:
        images = [args.image]
    if not images:
        sys.exit(f"No images found at: {args.image}")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Loading model on {args.device} ...")
    model = load_model(args.checkpoint, args.device)

    for path in images:
        image, _, label_native = predict_one(model, path, args.device)
        stem = os.path.splitext(os.path.basename(path))[0]
        Image.fromarray(label_native).save(os.path.join(args.outdir, f"{stem}_mask.png"))
        Image.fromarray(make_overlay(image, label_native)).save(
            os.path.join(args.outdir, f"{stem}_overlay.png"))

        m = morphometrics(label_native)
        print(f"\n{os.path.basename(path)}")
        print(f"  FBS area: {m['FBS_pct_area']:.1f}%   MBS area: {m['MBS_pct_area']:.1f}%"
              f"   FBS:MBS = {m['FBS_MBS_area_ratio']:.3f}")
        print(f"  FBS perimeter-to-area: {m['FBS_perimeter_to_area']:.4f}")

    print(f"\nWrote masks and overlays to {args.outdir}/")


if __name__ == "__main__":
    main()

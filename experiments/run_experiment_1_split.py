#!/usr/bin/env python3
"""Experiment 1: Data Discovery and Splitting.

Steps:
1. List all files, identify naming convention, extract placenta IDs
2. Confirm 100 image-mask pairs from 20 placentas (5 each)
3. Verify mask class values (0, 1, 2)
4. Print summary and save to results/data_summary.json
5. Split at placenta level: 5 test, 2 val, 13 train (seed=42)
6. Save split to results/experiment_1_data_split/split_info.json
"""

import os
import sys
import json
import random
import numpy as np
from PIL import Image
from collections import defaultdict

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "data", "cd31_images")
MASK_DIR = os.path.join(BASE_DIR, "data", "masks")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SPLIT_DIR = os.path.join(RESULTS_DIR, "experiment_1_data_split")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)


def discover_data():
    """Discover data structure and return summary."""
    # List files
    img_files = sorted(
        [f for f in os.listdir(IMG_DIR) if f.endswith(".png")],
        key=lambda x: int(x.replace(".png", "")),
    )
    mask_files = sorted(
        [f for f in os.listdir(MASK_DIR) if f.endswith(".png")],
        key=lambda x: int(x.replace(".png", "")),
    )

    assert len(img_files) == 100, f"Expected 100 images, got {len(img_files)}"
    assert len(mask_files) == 100, f"Expected 100 masks, got {len(mask_files)}"
    assert img_files == mask_files, "Image and mask filenames do not match"

    # Assign placenta IDs: images 1-5 -> placenta 1, 6-10 -> placenta 2, etc.
    placenta_map = {}
    placentas = defaultdict(list)
    for f in img_files:
        img_id = int(f.replace(".png", ""))
        placenta_id = (img_id - 1) // 5 + 1
        placenta_map[f] = placenta_id
        placentas[placenta_id].append(f)

    assert len(placentas) == 20, f"Expected 20 placentas, got {len(placentas)}"
    for pid, files in placentas.items():
        assert len(files) == 5, f"Placenta {pid} has {len(files)} images, expected 5"

    # Check image dimensions
    sample_img = Image.open(os.path.join(IMG_DIR, img_files[0]))
    img_width, img_height = sample_img.size
    img_mode = sample_img.mode

    # Check mask class values and distribution
    all_unique = set()
    class_pixel_counts = defaultdict(int)
    for f in mask_files:
        mask = np.array(Image.open(os.path.join(MASK_DIR, f)))
        unique = np.unique(mask)
        all_unique.update(unique.tolist())
        for cls in unique:
            class_pixel_counts[int(cls)] += int(np.sum(mask == cls))

    total_pixels = sum(class_pixel_counts.values())
    class_distribution = {
        int(cls): {
            "pixel_count": int(cnt),
            "percentage": round(100 * cnt / total_pixels, 2),
        }
        for cls, cnt in sorted(class_pixel_counts.items())
    }

    # Class mapping
    class_names = {0: "FBS", 1: "MBS", 2: "Background"}

    summary = {
        "num_images": len(img_files),
        "num_masks": len(mask_files),
        "num_placentas": len(placentas),
        "images_per_placenta": 5,
        "image_dimensions": {"width": img_width, "height": img_height},
        "image_mode": img_mode,
        "mask_class_values": sorted(all_unique),
        "class_names": class_names,
        "class_distribution": class_distribution,
        "placenta_ids": sorted(placentas.keys()),
        "placenta_to_images": {
            int(pid): files for pid, files in sorted(placentas.items())
        },
        "naming_convention": "Sequential integers 1-100; images (pid-1)*5+1 to pid*5 belong to placenta pid",
    }

    return summary, placenta_map, placentas


def create_split(placentas, seed=42):
    """Create placenta-level train/val/test split."""
    rng = random.Random(seed)

    all_pids = sorted(placentas.keys())
    shuffled = all_pids.copy()
    rng.shuffle(shuffled)

    # 5 test, 2 val, 13 train
    test_placentas = sorted(shuffled[:5])
    val_placentas = sorted(shuffled[5:7])
    train_placentas = sorted(shuffled[7:])

    assert len(test_placentas) == 5
    assert len(val_placentas) == 2
    assert len(train_placentas) == 13

    # Get image lists
    test_images = sorted(
        [f for pid in test_placentas for f in placentas[pid]],
        key=lambda x: int(x.replace(".png", "")),
    )
    val_images = sorted(
        [f for pid in val_placentas for f in placentas[pid]],
        key=lambda x: int(x.replace(".png", "")),
    )
    train_images = sorted(
        [f for pid in train_placentas for f in placentas[pid]],
        key=lambda x: int(x.replace(".png", "")),
    )

    split_info = {
        "seed": seed,
        "test_placentas": test_placentas,
        "val_placentas": val_placentas,
        "train_placentas": train_placentas,
        "test_images": test_images,
        "val_images": val_images,
        "train_images": train_images,
        "num_test": len(test_images),
        "num_val": len(val_images),
        "num_train": len(train_images),
    }

    return split_info


def main():
    print("=" * 60)
    print("EXPERIMENT 1: Data Discovery and Splitting")
    print("=" * 60)

    # Step 1-5: Data discovery
    print("\n--- Data Discovery ---")
    summary, placenta_map, placentas = discover_data()

    print(f"Images: {summary['num_images']}")
    print(f"Masks: {summary['num_masks']}")
    print(f"Placentas: {summary['num_placentas']}")
    print(f"Images per placenta: {summary['images_per_placenta']}")
    print(f"Image dimensions: {summary['image_dimensions']}")
    print(f"Image mode: {summary['image_mode']}")
    print(f"Mask class values: {summary['mask_class_values']}")
    print(f"\nClass distribution:")
    for cls, info in summary["class_distribution"].items():
        name = summary["class_names"].get(int(cls), "Unknown")
        print(f"  Class {cls} ({name}): {info['percentage']:.1f}%")

    print(f"\nPlacenta IDs: {summary['placenta_ids']}")
    print(f"Naming convention: {summary['naming_convention']}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "data_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved data summary to {summary_path}")

    # Step 6: Create split
    print("\n--- Data Splitting ---")
    split_info = create_split(placentas, seed=42)

    print(f"Test placentas ({len(split_info['test_placentas'])}): {split_info['test_placentas']}")
    print(f"Val placentas ({len(split_info['val_placentas'])}): {split_info['val_placentas']}")
    print(f"Train placentas ({len(split_info['train_placentas'])}): {split_info['train_placentas']}")
    print(f"\nTest images: {split_info['num_test']}")
    print(f"Val images: {split_info['num_val']}")
    print(f"Train images: {split_info['num_train']}")

    # Verify no overlap
    test_set = set(split_info["test_images"])
    val_set = set(split_info["val_images"])
    train_set = set(split_info["train_images"])
    assert len(test_set & val_set) == 0, "Test/val overlap!"
    assert len(test_set & train_set) == 0, "Test/train overlap!"
    assert len(val_set & train_set) == 0, "Val/train overlap!"
    assert len(test_set | val_set | train_set) == 100, "Missing images!"
    print("\nNo overlap between splits. All 100 images accounted for.")

    # Save split
    split_path = os.path.join(SPLIT_DIR, "split_info.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved split to {split_path}")

    print("\n" + "=" * 60)
    print("Experiment 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

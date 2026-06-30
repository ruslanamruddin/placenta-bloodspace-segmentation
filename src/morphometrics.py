"""Morphometric metrics and contour extraction for downstream validation."""

import numpy as np
import cv2


def smooth_mask(mask, kernel_size=7):
    """Smooth a multi-class segmentation mask using morphological close-open.

    Applies per-class elliptical close then open to reduce boundary jaggedness,
    then resolves overlaps by assigning each pixel to the class with highest
    smoothed confidence.

    Args:
        mask: (H, W) array with class indices (0=FBS, 1=MBS, 2=Background)
        kernel_size: diameter of elliptical structuring element (default 7)

    Returns:
        Smoothed mask with same shape and class indices.
    """
    h, w = mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    scores = np.zeros((3, h, w), dtype=np.float32)
    for cls in range(3):
        binary = (mask == cls).astype(np.uint8)
        smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
        scores[cls] = smoothed.astype(np.float32)
    return np.argmax(scores, axis=0).astype(np.uint8)


def compute_morphometrics(mask, num_classes=3):
    """Compute 4 morphometric metrics from a segmentation mask.

    Args:
        mask: (H, W) array with class indices (0=FBS, 1=MBS, 2=Background)

    Returns:
        dict with 4 metrics: area proportions, area ratio, P:A ratio
    """
    fbs_mask = (mask == 0).astype(np.uint8)
    mbs_mask = (mask == 1).astype(np.uint8)

    fbs_pixels = fbs_mask.sum()
    mbs_pixels = mbs_mask.sum()

    # Perimeter via contours (RETR_EXTERNAL = outer contours only)
    fbs_perimeter = _compute_perimeter(fbs_mask)

    # Avoid division by zero
    total_area = fbs_pixels + mbs_pixels

    if total_area == 0:
        return {
            "mbs_pct_area": 0.0,
            "fbs_pct_area": 0.0,
            "fbs_mbs_area_ratio": 0.0,
            "fbs_pa_ratio": 0.0,
        }

    metrics = {
        "mbs_pct_area": 100 * mbs_pixels / total_area if total_area > 0 else 0.0,
        "fbs_pct_area": 100 * fbs_pixels / total_area if total_area > 0 else 0.0,
        "fbs_mbs_area_ratio": fbs_pixels / mbs_pixels if mbs_pixels > 0 else 0.0,
        "fbs_pa_ratio": fbs_perimeter / fbs_pixels if fbs_pixels > 0 else 0.0,
    }

    return metrics


def _compute_perimeter(binary_mask):
    """Compute total perimeter of outer contours."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return sum(cv2.arcLength(c, closed=True) for c in contours)

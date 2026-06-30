"""Data augmentation strategies A-G using Albumentations."""

import albumentations as A


def get_augmentation(strategy):
    """Return augmentation pipeline by strategy name (A-G)."""
    strategies = {
        "A": strategy_a,
        "B": strategy_b,
        "C": strategy_c,
        "D": strategy_d,
        "E": strategy_e,
        "F": strategy_f,
        "G": strategy_g,
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")
    return strategies[strategy]()


def strategy_a():
    """Baseline: HFlip, VFlip, Rotate90."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])


def strategy_b():
    """Geometric + Color."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(25, 51),
                        hole_width_range=(25, 51), p=0.3),
    ])


def strategy_c():
    """Elastic (B minus dropout + elastic/grid)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.ElasticTransform(alpha=120, sigma=10, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
    ])


def strategy_d():
    """Stain-specific color."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.3),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
    ])


def strategy_e():
    """D + morphological (brightness-contrast + erosion/dilation)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.3),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.Morphological(scale=(2, 4), operation="erosion", p=0.2),
        A.Morphological(scale=(2, 4), operation="dilation", p=0.2),
    ])


def strategy_f():
    """B minus dropout (color-only, no distortion)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    ])


def strategy_g():
    """Kitchen sink (C + D + brightness-contrast)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.ElasticTransform(alpha=120, sigma=10, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.3),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
    ])

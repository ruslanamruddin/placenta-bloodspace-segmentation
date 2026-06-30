"""PyTorch Dataset for CD31-stained placental histology."""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_normalize_transform():
    """Return normalization + tensor conversion transforms."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class PlacentaDataset(Dataset):
    """Dataset for placental segmentation.

    Args:
        image_dir: Path to CD31 images.
        mask_dir: Path to mask images.
        image_files: List of filenames to include.
        augmentation: Albumentations Compose for augmentation (applied before normalization).
        input_size: Resize target (height, width).
    """

    def __init__(self, image_dir, mask_dir, image_files, augmentation=None, input_size=512):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(image_files, key=lambda x: int(x.replace(".png", "")))
        self.augmentation = augmentation
        self.input_size = input_size
        self.resize = A.Resize(input_size, input_size)
        self.normalize = get_normalize_transform()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]

        # Load image and mask
        image = np.array(Image.open(os.path.join(self.image_dir, fname)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, fname)))

        # Resize to input_size x input_size
        resized = self.resize(image=image, mask=mask)
        image = resized["image"]
        mask = resized["mask"]

        # Apply augmentation
        if self.augmentation is not None:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Normalize and convert to tensor
        normalized = self.normalize(image=image)
        image = normalized["image"]  # (3, H, W) float32
        mask = torch.from_numpy(mask).long()  # (H, W) int64

        return image, mask, fname

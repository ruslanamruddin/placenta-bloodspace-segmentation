"""Evaluation metrics: Dice, IoU, precision, recall, inference timing."""

import time
import numpy as np
import torch
import torch.nn.functional as F


def compute_metrics(preds, targets, num_classes=3):
    """Compute per-class Dice, IoU, precision, recall.

    Args:
        preds: (N, H, W) predicted class indices
        targets: (N, H, W) ground truth class indices
        num_classes: number of classes

    Returns:
        dict with per-class and mean metrics
    """
    metrics = {}
    dices, ious, precisions, recalls = [], [], [], []

    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()

        tp = (pred_c * target_c).sum().item()
        fp = (pred_c * (1 - target_c)).sum().item()
        fn = ((1 - pred_c) * target_c).sum().item()

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        class_name = ["fbs", "mbs", "bg"][c]
        metrics[f"{class_name}_dice"] = round(dice, 4)
        metrics[f"{class_name}_iou"] = round(iou, 4)
        metrics[f"{class_name}_precision"] = round(precision, 4)
        metrics[f"{class_name}_recall"] = round(recall, 4)

        dices.append(dice)
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)

    metrics["mean_dice"] = round(np.mean(dices), 4)
    metrics["mean_iou"] = round(np.mean(ious), 4)
    metrics["mean_precision"] = round(np.mean(precisions), 4)
    metrics["mean_recall"] = round(np.mean(recalls), 4)

    return metrics


@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes=3):
    """Evaluate model on a dataloader. Returns metrics dict."""
    model.eval()
    all_preds = []
    all_targets = []

    for images, masks, _ in dataloader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(masks)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return compute_metrics(all_preds, all_targets, num_classes)


@torch.no_grad()
def measure_inference_time(model, dataloader, device, num_warmup=3):
    """Measure average inference time per tile in milliseconds."""
    model.eval()
    times = []

    for i, (images, _, _) in enumerate(dataloader):
        images = images.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) * 1000  # ms
        per_tile = elapsed / images.shape[0]

        if i >= num_warmup:
            times.append(per_tile)

    return round(np.mean(times), 2) if times else 0.0


@torch.no_grad()
def measure_gpu_memory(model, input_size=512, device=None):
    """Measure GPU memory usage during inference (MB)."""
    if device is None:
        device = next(model.parameters()).device
    if device.type != "cuda":
        return 0.0

    torch.cuda.reset_peak_memory_stats(device)
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    _ = model(dummy)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return round(peak_mb, 1)

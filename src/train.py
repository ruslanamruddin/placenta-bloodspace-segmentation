"""Training loop: AdamW, cosine annealing, warmup, AMP, gradient clipping, checkpointing."""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from src.evaluate import evaluate_model, measure_inference_time, measure_gpu_memory
from src.utils import set_seed


def train_model(config):
    """Train a segmentation model.

    Args:
        config: dict with keys:
            - model: nn.Module
            - train_dataset: Dataset
            - val_dataset: Dataset
            - test_dataset: Dataset (optional, for final evaluation)
            - loss_fn: nn.Module
            - seed: int
            - max_epochs: int (default 100)
            - batch_size: int (default 8)
            - lr: float (default 1e-4)
            - weight_decay: float (default 1e-4)
            - warmup_epochs: int (default 5)
            - checkpoint_dir: str
            - run_name: str
            - device: torch.device

    Returns:
        dict with training results and metrics
    """
    # Unpack config
    model = config["model"]
    train_dataset = config["train_dataset"]
    val_dataset = config["val_dataset"]
    test_dataset = config.get("test_dataset")
    loss_fn = config["loss_fn"]
    seed = config.get("seed", 42)
    max_epochs = config.get("max_epochs", 100)
    batch_size = config.get("batch_size", 8)
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    warmup_epochs = config.get("warmup_epochs", 5)
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    run_name = config.get("run_name", "run")
    device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    set_seed(seed)

    model = model.to(device)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Optimizer — only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # LR schedule: linear warmup then cosine annealing
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (0.1 + 0.9 * epoch / warmup_epochs)
        else:
            import math
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    scaler = GradScaler("cuda")

    # Training loop
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_dice = -1
    best_epoch = -1
    history = {"train_loss": [], "val_dice": [], "lr": []}

    start_time = time.time()

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            with autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        # Validate
        val_metrics = evaluate_model(model, val_loader, device)
        val_dice = val_metrics["mean_dice"]

        history["train_loss"].append(round(avg_loss, 4))
        history["val_dice"].append(round(val_dice, 4))
        history["lr"].append(round(current_lr, 6))

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_best.pth")
            torch.save(model.state_dict(), ckpt_path)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{max_epochs} | loss={avg_loss:.4f} | "
                  f"val_dice={val_dice:.4f} | best={best_val_dice:.4f}@{best_epoch+1} | "
                  f"lr={current_lr:.6f}")

    training_time = (time.time() - start_time) / 60  # minutes

    # Load best model for evaluation
    ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_best.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    # Evaluate on validation set with best model
    val_metrics = evaluate_model(model, val_loader, device)

    # Evaluate on test set if provided
    test_metrics = {}
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)
        test_metrics = evaluate_model(model, test_loader, device)
        inference_ms = measure_inference_time(model, test_loader, device)
    else:
        inference_ms = measure_inference_time(model, val_loader, device)

    # GPU memory
    gpu_mb = measure_gpu_memory(model, input_size=train_dataset.input_size, device=device)

    # Parameter counts
    from src.models import count_parameters
    total_params, trainable_params_m = count_parameters(model)

    results = {
        "run_name": run_name,
        "seed": seed,
        "best_epoch": best_epoch + 1,
        "best_val_dice": round(best_val_dice, 4),
        "training_min": round(training_time, 1),
        "total_params_M": total_params,
        "trainable_params_M": trainable_params_m,
        "inference_ms": inference_ms,
        "gpu_memory_MB": gpu_mb,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }

    # Save run results
    results_path = os.path.join(checkpoint_dir, f"{run_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results

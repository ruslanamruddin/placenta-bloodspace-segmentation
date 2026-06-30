"""Loss functions for segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dims)
        cardinality = (probs + targets_oh).sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    """Cross-entropy + Dice, equally weighted."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)


class FocalLoss(nn.Module):
    """Focal loss with gamma."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class FocalDiceLoss(nn.Module):
    """Focal + Dice, equally weighted."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(gamma)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.focal(logits, targets) + 0.5 * self.dice(logits, targets)


class LovaszSoftmax(nn.Module):
    """Lovász-Softmax loss for multi-class segmentation."""

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        losses = []
        for c in range(num_classes):
            fg = (targets == c).float()
            errors = (fg - probs[:, c]).abs()
            errors_sorted, perm = torch.sort(errors.view(-1), descending=True)
            fg_sorted = fg.view(-1)[perm]
            grad = _lovasz_grad(fg_sorted)
            losses.append(torch.dot(F.relu(errors_sorted), grad))
        return sum(losses) / num_classes


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


class TverskyLoss(nn.Module):
    """Tversky loss with configurable alpha/beta."""

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        tp = (probs * targets_oh).sum(dims)
        fp = (probs * (1 - targets_oh)).sum(dims)
        fn = ((1 - probs) * targets_oh).sum(dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class UnifiedFocalLoss(nn.Module):
    """Unified focal loss: focal weighting + Tversky."""

    def __init__(self, gamma=2.0, alpha=0.5, beta=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        tp = (probs * targets_oh).sum(dims)
        fp = (probs * (1 - targets_oh)).sum(dims)
        fn = ((1 - probs) * targets_oh).sum(dims)

        tversky = (tp + 1.0) / (tp + self.alpha * fp + self.beta * fn + 1.0)
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky.mean()


class ComboLoss(nn.Module):
    """Combo loss: CE + Dice + Focal, equally weighted."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(gamma)

    def forward(self, logits, targets):
        return (self.ce(logits, targets) + self.dice(logits, targets) + self.focal(logits, targets)) / 3.0


# ---------- Loss factory ----------

LOSS_REGISTRY = {
    "ce": lambda: nn.CrossEntropyLoss(),
    "dice": lambda: DiceLoss(),
    "ce_dice": lambda: CEDiceLoss(),
    "focal_dice": lambda: FocalDiceLoss(gamma=2.0),
    "lovasz": lambda: LovaszSoftmax(),
    "tversky_recall": lambda: TverskyLoss(alpha=0.3, beta=0.7),
    "tversky_precision": lambda: TverskyLoss(alpha=0.7, beta=0.3),
    "unified_focal": lambda: UnifiedFocalLoss(gamma=2.0, alpha=0.5, beta=0.5),
    "combo": lambda: ComboLoss(gamma=2.0),
}


def create_loss(name):
    """Create loss function by name."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]()

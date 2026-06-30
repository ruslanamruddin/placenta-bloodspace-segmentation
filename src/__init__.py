"""Source package for the placental blood-space segmentation pipeline.

Modules
-------
dataset       : PyTorch Dataset (resize, ImageNet normalization, augmentation hook)
models        : encoder/decoder model factory (CNN, ConvNeXt, histology ViT adapters)
losses        : the nine loss functions compared in Experiment 3
augmentations : the seven augmentation strategies (A-G) compared in Experiment 4
train         : single reusable training loop (AdamW, cosine, warmup, AMP, clipping)
evaluate      : segmentation metrics (Dice, IoU, precision, recall, timing)
morphometrics : downstream morphometric measurements + contour extraction
agreement     : Pearson, ICC, Lin's CCC, Bland-Altman, paired t-test
utils         : seeding, JSON/IO, figure style helpers
"""

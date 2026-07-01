# Deep-learning segmentation of fetal and maternal blood spaces in rat placental histology

Reproducibility code for the paper:

> **Deep Learning Semantic Segmentation of Fetal and Maternal Blood Spaces in Rat Placental Histology.**
> Ruslan Amruddin, Naween Sarwari, Savvy Liu, Bryony V. Natale, David R. C. Natale.

This repository contains **all of the code and experimental setup** used to
develop, optimize, and validate a semantic-segmentation pipeline that separates
**fetal blood space (FBS)** and **maternal blood space (MBS)** in
CD31-immunostained embryonic day 19.5 rat placental sections. It is provided so
that reviewers and readers can inspect exactly how every result in the paper was
produced and reproduce the study end to end.

It is **code and setup only.** The histology images, ground-truth masks, trained
model weights, and result files are not included here. Plug the data into `data/`
(see [`data/README.md`](data/README.md)) and the pipeline reproduces everything,
including the trained model weights.

---

## What the pipeline does

Segmentation is a three-class, pixel-wise problem: **FBS (0)**, **MBS (1)**, and
**non-vascular background (2)**. The study is a sequence of controlled
experiments, each isolating one design choice and **carrying forward the best
configuration** from the previous stage. Stages communicate through files in
`results/` (the immutable data split, `best_encoder.json`, `best_loss.json`, …),
so the whole thing runs as one chained pipeline with no manual hand-off.

| Stage | What it compares | Selected in the paper |
|-------|------------------|-----------------------|
| **1 — Data split** | placenta-level train/val/test split (13 / 2 / 5 placentas, seed 42) | leakage-free split, fixed for all later stages |
| **2a — Encoder** | 14 encoders (ResNet, EfficientNet(V2), ConvNeXt, histology ViTs) | **ConvNeXt-Small** |
| **2b — Decoder** | U-Net, U-Net++, MAnet, FPN, DeepLabV3+ | **U-Net** |
| **3 — Loss** | 9 losses (CE, Dice, Tversky, Lovász, Focal, …) | **CE + Dice** |
| **4 — Augmentation** | 7 strategies (geometric → "kitchen sink") | **geometric (baseline)** |
| **5 — Data efficiency** | training on 1–13 placentas | plateau at **5–7 placentas** |
| **6 — Validation** | morphometric agreement vs. expert annotations on the held-out test set | mean Dice **0.821** |
| **6v2 — Validation (native)** | morphometrics recomputed at native 2048×1536 resolution | corrected perimeter metrics |

**Final model:** ConvNeXt-Small + U-Net + (CE + Dice) + geometric augmentation.

---

## Repository layout

```
.
├── run_pipeline.py          # ONE command to run the whole study (stages 1 → 6v2)
├── predict.py               # quick-start: segment your own image with the trained model
├── requirements.txt
├── src/                     # reusable modules
│   ├── dataset.py           #   resize + ImageNet normalization + augmentation hook
│   ├── models.py            #   encoder/decoder factory (+ ViT→U-Net adapter)
│   ├── losses.py            #   the 9 loss functions
│   ├── augmentations.py     #   the 7 augmentation strategies (A–G)
│   ├── train.py             #   single training loop (AdamW, cosine, warmup, AMP, clipping)
│   ├── evaluate.py          #   Dice / IoU / precision / recall / timing
│   ├── morphometrics.py     #   area & perimeter measurements + contour extraction
│   ├── agreement.py         #   Pearson, ICC, Lin's CCC, Bland–Altman, paired t-test
│   └── utils.py             #   seeding, I/O, figure style
├── experiments/             # one runnable script per stage (called by run_pipeline.py)
│   └── run_experiment_*.py
└── data/                    # (empty) expected data layout — see data/README.md
```

`checkpoints/`, `results/`, and `predictions/` are created at run time and are
intentionally git-ignored.

---

## Installation

Python 3.10+ and a CUDA GPU are recommended (the training stages 2a–6 are
GPU-bound; the study used a single NVIDIA A40).

```bash
git clone <this-repo-url>
cd placenta-bloodspace-segmentation

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The histology-pretrained ViT encoders in stage 2a (Phikon-v2, UNI, CONCH,
H-optimus-0, Virchow2) are optional. To include them, also install the extras
and authenticate with HuggingFace (some weights are gated):

```bash
pip install "transformers>=4.40" "huggingface-hub>=0.24"
huggingface-cli login
```

If a model is unavailable the encoder comparison simply skips it; the rest of
the pipeline and `predict.py` do not need these packages.

---

## Reproduce the study (one command)

1. Populate `data/cd31_images/` and `data/masks/` as described in
   [`data/README.md`](data/README.md).
2. Run the pipeline:

```bash
python run_pipeline.py            # stages 1 → 6v2, in order
```

Each stage writes its CSVs, figures, and a `best_*.json` decision file under
`results/`, and the next stage reads them automatically. Useful variants:

```bash
python run_pipeline.py --list     # show all stages
python run_pipeline.py --only 1   # just discover the data and create the split (seconds)
python run_pipeline.py --from 3   # resume from the loss experiment onward
python run_pipeline.py --dry-run  # print the commands without running them
```

> **Compute note.** Stage 1 runs in seconds. The training stages (2a–6) involve
> ~120 training runs across three seeds and take several GPU-hours in total;
> stage 6v2 reuses stage 6's predictions and is fast.

---

## Trained model weights

The trained weights (`final_model_best.pth`, ConvNeXt-Small + U-Net, ~205 MB) are
**not stored in this repository**. Reproduce them locally with:

```bash
python run_pipeline.py --only 6
```

which trains the final model and writes the checkpoint to
`checkpoints/experiment_6/final_model_best.pth`.

## Quick start: segment your own image

With the trained weights downloaded (see above):

```bash
python predict.py --image my_section.png --checkpoint final_model_best.pth
```

For each input this writes a label-map PNG and a colour overlay to
`predictions/`, and prints the FBS/MBS area fractions, the FBS:MBS ratio, and the
FBS perimeter-to-area ratio. Pass a folder to `--image` to batch a directory.

---

## Reproducibility notes

- **Placenta-level splitting is inviolable** — all images from one placenta stay
  in the same partition (train/val/test), preventing the within-sample leakage
  that inflates image-level splits.
- **Seeds 42, 123, 456** are set across `random`, `numpy`, and `torch`; reported
  results are mean ± SD over the three seeds.
- **The test set is touched only in stage 6.** It is never used for training or
  model selection.
- Code was developed with AI-coding assistance (Claude Code, Anthropic) and
  reviewed, tested, and validated by the authors, who take full responsibility
  for its correctness.

## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)).

## License

Code released under the [MIT License](LICENSE). The histology data are subject to
the terms stated in the paper.

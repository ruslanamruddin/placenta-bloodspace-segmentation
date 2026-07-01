# Placental blood-space segmentation

Code for the paper:

> **Deep Learning Semantic Segmentation of Fetal and Maternal Blood Spaces in Rat Placental Histology.**
> Ruslan Amruddin, Naween Sarwari, Savvy Liu, Bryony V. Natale, David R. C. Natale.

A pipeline that segments **fetal (FBS)** and **maternal (MBS)** blood spaces in
CD31-stained E19.5 rat placental images. This repo is **code only** — no data,
weights, or results. Add your data and the pipeline reproduces the whole study.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+; a CUDA GPU is recommended for training.

## Run the whole study

1. Put your images in `data/cd31_images/` and masks in `data/masks/`
   (format: see [`data/README.md`](data/README.md)).
2. Run the pipeline:

```bash
python run_pipeline.py
```

This runs all stages in order, each feeding the next. The final model
(ConvNeXt-Small + U-Net) reaches a mean Dice of **0.821**.

| Stage | What it does |
|-------|--------------|
| 1 | Split data by placenta (train / val / test) |
| 2 | Pick the best encoder and decoder |
| 3 | Pick the best loss function |
| 4 | Pick the best augmentation |
| 5 | Measure how much data is needed |
| 6 | Validate against expert annotations |

Handy options: `--list` (show stages), `--only 1` (run one stage),
`--from 3` (resume from a stage).

## Segment an image

The trained weights aren't stored here — create them with
`python run_pipeline.py --only 6` (writes `checkpoints/experiment_6/final_model_best.pth`),
then:

```bash
python predict.py --image my_section.png --checkpoint final_model_best.pth
```

This saves a mask and overlay and prints the FBS/MBS measurements.

## What's inside

```
run_pipeline.py    # run the full study
predict.py         # segment your own image
src/               # dataset, models, losses, augmentations, training, metrics
experiments/       # one script per stage
data/              # where your images and masks go
```

## Notes

- Data is split **by placenta**, so images from one placenta never span
  train and test.
- Everything uses seeds 42, 123, 456; results are reported as mean ± SD.
- Code released under the [MIT License](LICENSE); please cite the paper
  (see [`CITATION.cff`](CITATION.cff)).

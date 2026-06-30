# Data layout

The image and mask data are **not** included in this repository. This folder
documents the exact layout and format the pipeline expects, so that the study
can be reproduced either with the archived dataset (see the paper's *Data and
materials availability* statement) or with your own CD31-stained sections
prepared in the same way.

```
data/
├── cd31_images/    # input RGB images, named 1.png ... N.png
└── masks/          # ground-truth label maps, same filenames as the images
```

## Image requirements

| Property        | Value                                                          |
|-----------------|----------------------------------------------------------------|
| Format          | PNG, RGB                                                        |
| Naming          | Integer filenames `1.png`, `2.png`, ... (one extension, `.png`)|
| Acquisition size| 2048 × 1536 px in the study (any size works; resized to 512²)  |
| Grouping        | 5 consecutive images per placenta (1–5 = placenta 1, 6–10 = placenta 2, …) |

The placenta grouping is derived purely from the filename:
`placenta_id = (image_number - 1) // 5 + 1`. This is what enables the
**placenta-level** train/validation/test split that prevents information leakage
between images of the same placenta. To use a different number of images per
placenta, adjust the grouping logic in `experiments/run_experiment_1_split.py`
and `src/` accordingly.

## Mask requirements

* Single-channel PNG, same filename as the corresponding image.
* Integer class labels per pixel:

  | Value | Class                  |
  |-------|------------------------|
  | `0`   | FBS (fetal blood space)|
  | `1`   | MBS (maternal blood space)|
  | `2`   | Background (non-vascular tissue) |

Confirm the mapping on your own masks with `numpy.unique(mask)` before training —
`run_experiment_1_split.py` reports the class distribution it finds. In the study
dataset the classes are imbalanced (~9.9% FBS, ~25.7% MBS, ~64.4% background).

## Sanity check

Once `cd31_images/` and `masks/` are populated, run:

```bash
python run_pipeline.py --only 1
```

This discovers the data, verifies the image/mask pairing and class values, and
writes the immutable split to `results/experiment_1_data_split/split_info.json`.

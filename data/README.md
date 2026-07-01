# Data folder

The data isn't included in this repo. Put your own here in this layout:

```
data/
├── cd31_images/    # input images: 1.png, 2.png, ...
└── masks/          # matching masks: same filenames
```

**Images:** RGB PNGs named with numbers (`1.png`, `2.png`, ...). Any size works
(they're resized to 512×512). Images are grouped 5-per-placenta by filename
(1–5 = placenta 1, 6–10 = placenta 2, ...), which is how the pipeline keeps each
placenta in a single train/val/test split.

**Masks:** one PNG per image (same filename), with pixel values:

| Value | Class |
|-------|-------|
| 0 | FBS (fetal blood space) |
| 1 | MBS (maternal blood space) |
| 2 | Background |

**Check it's set up right:**

```bash
python run_pipeline.py --only 1
```

This verifies the images and masks match and creates the data split.

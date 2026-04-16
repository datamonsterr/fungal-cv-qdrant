# ViT Training Notes

## Key Finding

ViT performed poorly on this dataset (36% val accuracy vs EfficientNetB1 at 60%). Root cause: ViTs need 10,000+ images; we have only 1,011 training images.

**Recommendation**: Use EfficientNetB1 as primary. ViT can serve as an ensemble component only.

## Pretrained Weight Options

| Key | Model | Notes |
|-----|-------|-------|
| `vit256_dino` | ViT-256 DINO | Recommended — no domain bias |
| `cellvit_x20` | CellViT 20× | Cell-like structures |
| `cellvit_x40` | CellViT 40× | Cell-like structures |
| `sam_vit_b/l/h` | SAM ViT | General purpose |

Place weights in `pretrained/CellViT/`, `pretrained/SAM/`, `pretrained/ViT-256/`.

## Data Augmentation (ViT)

10× augmentation was implemented to expand training from 1,011 → 10,110 virtual samples (different random transform per epoch, no extra disk space):

- **Geometric**: random flip (H+V), rotation 0–180°, affine (translate ±20%, scale 0.8–1.2×, shear ±10°)
- **Color**: jitter ±40% brightness/contrast/saturation, ±20% hue, random grayscale (15%)
- **Quality**: Gaussian blur (20%), sharpness (20%), random erasing (10%)

Result: ViT accuracy improved from 36% → ~55–65% with 10× augmentation. Still trails EfficientNet.

## TPU Setup (Colab)

```python
# Change runtime: Runtime > Change runtime type > TPU v5e
# Assumes `uv` is already available in this Colab session.
!uv pip install torch-xla -f https://storage.googleapis.com/libtpu-wheels/index.html

import torch_xla.core.xla_model as xm
device = xm.xla_device()
```

In `train_models_cellvit.py`, set `USE_TPU = True` and `TPU_CORES = 1` (or 8 for multi-core).

**TPU tips**: use batch size 32–128; set `num_workers=0`; use `xm.save()` for checkpoints.

Expected speedup: 5–10× (single core) or 30–50× (8 cores) vs CPU.

# Colab Training Guide

Three training approaches for fungal species classification. All scripts run in Google Colab and save backbone weights (without classification head) to `weights/`.

## Approaches

| Approach | Script | Pretraining | Expected Acc | Time | GPU |
|----------|--------|-------------|--------------|------|-----|
| ImageNet (baseline) | `train_models.py` | 14M general images | 70–85% | 2–3h | ~4GB |
| CellViT ViT | `train_models_cellvit.py` | Cell microscopy | 75–90% | 4–5h | ~6GB |
| SimCLR self-supervised | `train_models_selfsupervised.py` | 1305 fungal images | 75–95% | 5–6h | ~8GB |

**When to use:**
- **ImageNet** — quick baseline or limited compute
- **CellViT** — have pretrained microscopy weights; domain similarity helps
- **SimCLR** — maximum performance; leverages all 1305 images (including test strains, no labels used)

## Colab Setup (all scripts)

```python
from google.colab import drive
drive.mount('/content/drive')

!curl -LsSf https://astral.sh/uv/install.sh | sh

import os
os.environ["PATH"] = f"/root/.local/bin:{os.environ['PATH']}"

!uv pip install torch torchvision tqdm scikit-learn pandas matplotlib pillow
```

## Running Each Script

### ImageNet Baseline

```python
!python /content/drive/MyDrive/mycoai/colab/train_models.py
```

Outputs: `weights/ResNet50_finetuned.pth`, `MobileNetV2_finetuned.pth`, `EfficientNetB1_finetuned.pth`

### CellViT ViT

Download pretrained weights (optional but recommended):
- Source: `pretrained/vit_256_teacher.pth` (see `VIT_NOTES.md` for weight types)

```python
!python /content/drive/MyDrive/mycoai/colab/train_models_cellvit.py
```

Output: `weights/ViT_CellViT_finetuned.pth`

### SimCLR Self-Supervised

Two-stage: Stage 1 = self-supervised pretraining on all 1305 images (100 epochs, ~3–4h), Stage 2 = supervised fine-tune on 1011 training images (50 epochs, ~1–2h).

```python
!python /content/drive/MyDrive/mycoai/colab/train_models_selfsupervised.py
```

Outputs: `weights/ResNet50_SimCLR_pretrained_encoder.pth`, `weights/ResNet50_SimCLR_finetuned.pth`

## Technical Details

| | ImageNet | CellViT | SimCLR (Stage 1 / Stage 2) |
|---|---|---|---|
| Loss | Cross-Entropy | Cross-Entropy | NT-Xent / Cross-Entropy |
| Optimizer | Adam lr=1e-4 | AdamW lr=1e-4 wd=0.05 | Adam lr=3e-4 / Adam lr=1e-5 |
| Augmentation | Flips, rotation, color jitter | Same | Strong contrastive (crop, blur, grayscale) |
| Early stopping | patience=10 | patience=10 | patience=10 |

All scripts use the same train/val split: 1011 training images (24 strains), 294 validation images (7 test strains).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM | Reduce batch size: CellViT → 8, SimCLR pretrain → 32 |
| Slow convergence | Reduce LR by 10×; check augmentation isn't too aggressive |
| Missing files | Verify `strain_to_specy.csv` and `segmented_image_metadata.json` exist |

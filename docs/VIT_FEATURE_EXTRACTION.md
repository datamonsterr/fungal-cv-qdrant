# Vision Transformer (ViT) Feature Extraction

Extracts 768-dim features using ViT models pretrained on microscopy/medical imaging.

## Available Weight Types

| Key | Model | Best For |
|-----|-------|----------|
| `vit256_dino` | ViT-256 DINO (self-supervised) | **Recommended for fungi** — no domain bias |
| `cellvit_x20` | CellViT at 20x magnification | Cell/nuclei-like structures |
| `cellvit_x40` | CellViT at 40x magnification | Cell/nuclei-like structures |
| `sam_vit_b` | SAM ViT-Base encoder | General purpose |
| `sam_vit_l` / `sam_vit_h` | SAM ViT-Large/Huge | General purpose (larger) |

## Setup

Download pretrained weights and place in:
```
pretrained/
├── CellViT/CellViT-256-x20.pth, CellViT-256-x40.pth
├── SAM/sam_vit_b.pth, sam_vit_l.pth, sam_vit_h.pth
└── ViT-256/vit256_small_dino.pth
```

## Usage

```bash
# Extract features
uv run python src/main.py extract-vit --weights-type vit256_dino

# Upload to Qdrant
docker compose up -d
uv run python src/main.py upload-vit --collection myco_fungi_features_vit

# Evaluate
uv run python src/main.py evaluate-all \
  --collection myco_fungi_features_vit --k 7
```

## Notes

- GPU recommended (~2GB VRAM); falls back to CPU
- Fine-tuning: see `colab/train_models_cellvit.py`
- References: [CellViT](https://github.com/TIO-IKIM/CellViT) · [SAM](https://github.com/facebookresearch/segment-anything) · [DINO](https://github.com/facebookresearch/dino)

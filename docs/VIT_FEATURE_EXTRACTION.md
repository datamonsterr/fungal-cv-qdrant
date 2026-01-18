# Vision Transformer (ViT) Feature Extraction

This document explains how to extract features using Vision Transformer models pretrained on microscopy and medical imaging datasets.

## Available Pretrained Weights

### CellViT
Pretrained on nuclei segmentation from PanNuke dataset:
- `cellvit_x20` - CellViT-256 trained on 20x magnification images
- `cellvit_x40` - CellViT-256 trained on 40x magnification images

### SAM (Segment Anything Model)
ViT encoder-only weights from SAM:
- `sam_vit_b` - SAM ViT-Base (encoder only)
- `sam_vit_l` - SAM ViT-Large (encoder only)
- `sam_vit_h` - SAM ViT-Huge (encoder only)

### ViT-256 DINO
Self-supervised learning on microscopy:
- `vit256_dino` - ViT-256 trained with DINO (recommended default)

## Setup

### 1. Download Pretrained Weights

Download from: https://drive.google.com/drive/folders/1zFO4bgo7yvjT9rCJi_6Mt6_07wfr0CKU?usp=sharing

Organize in your project:
```
pretrained/
├── CellViT/
│   ├── CellViT-256-x20.pth
│   └── CellViT-256-x40.pth
├── SAM/
│   ├── sam_vit_b.pth
│   ├── sam_vit_l.pth
│   └── sam_vit_h.pth
└── ViT-256/
    └── vit256_small_dino.pth
```

## Usage

### Extract ViT Features

```bash
# Using ViT-256 DINO (default, recommended for fungi)
uv run python -m src.main extract-vit --weights-type vit256_dino

# Using CellViT-x20 (for nuclei/cell-like structures)
uv run python -m src.main extract-vit --weights-type cellvit_x20

# Using SAM ViT-Base (general purpose)
uv run python -m src.main extract-vit --weights-type sam_vit_b

# Custom output path
uv run python -m src.main extract-vit \
  --weights-type vit256_dino \
  --output Dataset/custom_vit_features.json
```

### Upload to Qdrant

```bash
# Upload to default collection (myco_fungi_features_vit)
uv run python -m src.main upload-vit

# Custom collection name
uv run python -m src.main upload-vit \
  --collection myco_fungi_features_vit_dino \
  --features Dataset/vit_features.json
```

### Combined Extraction and Upload

```bash
# Extract and upload in sequence
uv run python -m src.main extract-vit --weights-type vit256_dino
uv run python -m src.main upload-vit
```

## Evaluation

Once features are uploaded, evaluate using the ViT collection:

```bash
# Evaluate all configurations with ViT features
uv run python -m src.main evaluate-all \
  --collection myco_fungi_features_vit \
  --k 7

# Evaluate specific extractor (after fine-tuning)
uv run python -m src.main evaluate \
  --collection myco_fungi_features_vit \
  --extractor vit_finetuned \
  --k 7 \
  --strategy weighted
```

## Feature Dimensions

All ViT models use the same architecture and produce **768-dimensional** feature vectors (from the cls token).

## Which Weights to Use?

### For Fungal Classification:
**Recommended: `vit256_dino`**
- Self-supervised learning provides robust general-purpose features
- 256x256 resolution matches our image size
- No domain bias from specific pretraining tasks

### For Cell/Nuclei-like Structures:
**Consider: `cellvit_x20` or `cellvit_x40`**
- If your fungi images have cell-like structures
- Choose x20 or x40 based on your microscopy magnification

### For General Purpose:
**Consider: `sam_vit_b`**
- Trained on diverse natural images
- Good generalization
- Base model (`sam_vit_b`) is sufficient; larger models may overfit

## Fine-tuning ViT

After extracting initial features, you can fine-tune the ViT model:

1. Train ViT classifier (see `colab/train_models_cellvit.py`)
2. Extract features from fine-tuned model
3. Upload to new collection
4. Evaluate performance improvement

## Performance Notes

- **GPU Recommended**: ViT extraction is much faster with GPU
- **TPU Support**: Available in Colab (see `colab/train_models_cellvit.py`)
- **Batch Processing**: Features are extracted one image at a time
- **Memory**: ~2GB GPU memory for inference

## Troubleshooting

### Weights not loading
```
Error: Pretrained weights not found
```
**Solution**: Download weights and place in `pretrained/` directory

### Out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: 
- Close other GPU processes
- Use smaller ViT model (sam_vit_b instead of sam_vit_h)
- Process on CPU (slower but works)

### Import errors
```
ImportError: cannot import name 'ViT256DinoExtractor'
```
**Solution**: Make sure you're using the latest code version

## References

- **CellViT**: https://github.com/TIO-IKIM/CellViT
- **SAM**: https://github.com/facebookresearch/segment-anything
- **DINO**: https://github.com/facebookresearch/dino

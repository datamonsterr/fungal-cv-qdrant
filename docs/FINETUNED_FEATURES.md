# Using Fine-Tuned Models for Feature Extraction

This guide explains how to use your fine-tuned deep learning models for feature extraction and upload them to Qdrant.

## Overview

After training models with `colab/train_models.py`, you'll have fine-tuned weights in the `weights/` directory. These can now be used for feature extraction with better domain-specific features.

## Workflow

### 1. Train Models (Already Done)

```bash
# You've already done this:
# - Trained ResNet50, MobileNetV2, EfficientNetB1 on fungal dataset
# - Saved backbone weights to weights/ directory
```

**Output files:**
- `weights/ResNet50_finetuned.pth`
- `weights/MobileNetV2_finetuned.pth`
- `weights/EfficientNetB1_finetuned.pth`

### 2. Extract Features with Fine-Tuned Models

Run feature extraction using the fine-tuned weights:

```bash
uv run python -m src.main extract-finetuned
```

Or directly:

```bash
uv run python src/scripts/extract_finetuned_features.py
```

**What it does:**
- Loads fine-tuned weights from `weights/` directory
- Extracts features from all segmented images
- Saves to `Dataset/finetuned_dl_features.json`

**Output:**
- `Dataset/finetuned_dl_features.json` - Contains three feature vectors per image:
  - `ResNet50_finetuned` (2048 dimensions)
  - `MobileNetV2_finetuned` (1280 dimensions)
  - `EfficientNetB1_finetuned` (1280 dimensions)

### 3. Upload to Qdrant

Add the fine-tuned features to your existing Qdrant collection:

```bash
uv run python -m src.main upload-finetuned
```

Or directly:

```bash
uv run python src/scripts/upload_finetuned_features.py
```

**What it does:**
- Connects to your existing Qdrant collection
- Adds new vector fields for fine-tuned features
- Updates all existing points with new vectors
- **Does NOT delete or re-extract old features**

**Result:**
Your collection now has both:
- **Old features:** HOG, Gabor, ColorHistogram, ColorHistogramHS, ResNet50 (ImageNet), MobileNetV2 (ImageNet), EfficientNetB1 (ImageNet)
- **New features:** ResNet50_finetuned, MobileNetV2_finetuned, EfficientNetB1_finetuned

## Complete Workflow Example

```bash
# Step 1: Train models (in Google Colab)
# Run: colab/train_models.py
# Download weights/ directory to your local machine

# Step 2: Extract fine-tuned features
uv run python -m src.main extract-finetuned

# Step 3: Upload to Qdrant (make sure Qdrant is running)
docker-compose up -d
uv run python -m src.main upload-finetuned
```

## Using Fine-Tuned Features for Prediction

After uploading, you can use the fine-tuned features for better predictions:

```python
from src.feature_extraction.feature_extractors import ResNet50Extractor

# Use fine-tuned ResNet50
extractor = ResNet50Extractor(weights_path="weights/ResNet50_finetuned.pth")

# Extract features
features = extractor.extract(image)

# Search in Qdrant using "ResNet50_finetuned" vector
results = client.search(
    collection_name="myco_fungi_features_full",
    query_vector=("ResNet50_finetuned", features.tolist()),
    limit=5
)
```

## Performance Comparison

Expected improvements with fine-tuned models:

| Model | ImageNet Accuracy | Fine-Tuned Accuracy | Improvement |
|-------|------------------|---------------------|-------------|
| ResNet50 | ~75% | ~85-90% | +10-15% |
| MobileNetV2 | ~70% | ~80-85% | +10-15% |
| EfficientNetB1 | ~78% | ~88-92% | +10-14% |

*Note: Actual improvements depend on your training results*

## Verification

After uploading, verify the collection has new features:

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
collection_info = client.get_collection("myco_fungi_features_full")

print("Vector fields:", collection_info.config.params.vectors.keys())
# Should show: ResNet50_finetuned, MobileNetV2_finetuned, EfficientNetB1_finetuned
```

## Troubleshooting

### Issue: "No fine-tuned weights found"

**Solution:** Make sure you have these files in `weights/`:
```bash
ls weights/
# Should see:
# ResNet50_finetuned.pth
# MobileNetV2_finetuned.pth
# EfficientNetB1_finetuned.pth
```

### Issue: "Collection not found"

**Solution:** Run initial feature extraction first:
```bash
uv run python -m src.main extract
```

### Issue: GPU out of memory during extraction

**Solution:** The extractors automatically use CPU if GPU is not available. If using GPU and getting OOM, restart Python to clear GPU memory.

## Next Steps

1. **Evaluate performance:** Run evaluation with fine-tuned features
2. **Compare results:** Test both ImageNet and fine-tuned versions
3. **Use best model:** Choose the feature extractor with highest accuracy for your use case

## File Structure

```
fungal-cv-qdrant/
├── weights/                           # Fine-tuned model weights
│   ├── ResNet50_finetuned.pth
│   ├── MobileNetV2_finetuned.pth
│   ├── EfficientNetB1_finetuned.pth
│   └── classes.npy                   # Label encoder classes
├── Dataset/
│   ├── segmented_features.json       # Original features (all extractors)
│   └── finetuned_dl_features.json    # NEW: Fine-tuned DL features only
└── src/
    ├── scripts/
    │   ├── extract_finetuned_features.py    # NEW: Extract fine-tuned features
    │   └── upload_finetuned_features.py     # NEW: Upload to Qdrant
    └── feature_extraction/
        └── feature_extractors.py     # UPDATED: Now loads fine-tuned weights
```

## Benefits

✅ **No re-extraction:** Keep all your existing features  
✅ **Better accuracy:** Fine-tuned on your specific fungi dataset  
✅ **Easy comparison:** Compare ImageNet vs fine-tuned in same collection  
✅ **Flexible:** Use whichever features work best for your task  
✅ **Incremental:** Add more fine-tuned models later (e.g., from CellViT or SimCLR)

# Quick Start: Fine-Tuned Feature Extraction

After training your models in Google Colab, follow these steps to extract features and update Qdrant:

## Step-by-Step Guide

### 1. Download Weights from Colab

After running `colab/train_models.py` in Colab, download the `weights/` directory to your local machine:

```python
# In Colab, zip and download weights
!cd /content/drive/MyDrive/mycoai && zip -r weights.zip weights/
```

Then extract to your project's `weights/` directory.

### 2. Extract Fine-Tuned Features

```bash
uv run python -m src.main extract-finetuned
```

This will:
- Load `weights/ResNet50_finetuned.pth`
- Load `weights/MobileNetV2_finetuned.pth`  
- Load `weights/EfficientNetB1_finetuned.pth`
- Extract features from all 1305 segmented images
- Save to `Dataset/finetuned_dl_features.json`

**Expected output:**
```
Found 1305 images in metadata
Initializing ResNet50 with fine-tuned weights: weights/ResNet50_finetuned.pth
✓ Fine-tuned ResNet50 weights loaded successfully
Initializing MobileNetV2 with fine-tuned weights: weights/MobileNetV2_finetuned.pth
✓ Fine-tuned MobileNetV2 weights loaded successfully
Initializing EfficientNetB1 with fine-tuned weights: weights/EfficientNetB1_finetuned.pth
✓ Fine-tuned EfficientNetB1 weights loaded successfully

Extracting features with 3 fine-tuned models...
Processed 100/1305 images...
Processed 200/1305 images...
...
Processed 1305/1305 images...

Fine-tuned feature extraction complete!
Processed 1305 images
Feature types: ['ResNet50_finetuned', 'MobileNetV2_finetuned', 'EfficientNetB1_finetuned']
Total feature dimension: 4608
Results saved to: Dataset/finetuned_dl_features.json
```

### 3. Upload to Qdrant

Start Qdrant if not running:
```bash
docker-compose up -d
```

Upload the new features:
```bash
uv run python -m src.main upload-finetuned
```

This will:
- Connect to existing Qdrant collection
- Add 3 new vector fields (ResNet50_finetuned, MobileNetV2_finetuned, EfficientNetB1_finetuned)
- Update all 1305 points with new vectors
- **Keep all existing features** (HOG, Gabor, ColorHistogram, etc.)

**Expected output:**
```
Connecting to Qdrant at http://localhost:6333...
✓ Successfully connected to Qdrant

Loaded 1305 fine-tuned feature records

New feature types to add: ['ResNet50_finetuned', 'MobileNetV2_finetuned', 'EfficientNetB1_finetuned']
New feature dimensions: {'ResNet50_finetuned': 2048, 'MobileNetV2_finetuned': 1280, 'EfficientNetB1_finetuned': 1280}

Current collection 'myco_fungi_features_full' info:
  Points: 1305
  Existing vectors: ['HOG', 'Gabor', 'ColorHistogram', 'ColorHistogramHS', 'ResNet50', 'MobileNetV2', 'EfficientNetB1']

Adding new vector fields to collection...
  ✓ Added vector field: ResNet50_finetuned (dim=2048)
  ✓ Added vector field: MobileNetV2_finetuned (dim=1280)
  ✓ Added vector field: EfficientNetB1_finetuned (dim=1280)

Fetching existing points from collection...
Retrieved 1305 existing points

Updating points with fine-tuned features...
  Updated 100/1305 points...
  Updated 200/1305 points...
  ...
  Updated 1305/1305 points...

============================================================
Update complete!
============================================================
Updated points: 1305
Skipped points (no matching features): 0

Final collection info:
  Total points: 1305
  Vector fields: ['HOG', 'Gabor', 'ColorHistogram', 'ColorHistogramHS', 'ResNet50', 'MobileNetV2', 'EfficientNetB1', 'ResNet50_finetuned', 'MobileNetV2_finetuned', 'EfficientNetB1_finetuned']
============================================================

✅ Successfully updated collection 'myco_fungi_features_full'
```

### 4. Verify Update

Check collection in Qdrant:

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
info = client.get_collection("myco_fungi_features_full")

print(f"Points: {info.points_count}")
print(f"Vectors: {list(info.config.params.vectors.keys())}")
```

You should see 10 vector fields total:
- 7 original: HOG, Gabor, ColorHistogram, ColorHistogramHS, ResNet50, MobileNetV2, EfficientNetB1
- 3 new: ResNet50_finetuned, MobileNetV2_finetuned, EfficientNetB1_finetuned

## What You Get

✅ **Better features:** Trained on your specific fungal dataset  
✅ **No re-extraction:** All old features are preserved  
✅ **Easy comparison:** Test both ImageNet and fine-tuned in same collection  
✅ **Flexible:** Use whichever features work best

## Next Steps

Test the fine-tuned features:

```bash
# Evaluate with fine-tuned ResNet50
uv run python -m src.main evaluate \
  --extractor resnet50 \
  --k 7 \
  --strategy weighted \
  --environment all
```

Compare with ImageNet version to see the improvement!

## Troubleshooting

**"No fine-tuned weights found"**
- Make sure you have downloaded the weights from Colab
- Check that files exist in `weights/` directory

**"Collection not found"**
- Run initial extraction first: `uv run python -m src.main extract`

**GPU memory issues**
- The extractors will automatically use CPU if GPU unavailable
- Close other applications to free GPU memory

## File Summary

| File | Purpose | Size |
|------|---------|------|
| `weights/ResNet50_finetuned.pth` | Fine-tuned ResNet50 backbone | ~90MB |
| `weights/MobileNetV2_finetuned.pth` | Fine-tuned MobileNetV2 backbone | ~9MB |
| `weights/EfficientNetB1_finetuned.pth` | Fine-tuned EfficientNetB1 backbone | ~24MB |
| `Dataset/finetuned_dl_features.json` | Extracted features | ~150MB |

Total storage needed: ~275MB

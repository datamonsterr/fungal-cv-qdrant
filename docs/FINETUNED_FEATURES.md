# Fine-Tuned Feature Extraction

Models are trained in Google Colab (`colab/train_models_fold_efficientnetb1.py`) and weights placed in `weights/`.

## Workflow

```bash
# 1. Extract features using fine-tuned weights
uv run python src/main.py extract-finetuned

# 2. Upload to Qdrant (adds vectors to existing collection, no re-extraction)
docker compose up -d
uv run python src/main.py upload-finetuned
```

**Output files:**
- `weights/ResNet50_finetuned.pth`
- `weights/MobileNetV2_finetuned.pth`
- `weights/EfficientNetB1_finetuned.pth`
- `Dataset/finetuned_dl_features.json` — 3 vectors per image (2048d, 1280d, 1280d)

After upload, the collection `myco_fungi_features_full_finetuned` contains both ImageNet and fine-tuned vectors.

## Evaluate Fine-Tuned Features

```bash
uv run python src/main.py evaluate \
  --extractor efficientnetb1_finetuned \
  --k 7 --strategy weighted \
  --collection myco_fungi_features_full_finetuned
```

## Using in Code

```python
from src.feature_extraction.feature_extractors import EfficientNetB1Extractor

extractor = EfficientNetB1Extractor(weights_path="weights/EfficientNetB1_finetuned.pth")
features = extractor.extract(image)  # 1280-dim vector
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "No fine-tuned weights found" | Ensure `weights/*.pth` files exist |
| "Collection not found" | Run `extract` + `upload` (base pipeline) first |
| GPU OOM | Extractors fall back to CPU automatically |

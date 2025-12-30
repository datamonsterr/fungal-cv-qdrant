# Training Pipeline Summary

## What Was Created

### 1. Main Training Script: `train_models.py`

A comprehensive script that:
- Loads segmented images from `../Dataset/segmented_image/`
- Uses the same test strain selection logic as `evaluate_species.py`
- Trains three deep learning models: ResNet50, MobileNetV2, EfficientNetV2B0
- Implements data augmentation, early stopping, and learning rate scheduling
- Saves trained weights to `weights/{model_name}/{timestamp}/`
- Generates training visualizations and comprehensive reports

**Key Features:**
- Species-level classification (not strain-level)
- One test strain per species excluded from training
- 85/15 train/validation split from remaining strains
- Batch size: 32, Max epochs: 100
- Automatic model checkpointing and early stopping

### 2. Updated Feature Extractors: `feature_extractors.py`

Modified three extractor classes to support fine-tuned weights:
- `ResNet50Extractor(weights_path=None)`
- `MobileNetV2Extractor(weights_path=None)`
- `EfficientNetV2B0Extractor(weights_path=None)`

**Usage:**
```python
# Default: ImageNet weights
extractor = ResNet50Extractor()

# With fine-tuned weights
extractor = ResNet50Extractor(weights_path='./weights/resnet50/20241229_143022/best_model.h5')
```

### 3. Example Script: `example_use_trained_weights.py`

Demonstrates how to:
- Load extractors with default ImageNet weights
- Load extractors with fine-tuned weights
- Use extractors in your code

### 4. Documentation

- **`TRAINING_PIPELINE.md`**: Comprehensive documentation covering architecture, training strategy, outputs, and troubleshooting
- **`TRAINING_QUICKSTART.md`**: Step-by-step guide to get started quickly

## How It Works

### Data Flow

```
reformat_dataset.py
        ↓
segmented_image/ + segmented_image_metadata.json
        ↓
train_models.py (selects test strains, trains models)
        ↓
weights/{model_name}/{timestamp}/
        ↓
feature_extractors.py (loads fine-tuned weights)
        ↓
evaluate_species.py (evaluates with fine-tuned features)
```

### Training Process

1. **Load Data**: Read segmented images and metadata
2. **Select Test Strains**: One strain per species (same as evaluation)
3. **Split Data**: 
   - Test set: All images from selected test strains
   - Training set: Remaining images split 85/15 train/val
4. **Build Model**: Pre-trained base + classification head
5. **Train**: With data augmentation, early stopping, LR scheduling
6. **Save**: Weights, charts, reports, metadata

### Test Strain Selection

Uses the same logic as `evaluate_species.py`:

```python
def select_test_strains(available_strains, strain_to_specy):
    species_to_strains = defaultdict(list)
    
    for strain in available_strains:
        if strain in strain_to_specy:
            species = strain_to_specy[strain]
            species_to_strains[species].append(strain)
    
    test_strains = {}
    for species, strains in species_to_strains.items():
        if len(strains) > 1:
            test_strains[species] = strains[1]  # Second strain
        else:
            test_strains[species] = strains[0]  # Only strain
    
    return test_strains
```

This ensures consistency between training and evaluation.

## Quick Start

```bash
# 1. Enter environment
nix-shell -r "zsh"
source .venv/bin/activate

# 2. Train models
uv run python train_models.py

# 3. Check results
ls -la weights/
cat weights/resnet50/*/resnet50_training_report.txt

# 4. Use trained weights
# Edit your code to use:
extractor = ResNet50Extractor(weights_path='./weights/resnet50/TIMESTAMP/best_model.h5')
```

## Output Structure

```
weights/
├── resnet50/
│   └── 20241229_143022/
│       ├── best_model.h5              # ← Use this for inference
│       ├── final_weights.h5
│       ├── label_encoder.npy
│       ├── training_history.json
│       ├── resnet50_training_history.png  # ← Visualization
│       ├── resnet50_training_report.txt   # ← Detailed metrics
│       └── metadata.json
├── mobilenetv2/
│   └── 20241229_143022/
│       └── ...
└── efficientnetv2b0/
    └── 20241229_143022/
        └── ...
```

## Integration with Existing Pipeline

### Before (ImageNet features only)

```python
from feature_extractors import ResNet50Extractor

extractor = ResNet50Extractor()
features = extractor.extract(image)
```

### After (Fine-tuned features)

```python
from feature_extractors import ResNet50Extractor

# Load with fine-tuned weights
extractor = ResNet50Extractor(
    weights_path='./weights/resnet50/20241229_143022/best_model.h5'
)
features = extractor.extract(image)
```

**No other code changes required!** The extractor API remains the same.

### Use in Evaluation

```python
from evaluate_species import run_species_evaluation

# With fine-tuned extractor
extractor = ResNet50Extractor(weights_path='./weights/resnet50/TIMESTAMP/best_model.h5')

results, test_strains = run_species_evaluation(
    client=client,
    collection_name="your_collection",
    feature_extractor=extractor,  # ← Fine-tuned extractor
    k=5
)
```

## Expected Improvements

Fine-tuned models should show better performance because:

1. **Domain Adaptation**: Models learn fungi-specific features instead of general ImageNet features
2. **Species Discrimination**: Trained specifically to distinguish between myco fungi species
3. **Context Awareness**: Learn patterns relevant to your segmented images
4. **Reduced Dimensionality Confusion**: Classification head trained on your specific species classes

## Files Created/Modified

### New Files
- ✅ `train_models.py` (685 lines)
- ✅ `example_use_trained_weights.py` (88 lines)
- ✅ `TRAINING_PIPELINE.md` (306 lines)
- ✅ `TRAINING_QUICKSTART.md` (227 lines)
- ✅ `TRAINING_SUMMARY.md` (this file)

### Modified Files
- ✅ `feature_extractors.py`
  - Added `weights_path` parameter to ResNet50Extractor
  - Added `weights_path` parameter to MobileNetV2Extractor
  - Added `weights_path` parameter to EfficientNetV2B0Extractor
  - Added `load_model` import from tensorflow.keras.models

### No Changes Needed
- ✅ `requirements.txt` (all dependencies already present)
- ✅ `evaluate_species.py` (works with updated extractors)
- ✅ `reformat_dataset.py` (provides input data)

## Next Steps

1. **Train Models**: Run `train_models.py`
2. **Review Results**: Check training reports and charts
3. **Select Best Model**: Compare validation accuracies
4. **Update Extractors**: Use best model weights in your code
5. **Evaluate**: Run evaluation pipeline with fine-tuned features
6. **Compare**: Compare results with ImageNet-only features

## Benefits

✅ **Consistent Data Split**: Same test strains as evaluation pipeline
✅ **Easy Integration**: No API changes, just add `weights_path` parameter
✅ **Comprehensive Logging**: Detailed reports, charts, and metadata
✅ **Automatic Best Model**: Saves best model based on validation accuracy
✅ **Reproducible**: All training parameters saved in metadata
✅ **Flexible**: Can still use ImageNet weights if needed

## Training Time Estimates

| Dataset Size | Hardware | Time per Model |
|-------------|----------|----------------|
| ~10k images | CPU | 4-8 hours |
| ~10k images | GPU (consumer) | 30-90 minutes |
| ~10k images | GPU (datacenter) | 15-30 minutes |

With early stopping, actual training time may be less.

## Support

- **Quick Start**: See `TRAINING_QUICKSTART.md`
- **Detailed Info**: See `TRAINING_PIPELINE.md`
- **Example Code**: See `example_use_trained_weights.py`
- **Issues**: Check troubleshooting sections in documentation

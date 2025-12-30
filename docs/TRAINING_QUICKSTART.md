# Quick Start: Training Fine-Tuned Models

This guide walks you through training fine-tuned deep learning models for myco fungi species classification.

## Prerequisites

1. **Segmented Dataset Ready**: You should have already run `reformat_dataset.py` to create segmented images
2. **Environment Set Up**: Nix shell and virtual environment configured
3. **Required Files**:
   - `../Dataset/segmented_image/` (segmented images)
   - `../Dataset/segmented_image_metadata.json` (metadata)
   - `../Dataset/strain_to_specy.csv` (strain-species mapping)

## Step 1: Enter Environment

```bash
# Enter Nix shell
nix-shell -r "zsh"

# Activate virtual environment  
source .venv/bin/activate
```

## Step 2: Verify Dependencies

```bash
# Install/update dependencies
uv pip install -r requirements.txt
```

## Step 3: Run Training

```bash
# Train all models (ResNet50, MobileNetV2, EfficientNetV2B0)
uv run python train_models.py
```

**What happens:**
- Loads segmented images and metadata
- Selects one test strain per species (excluded from training)
- Trains each model for up to 100 epochs with early stopping
- Saves weights, charts, and reports to `weights/{model_name}/{timestamp}/`

**Expected time:** 2-8 hours depending on dataset size and hardware

## Step 4: Monitor Training

Watch the console output for:
- Data loading progress
- Model architecture summary
- Training progress (epoch by epoch)
- Validation accuracy improvements
- Early stopping triggers

Example output:
```
================================================================================
Training ResNet50
================================================================================
Train samples: 8523
Validation samples: 1504
Test samples: 1842
Number of classes: 47

Model Summary:
[... model architecture ...]

Starting training...
Epoch 1/100
267/267 [==============================] - 45s 168ms/step - loss: 2.1234 - accuracy: 0.4567 - val_loss: 1.8765 - val_accuracy: 0.5234
...
```

## Step 5: Review Results

After training completes, check the results:

```bash
# List all trained models
ls -la weights/

# View training report
cat weights/resnet50/20241229_143022/resnet50_training_report.txt

# View training charts (Linux)
xdg-open weights/resnet50/20241229_143022/resnet50_training_history.png

# Or on macOS
open weights/resnet50/20241229_143022/resnet50_training_history.png
```

## Step 6: Use Trained Weights

### Option A: Test with Example Script

```bash
# Edit example_use_trained_weights.py to point to your weights
# Then run:
uv run python example_use_trained_weights.py
```

### Option B: Use in Your Code

```python
from feature_extractors import ResNet50Extractor
import cv2

# Load with fine-tuned weights
extractor = ResNet50Extractor(
    weights_path='./weights/resnet50/20241229_143022/best_model.h5'
)

# Extract features
image = cv2.imread('path/to/segmented/image.jpg')
features = extractor.extract(image)
print(f"Feature shape: {features.shape}")  # (2048,)
```

### Option C: Use in Evaluation Pipeline

```python
from feature_extractors import ResNet50Extractor
from evaluate_species import run_species_evaluation
from qdrant_client import QdrantClient

# Initialize
client = QdrantClient(url="http://localhost:6333")

# Load fine-tuned extractor
extractor = ResNet50Extractor(
    weights_path='./weights/resnet50/20241229_143022/best_model.h5'
)

# Run evaluation
results, test_strains = run_species_evaluation(
    client=client,
    collection_name="your_collection_name",
    feature_extractor=extractor,
    k=5,
    output_dir="./results"
)
```

## Understanding Output Files

Each training run creates a timestamped directory with:

| File | Description |
|------|-------------|
| `best_model.h5` | Best model checkpoint (use this for inference) |
| `final_weights.h5` | Final model after all epochs |
| `label_encoder.npy` | Species class labels |
| `training_history.json` | Per-epoch metrics |
| `training_history.png` | Training curves visualization |
| `training_report.txt` | Comprehensive training summary |
| `metadata.json` | Training configuration and paths |

## Training Configuration

Default settings in `train_models.py`:

```python
batch_size = 32              # Images per batch
epochs = 100                 # Maximum epochs
initial_learning_rate = 0.001  # Starting learning rate
target_size = (224, 224)     # Input image size
```

To modify, edit these variables in the `main()` function.

## Common Issues and Solutions

### GPU Not Detected

If TensorFlow doesn't detect GPU:
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Training will work on CPU but will be slower.

### Out of Memory

Reduce batch size in `train_models.py`:
```python
batch_size = 16  # or 8
```

### Training Taking Too Long

- Use GPU if available
- Reduce epochs (but may hurt performance)
- Train one model at a time instead of all three

### Poor Validation Accuracy

- Check if dataset is balanced
- Try training for more epochs
- Adjust learning rate
- Check data augmentation settings

## Next Steps

1. **Compare Models**: Review training reports to see which model performs best
2. **Fine-Tune Further**: Unfreeze base layers for additional fine-tuning
3. **Evaluate Performance**: Use trained models in evaluation pipeline
4. **Experiment**: Try different hyperparameters and retrain

## Tips for Best Results

- **Use GPU**: Training on GPU is 10-50x faster than CPU
- **Monitor Validation**: Watch for overfitting (val_loss increases while train_loss decreases)
- **Early Stopping**: Already implemented - training stops if no improvement for 15 epochs
- **Learning Rate**: Automatically reduces if validation loss plateaus
- **Data Augmentation**: Already enabled to prevent overfitting

## Questions?

See the comprehensive documentation in `TRAINING_PIPELINE.md` for detailed information about:
- Architecture details
- Training strategy
- Data augmentation
- Integration with evaluation pipeline
- Troubleshooting guide

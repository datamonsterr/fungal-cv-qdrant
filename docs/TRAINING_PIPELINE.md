# Model Training Pipeline

This document describes the training pipeline for fine-tuning deep learning models on myco fungi species classification.

## Overview

The training pipeline fine-tunes pre-trained deep learning models (ResNet50, MobileNetV2, EfficientNetV2B0) on segmented myco fungi images for species classification.

### Key Features

- **Train/Test Split Strategy**: Uses `select_one_strain_per_species` logic to exclude one test strain per species from training
- **Data Source**: Segmented images from `reformat_dataset.py`
- **Models Supported**: ResNet50, MobileNetV2, EfficientNetV2B0
- **Outputs**: Trained weights, training charts, detailed reports, and metadata

## File Structure

```
scripts/
├── train_models.py              # Main training script
├── feature_extractors.py        # Updated extractors with fine-tuned weights support
├── example_use_trained_weights.py  # Example usage of trained models
└── weights/                     # Output directory for trained models
    ├── resnet50/
    │   └── YYYYMMDD_HHMMSS/    # Timestamped training run
    │       ├── best_model.h5    # Best model checkpoint
    │       ├── final_weights.h5 # Final model weights
    │       ├── label_encoder.npy # Label encoder classes
    │       ├── training_history.json
    │       ├── training_history.png
    │       ├── training_report.txt
    │       └── metadata.json
    ├── mobilenetv2/
    └── efficientnetv2b0/
```

## Training Pipeline

### 1. Data Preparation

The training script:
1. Loads segmented image metadata from `../Dataset/segmented_image_metadata.json`
2. Loads strain-to-species mapping from `../Dataset/strain_to_specy.csv`
3. Selects one test strain per species (using same logic as `evaluate_species.py`)
4. Splits remaining data into training (85%) and validation (15%) sets

### 2. Model Architecture

Each model consists of:
- **Base Model**: Pre-trained on ImageNet (frozen initially)
- **Classification Head**:
  - GlobalAveragePooling2D
  - Dropout(0.5)
  - Dense(512, activation='relu')
  - Dropout(0.3)
  - Dense(num_classes, activation='softmax')

### 3. Training Configuration

```python
batch_size = 32
epochs = 100
initial_learning_rate = 0.001
target_size = (224, 224)
optimizer = Adam
loss = sparse_categorical_crossentropy
```

### 4. Data Augmentation

Training data augmentation includes:
- Random horizontal and vertical flips
- Random rotation (0°, 90°, 180°, 270°)
- Random brightness (±20%)
- Random contrast (0.8-1.2x)

### 5. Callbacks

- **ModelCheckpoint**: Saves best model based on validation accuracy
- **EarlyStopping**: Stops training if validation loss doesn't improve for 15 epochs
- **ReduceLROnPlateau**: Reduces learning rate by 0.5 if validation loss doesn't improve for 5 epochs

## Usage

### Training Models

```bash
# Ensure you're in the nix-shell environment
nix-shell -r "zsh"

# Activate virtual environment
source .venv/bin/activate

# Run training
uv run python train_models.py
```

The script will:
1. Train all three models (ResNet50, MobileNetV2, EfficientNetV2B0)
2. Save weights and results to `weights/{model_name}/{timestamp}/`
3. Generate training charts and comprehensive reports

### Using Trained Weights

#### Option 1: In Python Code

```python
from feature_extractors import ResNet50Extractor
import cv2

# Load extractor with fine-tuned weights
weights_path = './weights/resnet50/20241229_143022/best_model.h5'
extractor = ResNet50Extractor(weights_path=weights_path)

# Extract features
image = cv2.imread('path/to/image.jpg')
features = extractor.extract(image)
```

#### Option 2: Default ImageNet Weights

```python
from feature_extractors import ResNet50Extractor

# Use default ImageNet weights (no fine-tuning)
extractor = ResNet50Extractor()
features = extractor.extract(image)
```

### Example Script

Run the example script to see how to use trained weights:

```bash
uv run python example_use_trained_weights.py
```

## Training Outputs

### 1. Model Weights

- **best_model.h5**: Best model based on validation accuracy
- **final_weights.h5**: Final model weights after all epochs

### 2. Training Charts

`{model_name}_training_history.png` contains:
- Training and validation accuracy over epochs
- Training and validation loss over epochs

### 3. Training Report

`{model_name}_training_report.txt` includes:
- Dataset statistics (train/val/test sizes)
- Species classes and their IDs
- Test strains excluded from training
- Training metrics (accuracy, loss)
- Best validation performance

### 4. Metadata

`metadata.json` contains:
- Model configuration
- Training hyperparameters
- Final and best metrics
- Paths to all output files

### 5. Training History

`training_history.json` contains:
- Per-epoch metrics (accuracy, loss, val_accuracy, val_loss)

## Integration with Evaluation Pipeline

The trained models can be used in the evaluation pipeline:

```python
from feature_extractors import ResNet50Extractor
from evaluate_species import run_species_evaluation
from qdrant_client import QdrantClient

# Initialize client
client = QdrantClient(url="http://localhost:6333")

# Load fine-tuned extractor
extractor = ResNet50Extractor(
    weights_path='./weights/resnet50/20241229_143022/best_model.h5'
)

# Run evaluation
results, test_strains = run_species_evaluation(
    client=client,
    collection_name="segmented_features_colorhistogramhsconcatresnet50",
    feature_extractor=extractor,
    k=5,
    output_dir="./results"
)
```

## Train/Test Split Logic

The training script uses the same test strain selection logic as `evaluate_species.py`:

```python
def select_test_strains(available_strains, strain_to_specy):
    """Select one strain per species for testing."""
    species_to_strains = defaultdict(list)
    
    # Group strains by species
    for strain in available_strains:
        if strain in strain_to_specy:
            species = strain_to_specy[strain]
            species_to_strains[species].append(strain)
    
    # Select one strain per species
    test_strains = {}
    for species, strains in species_to_strains.items():
        if len(strains) > 1:
            test_strains[species] = strains[1]  # Take second strain
        else:
            test_strains[species] = strains[0]  # Take only available
    
    return test_strains
```

This ensures:
- One strain per species is held out for testing
- All images from test strains are excluded from training
- Training is performed on remaining strains
- Evaluation uses the same test strains

## Model Comparison

After training, compare models using:

```bash
# View training reports
cat weights/resnet50/20241229_143022/*_training_report.txt
cat weights/mobilenetv2/20241229_143022/*_training_report.txt
cat weights/efficientnetv2b0/20241229_143022/*_training_report.txt

# View training charts
open weights/resnet50/20241229_143022/*_training_history.png
open weights/mobilenetv2/20241229_143022/*_training_history.png
open weights/efficientnetv2b0/20241229_143022/*_training_history.png
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size in `train_models.py`:
```python
batch_size = 16  # or 8
```

### Issue: Training Too Slow

**Solution**: 
- Use GPU if available
- Reduce image size (not recommended as it may hurt accuracy)
- Train fewer models at once

### Issue: Overfitting

**Symptoms**: High training accuracy, low validation accuracy

**Solutions**:
- Increase dropout rates
- Add more data augmentation
- Use stronger regularization
- Early stopping (already implemented)

### Issue: Underfitting

**Symptoms**: Low training and validation accuracy

**Solutions**:
- Unfreeze base model layers for fine-tuning
- Train for more epochs
- Increase model capacity
- Reduce regularization

## Next Steps

1. **Train models**: Run `train_models.py`
2. **Compare results**: Check training reports and charts
3. **Use best model**: Update `feature_extractors.py` usage with best weights path
4. **Evaluate**: Run `evaluate_species.py` with fine-tuned extractors
5. **Iterate**: Adjust hyperparameters and retrain if needed

## Dependencies

Required packages (already in `requirements.txt`):
- tensorflow
- keras
- numpy
- opencv-python
- matplotlib
- pandas
- scikit-learn
- pydantic

## Notes

- Training time depends on dataset size and hardware (GPU recommended)
- Each model training creates a new timestamped directory
- Best model is automatically saved based on validation accuracy
- Fine-tuned models should perform better than ImageNet-only features for fungi species classification

# Training Approaches for Fungal Classification

This directory contains three different training approaches for fungal species classification:

## 📁 Training Scripts

### 1. **train_models.py** - ImageNet Pretraining (Baseline)
Standard transfer learning approach using models pretrained on ImageNet.

**Models**: ResNet50, MobileNetV2, EfficientNetB1

**Approach**:
- Start with ImageNet pretrained weights
- Replace classification head for 7 fungal species
- Fine-tune all layers on fungal dataset
- Training set: 1011 images (24 strains)
- Validation set: 294 images (7 test strains)

**Expected Performance**: Baseline accuracy ~70-85%

**Training Time**: ~2-3 hours per model (on GPU)

---

### 2. **train_models_cellvit.py** - CellViT ViT Pretraining (Microscopy-Specific)
Uses Vision Transformer pretrained on biomedical microscopy images.

**Model**: Vision Transformer (ViT-256)

**Approach**:
- Start with CellViT pretrained ViT weights (trained on cell/nucleus segmentation)
- Domain-specific features from microscopy images
- Fine-tune entire ViT for fungal classification
- Same train/val split as baseline

**Expected Performance**: Potentially 5-10% better than ImageNet due to domain similarity

**Training Time**: ~4-5 hours (ViT is more computationally intensive)

**Prerequisites**:
1. Download pretrained ViT weights from [CellViT Google Drive](https://drive.google.com/drive/folders/1zFO4bgo7yvjT9rCJi_6Mt6_07wfr0CKU?usp=sharing)
2. Save as `/content/drive/MyDrive/mycoai/pretrained/vit_256_teacher.pth`

**Note**: If pretrained weights are not available, it will train from random initialization.

---

### 3. **train_models_selfsupervised.py** - Self-Supervised Pretraining (Unlabeled Data)
Two-stage training: self-supervised pretraining on ALL images, then supervised fine-tuning.

**Model**: ResNet50 with SimCLR

**Approach**:

#### Stage 1: Self-Supervised Pretraining
- Uses ALL 1305 images (including test strains)
- No labels required
- SimCLR contrastive learning
- Learns fungal-specific visual representations
- Duration: 100 epochs (~3-4 hours)

#### Stage 2: Supervised Fine-tuning
- Uses only training strains (1011 images)
- Fine-tunes pretrained encoder for classification
- Duration: 50 epochs (~1-2 hours)

**Expected Performance**: 5-15% improvement over ImageNet baseline

**Training Time**: ~5-6 hours total (both stages)

**Benefits**:
- Leverages unlabeled data effectively
- Learns domain-specific features without manual annotation
- More robust representations
- Better generalization

---

## 🎯 Comparison

| Approach | Pretraining Data | Total Images Used | Expected Accuracy | Training Time | Complexity |
|----------|------------------|-------------------|-------------------|---------------|------------|
| **ImageNet (Baseline)** | 14M general images | 1011 labeled | 70-85% | 2-3h | Low |
| **CellViT ViT** | Cell microscopy | 1011 labeled | 75-90% | 4-5h | Medium |
| **SimCLR Self-Supervised** | 1305 fungal images | 1305 unlabeled + 1011 labeled | 75-95% | 5-6h | High |

---

## 🚀 Usage in Google Colab

### Common Setup (All Scripts)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies (if needed)
!pip install torch torchvision tqdm scikit-learn pandas matplotlib pillow
```

### Option 1: ImageNet Baseline

```python
# Run the baseline training
!python /content/drive/MyDrive/mycoai/train_models.py
```

**Output**:
- `weights/ResNet50_finetuned.pth`
- `weights/MobileNetV2_finetuned.pth`
- `weights/EfficientNetB1_finetuned.pth`
- Training history plots and JSON files

---

### Option 2: CellViT ViT

**Step 1**: Download pretrained weights
```python
# Download from: https://drive.google.com/drive/folders/1zFO4bgo7yvjT9rCJi_6Mt6_07wfr0CKU?usp=sharing
# Save to: /content/drive/MyDrive/mycoai/pretrained/vit_256_teacher.pth
```

**Step 2**: Run training
```python
!python /content/drive/MyDrive/mycoai/train_models_cellvit.py
```

**Output**:
- `weights/ViT_CellViT_finetuned.pth`
- `weights/ViT_CellViT_training_history.png`
- `weights/ViT_CellViT_history.json`

---

### Option 3: Self-Supervised SimCLR

```python
# Run two-stage training (longer but potentially best results)
!python /content/drive/MyDrive/mycoai/train_models_selfsupervised.py
```

**Output**:
- `weights/ResNet50_SimCLR_pretrained_encoder.pth` (after Stage 1)
- `weights/ResNet50_SimCLR_finetuned.pth` (after Stage 2)
- `weights/SimCLR_pretraining_loss.png`
- `weights/SimCLR_finetuning_history.png`
- JSON history files

---

## 📊 Results Comparison

After training all three approaches, compare results:

```python
import json
import matplotlib.pyplot as plt

# Load histories
with open('/content/drive/MyDrive/mycoai/weights/ResNet50_history.json') as f:
    imagenet_history = json.load(f)

with open('/content/drive/MyDrive/mycoai/weights/ViT_CellViT_history.json') as f:
    cellvit_history = json.load(f)

with open('/content/drive/MyDrive/mycoai/weights/SimCLR_finetuning_history.json') as f:
    simclr_history = json.load(f)

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(imagenet_history['val_accuracy'], label='ImageNet ResNet50', linewidth=2)
plt.plot(cellvit_history['val_accuracy'], label='CellViT ViT', linewidth=2)
plt.plot(simclr_history['val_accuracy'], label='SimCLR ResNet50', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(imagenet_history['val_loss'], label='ImageNet ResNet50', linewidth=2)
plt.plot(cellvit_history['val_loss'], label='CellViT ViT', linewidth=2)
plt.plot(simclr_history['val_loss'], label='SimCLR ResNet50', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/mycoai/results/comparison.png', dpi=300)
plt.show()

# Print final metrics
print("Final Validation Accuracies:")
print(f"ImageNet ResNet50: {max(imagenet_history['val_accuracy']):.4f}")
print(f"CellViT ViT: {max(cellvit_history['val_accuracy']):.4f}")
print(f"SimCLR ResNet50: {max(simclr_history['val_accuracy']):.4f}")
```

---

## 🔬 Technical Details

### Data Augmentation

**ImageNet & CellViT**:
- Random horizontal/vertical flips
- Random rotation (10-15°)
- Color jitter
- Resize to 256×256

**SimCLR (Stage 1 - Contrastive)**:
- Random resized crop (0.2-1.0 scale)
- Strong color jitter
- Gaussian blur
- Random grayscale
- Creates two augmented views per image

### Loss Functions

- **ImageNet & CellViT**: Cross-Entropy Loss
- **SimCLR Stage 1**: NT-Xent (Normalized Temperature-scaled Cross Entropy)
- **SimCLR Stage 2**: Cross-Entropy Loss

### Optimizers

- **ImageNet**: Adam (lr=0.0001)
- **CellViT**: AdamW (lr=0.0001, weight_decay=0.05)
- **SimCLR Pretrain**: Adam (lr=0.0003)
- **SimCLR Finetune**: Adam (lr=0.00001) - lower for pretrained model

### Early Stopping

All scripts use early stopping with patience=10 epochs to prevent overfitting.

---

## 💡 Recommendations

### When to use each approach:

1. **ImageNet (train_models.py)**
   - Quick baseline needed
   - Limited computational resources
   - Want to train multiple architectures quickly

2. **CellViT (train_models_cellvit.py)**
   - Have access to pretrained microscopy weights
   - Believe cell structure similarities help
   - Want better starting point than ImageNet

3. **SimCLR (train_models_selfsupervised.py)**
   - Want maximum performance
   - Have computational resources (GPU time)
   - Want to leverage all available images
   - Building production model

### Suggested Workflow:

1. Start with ImageNet baseline to establish performance floor
2. Try CellViT if pretrained weights are available
3. If performance is critical, invest time in SimCLR
4. Compare all three and use best performer for feature extraction

---

## 📈 Expected Results Summary

Based on similar tasks in literature:

| Metric | ImageNet | CellViT | SimCLR |
|--------|----------|---------|--------|
| **Validation Accuracy** | 70-85% | 75-90% | 75-95% |
| **Generalization** | Good | Better | Best |
| **Training Time** | Fastest | Medium | Slowest |
| **GPU Memory** | ~4GB | ~6GB | ~8GB |
| **Requires Pretrained Weights** | No | Yes (optional) | No |

---

## 🐛 Troubleshooting

### Out of Memory Errors
- Reduce batch size: `BATCH_SIZE = 8` instead of 16
- For SimCLR: `PRETRAIN_BATCH_SIZE = 32`

### Slow Training
- Increase `num_workers=4` in DataLoader (if CPU allows)
- Use mixed precision training (add to code if needed)

### Poor Convergence
- Reduce learning rate by 10x
- Increase patience for early stopping
- Check data augmentation isn't too aggressive

### Missing Files
- Verify hierarchical dataset structure exists
- Check strain_to_specy.csv has correct format
- Ensure segmented_image_metadata.json is present

---

## 📚 References

- **SimCLR**: [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709)
- **CellViT**: [Vision Transformers for Precise Cell Segmentation](https://arxiv.org/abs/2306.15350)
- **Transfer Learning**: [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)

---

## 📝 Notes

- All scripts save backbone weights WITHOUT classification head
- Use saved backbones in main feature extraction pipeline
- Classes are saved as numpy arrays for label encoding consistency
- Training histories saved as JSON for easy analysis
- Visualization plots automatically generated and saved

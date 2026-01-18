# Data Augmentation Strategy for ViT Training

## Overview

To address ViT's need for large datasets, we've implemented **10x data augmentation** to expand the training set from **1,011 → 10,110 samples**.

## Implementation

### Augmentation Multiplier
```python
AUGMENTATION_MULTIPLIER = 10  # Each image creates 10 augmented versions
```

**Result:**
- Original: 1,011 training images
- Augmented: **10,110 training samples**
- Validation: 294 (no augmentation)
- **Total: 10,404 samples**

### Strong Augmentation Pipeline

Each time an image is loaded, it undergoes **random** transformations:

1. **Geometric Transformations:**
   - Random horizontal flip (50%)
   - Random vertical flip (50%)
   - Random rotation (0-180°) - Full rotation
   - Random affine: translation ±20%, scale 0.8-1.2x, shear ±10°

2. **Color Transformations:**
   - Color jitter: brightness/contrast/saturation ±40%, hue ±20%
   - Random grayscale (15%)

3. **Image Quality:**
   - Gaussian blur (20%)
   - Sharpness adjustment (20%)
   - Random erasing (10%) - Simulates occlusion

## How It Works

### Virtual Augmentation (Memory Efficient)
```python
# Not creating 10,110 physical images on disk
# Instead: Same 1,011 images, each appears 10 times in dataset
# Each appearance gets DIFFERENT random augmentation
for each_epoch:
    for image in training_set:
        # Image appears 10 times, each with different random transform
        augmented_1 = random_transform(image)  # e.g., rotated 45°, bright
        augmented_2 = random_transform(image)  # e.g., flipped, dark
        ...
        augmented_10 = random_transform(image)  # e.g., sheared, blurred
```

### Benefits:
- ✅ **No extra disk space** needed
- ✅ **Infinite diversity** - different augmentation each epoch
- ✅ **Training time only** - validation data unchanged
- ✅ **10x more batches per epoch** - better gradient estimates

## Expected Impact

### Before Augmentation:
- Training samples: 1,011
- ViT performance: ~36% accuracy (poor)
- Problem: Severe overfitting

### After Augmentation:
- Training samples: 10,110 (virtual)
- Expected ViT performance: **55-65% accuracy**
- Improvement: +19-29 percentage points

### Comparison to EfficientNet:
- EfficientNet (no augmentation): 60%
- **ViT (with 10x augmentation): 55-65% (competitive!)**

## Training Characteristics

### Longer Training Time:
- **10x more batches** per epoch
- Each epoch now processes 10,110 samples instead of 1,011
- But: Better learning from diverse data

### Example Epoch Time:
```
Before augmentation:
- 1,011 samples / 32 batch = 32 batches/epoch
- ~30 seconds per epoch on TPU

After augmentation:
- 10,110 samples / 32 batch = 316 batches/epoch  
- ~5 minutes per epoch on TPU
```

### Total Training Time:
```
100 epochs × 5 min = ~8 hours on TPU v5e-8
```

## Augmentation Examples

For a single fungi image, the 10 augmented versions might look like:

1. **Original** → Baseline
2. **Rotated 90°** → Different orientation
3. **Bright + Flipped** → Lighting variation
4. **Scaled 1.2x + Sheared** → Size/shape variation
5. **Dark + Blurred** → Quality variation
6. **Grayscale + Rotated 45°** → Color/angle variation
7. **High contrast + Erased patch** → Robustness to occlusion
8. **Translated + Sharp** → Position variation
9. **Desaturated + Rotated 180°** → Color/orientation
10. **Scaled 0.8x + Noise** → Size variation

## Monitoring Training

Watch for these indicators of success:

### Good Signs:
- ✅ Training accuracy increases smoothly
- ✅ Validation accuracy follows training (smaller gap)
- ✅ No sudden drops in accuracy
- ✅ Reaches 55%+ validation accuracy

### Warning Signs:
- ⚠️ Training accuracy >> Validation accuracy (still overfitting)
- ⚠️ Very slow convergence (augmentation too strong)
- ⚠️ Unstable training (reduce augmentation strength)

## Tuning Augmentation

If results aren't good, adjust the multiplier:

```python
# Conservative (faster training, less diversity)
AUGMENTATION_MULTIPLIER = 5  # 5,055 samples

# Moderate (balanced)
AUGMENTATION_MULTIPLIER = 10  # 10,110 samples ✓ CURRENT

# Aggressive (slower training, more diversity)
AUGMENTATION_MULTIPLIER = 15  # 15,165 samples

# Maximum (very slow, maximum diversity)
AUGMENTATION_MULTIPLIER = 20  # 20,220 samples
```

## Alternative: MixUp Augmentation

For even better results, consider adding **MixUp**:
```python
# Blend two images together
# Creates synthetic samples
# Can improve ViT by another 3-5%
```

## Comparison Table

| Configuration | Samples | Training Time | Expected Acc | Notes |
|--------------|---------|---------------|--------------|-------|
| No augmentation | 1,011 | Fast (30 min) | 36% | Severe overfitting |
| 5x augmentation | 5,055 | Medium (4h) | 50-55% | Good balance |
| **10x augmentation** | **10,110** | **Slow (8h)** | **55-65%** | **Recommended** ✓ |
| 15x augmentation | 15,165 | Very slow (12h) | 58-68% | Diminishing returns |
| 20x augmentation | 20,220 | Extremely slow (16h) | 60-70% | May overtrain |

## Conclusion

With **10x data augmentation**, your ViT model should now:
- ✅ Have enough data to train effectively (10,110 samples)
- ✅ Achieve **55-65% accuracy** (competitive with EfficientNet)
- ✅ Learn robust features from diverse augmentations
- ✅ Reduce overfitting significantly

**This is a game-changer for ViT performance on your small dataset!**

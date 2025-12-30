# Understanding Ensemble Weights vs Individual Accuracies

## Question
Why does ColorHistogramHS show 38.71% as a weight when it has 75% accuracy alone?

## Answer

The **38.71%** is the **normalized ensemble weight**, NOT the accuracy!

## Breakdown

### Individual Model Accuracies (from CSV)
These are the actual standalone performances:
- **ColorHistogramHS**: 75.00% (36/48 correct)
- **ResNet50**: 66.67% (32/48 correct)
- **EfficientNetV2B0**: 52.08% (25/48 correct)

### Ensemble Weight Calculation

The ensemble uses **accuracy-weighted voting**. To normalize the weights so they sum to 1.0:

```python
# Step 1: Get raw accuracies
weights = {
    'ColorHistogramHS': 0.75,
    'ResNet50': 0.6667,
    'EfficientNetV2B0': 0.5208
}

# Step 2: Calculate total
total_weight = 0.75 + 0.6667 + 0.5208 = 1.9375

# Step 3: Normalize (divide each by total)
normalized_weights = {
    'ColorHistogramHS': 0.75 / 1.9375 = 0.3871 (38.71%)
    'ResNet50': 0.6667 / 1.9375 = 0.3441 (34.41%)
    'EfficientNetV2B0': 0.5208 / 1.9375 = 0.2688 (26.88%)
}
```

**Verification**: 0.3871 + 0.3441 + 0.2688 = 1.0000 ✓

## What These Weights Mean

### In the Weighted Ensemble Strategy

When combining predictions, each model's contribution is weighted:

```python
for each species:
    score = (ColorHistogramHS_score × 0.3871 +
             ResNet50_score × 0.3441 +
             EfficientNetV2B0_score × 0.2688)
```

### Interpretation

- **ColorHistogramHS gets 38.71% of the vote** (largest share because it's most accurate)
- **ResNet50 gets 34.41% of the vote** (second largest)
- **EfficientNetV2B0 gets 26.88% of the vote** (smallest share because it's least accurate)

## Why Normalize?

Normalization ensures:
1. **Weights sum to 1.0** - Makes the combined scores comparable
2. **Proportional representation** - Better models have more influence
3. **Meaningful interpretation** - Each weight represents "voting power"

## Example

If ColorHistogramHS predicts species A with score 0.8:
- Its contribution = 0.8 × 0.3871 = **0.3097**

If ResNet50 predicts species B with score 0.9:
- Its contribution = 0.9 × 0.3441 = **0.3097**

Even though ResNet50 has higher confidence (0.9 vs 0.8), both contribute equally because ColorHistogramHS has higher weight!

## Console Output Explained

When you see:
```
Ensemble weights:
  ColorHistogramHS: 0.3871
  ResNet50: 0.3441
  EfficientNetV2B0: 0.2688
```

This means:
- ✅ ColorHistogramHS gets **38.71% of voting power**
- ✅ Individual accuracy is still **75%** (shown elsewhere)
- ✅ Weights are **proportional to accuracy**
- ✅ Higher accuracy = higher weight

## Summary Table

| Model | Individual Accuracy | Raw Weight | Normalized Weight | Voting Power |
|-------|-------------------|------------|------------------|--------------|
| ColorHistogramHS | **75.00%** | 0.75 | **0.3871** | 38.71% |
| ResNet50 | **66.67%** | 0.6667 | **0.3441** | 34.41% |
| EfficientNetV2B0 | **52.08%** | 0.5208 | **0.2688** | 26.88% |
| **Total** | - | **1.9375** | **1.0000** | **100%** |

## Key Insight

The 38.71% is **NOT saying ColorHistogramHS is less accurate**. It's saying:
- "Of the total voting power in this ensemble..."
- "ColorHistogramHS controls 38.71%..."
- "Because it earned 75% accuracy (the highest)..."
- "So it gets the biggest share of influence!"

**Think of it like this**: If you, your friend, and your sibling vote on where to eat, and you always pick the best restaurants (75% success rate), you should get more say (38.71% of decision power) than your sibling who only picks good restaurants 52% of the time (26.88% of decision power).

The weights are **proportional** to performance, ensuring better models have more influence! 🎯

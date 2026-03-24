# Ensemble Strategy Reference

## Aggregation Strategies

| Strategy | Description |
|----------|-------------|
| `weighted` / `score` / `avg` | Score-weighted: each model's vote is proportional to its overall accuracy |
| `uni` | Uniform: all models vote equally |
| `manual_weighted` | Per-species weights from `species_weights.json` |

## Accuracy-Based Weights (auto-computed)

The `weighted` strategy normalizes each model's accuracy so all weights sum to 1.0:

```
normalized_weight = model_accuracy / sum(all_accuracies)
```

Example with 3 models:
| Model | Accuracy | Normalized Weight |
|-------|----------|------------------|
| ColorHistogramHS | 75.00% | 38.71% |
| ResNet50 | 66.67% | 34.41% |
| EfficientNetV2B0 | 52.08% | 26.88% |

Higher accuracy → higher voting power. The 38.71% is **not** the model's accuracy — it's its share of decision power.

## Manual Per-Species Weights (`species_weights.json`)

Allows fine-grained control when specific models excel or fail for specific species:

```json
{
  "weights": {
    "Penicillium cyclopium": {
      "ColorHistogramHS": 0.3,
      "ResNet50": 1.0,
      "EfficientNetB1_finetuned": 0.9
    },
    "default": {
      "ColorHistogramHS": 1.0,
      "ResNet50": 1.0,
      "EfficientNetB1_finetuned": 1.0
    }
  }
}
```

- `1.0` = standard trust; `< 1.0` = reduce influence; `0.0` = disable for this species
- Species not listed fall back to `"default"` weights
- Score combination: `sum(model_score × species_weight[species][model])`

### Tuning workflow

1. Run evaluation and inspect `results/` for per-species failure patterns
2. Lower weights for consistently failing models per species
3. Re-run and compare against `weighted` / `uni` baselines
4. Avoid over-tuning on the test set

## Comparison

| Strategy | Pros | Cons |
|----------|------|------|
| `weighted` | Automatic, data-driven | Ignores species-specific strengths |
| `uni` | Simple, no tuning | Treats all models equally |
| `manual_weighted` | Fine-grained domain control | Requires manual tuning, overfitting risk |

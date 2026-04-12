# Ensemble Strategy Reference

> **Note:** The `weighted` strategy was previously called `avg` in older code. All new code should use `weighted` (score-weighted) and `uni` (uniform count). See [`docs/TERMINOLOGY.md`](docs/TERMINOLOGY.md) for full definitions.

## Aggregation Strategies

| Strategy | Aliases | Description |
|----------|---------|-------------|
| `weighted` | `score` | Score-weighted by cosine similarity — high-similarity neighbours have more influence |
| `uni` | `uniform`, `count` | Uniform count — each neighbour contributes equally (1/k) |
| `manual_weighted` | — | Per-species weights from `species_weights.json` on top of `weighted` |

## Score-Weighted Math (`weighted`)

```
species_score[s] = Σ  cosine_similarity(neighbour_i)   for all neighbours where neighbour_i.species == s
prediction = argmax_s species_score[s]
```

## Uniform Count Math (`uni`)

```
species_count[s] = Σ  1   for all neighbours where neighbour_i.species == s
prediction = argmax_s species_count[s]
```

## Accuracy-Based Weights

When combining multiple extractors:
```
normalized_weight = model_accuracy / Σ(all_model_accuracies)
```

## Manual Per-Species Weights (`species_weights.json`)

Allows fine-grained control per species per extractor:
- `1.0` = standard trust
- `< 1.0` = reduced influence
- `0.0` = completely disable that model for this species

Example:
```json
{
  "Penicillium commune": {
    "ResNet50_finetuned": 1.2,
    "EfficientNetB1_finetuned": 0.8,
    "ResNet50_finetuned": 0.0
  }
}
```

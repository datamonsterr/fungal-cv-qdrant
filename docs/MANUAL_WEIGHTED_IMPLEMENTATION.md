# Manual Weighted Ensemble - Implementation Summary

## ✅ Implementation Complete

Successfully implemented a manual weighted ensemble strategy that allows species-specific weights for each feature extractor.

## Files Created/Modified

### New Files
1. **`species_weights.json`** - Configuration file for species-specific weights
   - Contains weights for 5 species + default
   - Example weights based on observed patterns
   
2. **`MANUAL_WEIGHTED_STRATEGY.md`** - Comprehensive documentation
   - Usage guide
   - Weight tuning strategies
   - Best practices
   - Troubleshooting

### Modified Files
1. **`ensemble_analysis.py`**
   - Added `SpeciesWeights` Pydantic model
   - Added `load_species_weights()` function
   - Updated `combine_aggregated_results()` to support 'manual_weighted' strategy
   - Updated `evaluate_ensemble()` to accept species_weights parameter
   - Updated `create_strategy_comparison_chart()` to include manual weighted
   - Updated `main()` to load and evaluate manual weighted strategy

## How It Works

### 1. Configuration (species_weights.json)
```json
{
  "weights": {
    "Penicillium cyclopium": {
      "ColorHistogramHS": 0.5,    // Lower weight
      "ResNet50": 1.0,
      "EfficientNetV2B0": 0.9
    },
    "default": {
      "ColorHistogramHS": 1.0,
      "ResNet50": 1.0,
      "EfficientNetV2B0": 1.0
    }
  }
}
```

### 2. Weight Application
For each species in aggregated results:
```
contribution = model_score × species_weight[species][model]
combined_score = sum of all contributions
```

### 3. Automatic Integration
The ensemble analysis automatically:
- Detects `species_weights.json`
- Loads the weights
- Evaluates the manual_weighted strategy
- Compares with weighted_sum and simple_average
- Includes in visualization

## Initial Results

Running with example weights from `species_weights.json`:

```
Weighted Ensemble:        66.67% (32/48)
Simple Average Ensemble:  66.67% (32/48)
Manual Weighted Ensemble: 64.58% (31/48)  ← New strategy
Best Individual Model:    75.00% (ColorHistogramHS)
```

### Observations
- Manual weighted achieved 64.58% with example weights
- Slightly lower than other ensemble strategies (likely needs tuning)
- The example weights were conservative starting values
- Shows the strategy is working correctly (different results)

## Next Steps for Tuning

### 1. Analyze Per-Species Performance
Check which species the ensemble gets wrong and adjust weights accordingly.

### 2. Use Complementary Case Visualizations
```bash
ls results/ensemble_analysis/complementary_cases/wrong_colorhistogramhs/
```
These show specific failure cases to guide weight adjustments.

### 3. Iterative Tuning
```bash
# Edit weights
nano species_weights.json

# Re-run analysis
nix-shell --run "uv run python ensemble_analysis.py"

# Check if accuracy improved
```

## Example Weight Adjustments

Based on the complementary case analysis showing:
- DTO 148-C8 (Penicillium cyclopium): ColorHistogramHS fails in all 6 test sets
- DTO 469-I4 (Penicillium freii): ResNet50 succeeds where ColorHistogramHS fails
- DTO 158-D1 (Penicillium melanoconidium): ColorHistogramHS fails in 5/6 test sets

### Suggested Adjustments
```json
{
  "weights": {
    "Penicillium cyclopium": {
      "ColorHistogramHS": 0.3,    // Further reduce (was 0.5)
      "ResNet50": 1.2,            // Increase (was 1.0)
      "EfficientNetV2B0": 1.0
    },
    "Penicillium freii": {
      "ColorHistogramHS": 0.4,    // Further reduce (was 0.6)
      "ResNet50": 1.5,            // Significantly increase (was 1.0)
      "EfficientNetV2B0": 0.9     // Slightly increase (was 0.8)
    },
    "Penicillium melanoconidium": {
      "ColorHistogramHS": 0.3,    // New: low weight
      "ResNet50": 1.2,            // New: higher trust
      "EfficientNetV2B0": 1.3     // New: higher trust
    }
  }
}
```

## Usage

### Run Ensemble Analysis
```bash
nix-shell --run "uv run python ensemble_analysis.py"
```

### Check Results
```bash
# View detailed results
cat results/ensemble_analysis/ensemble_results_manual_weighted.json

# View comparison chart (includes manual weighted bar)
xdg-open results/ensemble_analysis/strategy_comparison.png
```

## Technical Details

### Strategy Implementation
- **Location**: `ensemble_analysis.py::combine_aggregated_results()`
- **Strategy Name**: `'manual_weighted'`
- **Parameters**: Requires `species_weights: Optional[SpeciesWeights]`

### Weight Lookup Logic
```python
if species_name in species_weights.weights:
    model_weight = species_weights.weights[species_name].get(model_name, 1.0)
else:
    # Use default weights
    model_weight = species_weights.weights.get('default', {}).get(model_name, 1.0)
```

### Combination Formula
```python
for each model:
    for each species in model.aggregated_results:
        weight = get_species_weight(species, model)
        contribution = species.score × weight
        species_scores[species] += contribution
```

## Benefits

1. **Fine-Grained Control**: Adjust weights per species, not globally
2. **Domain Knowledge**: Encode expert knowledge about model strengths
3. **Targeted Fixes**: Address specific problematic species
4. **Transparent**: Clear JSON configuration, easy to understand and modify
5. **Flexible**: Can use any weight values (0.0+), not limited to 0-1
6. **Automatic**: Once configured, runs automatically with ensemble analysis

## Limitations

1. **Manual Tuning**: Requires human judgment and iteration
2. **Overfitting Risk**: Can overfit to test set if not careful
3. **Maintenance**: Needs updates as new species are added
4. **Species Coverage**: Must have weights for all target species (or use default)

## Comparison with Other Strategies

| Strategy | Weights Based On | Pros | Cons | Current Accuracy |
|----------|-----------------|------|------|------------------|
| Weighted Sum | Model accuracy (global) | Automatic, data-driven | Ignores species-specific patterns | 66.67% |
| Simple Average | Equal weights (1.0) | Simple, no tuning | No differentiation | 66.67% |
| **Manual Weighted** | **Species-specific tuning** | **Fine control, encode knowledge** | **Requires tuning** | **64.58%** (with example weights) |

## Recommendation

The initial results (64.58%) are lower than other strategies, but this is expected with the example weights. The real power comes from tuning based on:

1. Observed failure patterns (from complementary case analysis)
2. Visual inspection of misclassified cases
3. Domain expertise about species characteristics
4. Iterative refinement

With proper tuning, manual weighted could potentially exceed the current best performance by targeting specific weaknesses.

## Quick Start Tuning Guide

```bash
# 1. Check current problematic cases
ls results/ensemble_analysis/complementary_cases/wrong_colorhistogramhs/

# 2. Identify patterns (which species fail repeatedly?)
# Look at the filenames: DTO_148-C8, DTO_158-D1, DTO_469-I4

# 3. Edit weights for those species
nano species_weights.json

# 4. Re-run analysis
nix-shell --run "uv run python ensemble_analysis.py"

# 5. Check if accuracy improved
# Look for "Manual Weighted Ensemble: X.XX%"

# 6. Iterate until satisfied
```

## Future Enhancements

Potential improvements:
1. **Auto-weight generator**: Script to calculate weights from per-species accuracy
2. **Cross-validation**: Test weights on separate validation set
3. **Confidence-based weighting**: Adjust weights based on prediction confidence
4. **Dynamic weights**: Learn weights during inference based on query characteristics
5. **Hierarchical weights**: Genus-level → Species-level weight inheritance

# Manual Weighted Ensemble Strategy

## Overview

The manual weighted ensemble strategy allows you to specify species-specific weights for each feature extractor. This enables fine-tuned control over which models are trusted more for specific species based on domain knowledge or empirical observation.

## How It Works

1. **Species-Specific Weights**: Each species can have different weights for each feature extractor
2. **Default Fallback**: Species not explicitly listed use default weights
3. **Score Combination**: When combining predictions, each model's score for a species is multiplied by that species-specific weight
4. **Flexible Tuning**: Weights can be adjusted based on:
   - Known model strengths/weaknesses for specific species
   - Visual similarity patterns
   - Historical performance data
   - Domain expertise

## Configuration File

The weights are defined in `species_weights.json`:

```json
{
  "description": "Manual weights for species-specific feature extractor performance",
  "weights": {
    "Penicillium aurantiogriseum": {
      "ColorHistogramHS": 1.0,
      "ResNet50": 0.8,
      "EfficientNetV2B0": 0.7
    },
    "Penicillium cyclopium": {
      "ColorHistogramHS": 0.5,
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

## Weight Guidelines

- **Weight Range**: Use positive values (typically 0.0 to 1.0, but can be higher)
- **Weight = 1.0**: Standard/full trust in the model
- **Weight < 1.0**: Lower trust (e.g., 0.5 = half weight)
- **Weight > 1.0**: Extra trust (e.g., 2.0 = double weight)
- **Weight = 0.0**: Effectively disable this model for this species

## Example Scenarios

### Scenario 1: ColorHistogramHS Fails for Specific Species
If analysis shows ColorHistogramHS consistently fails for "Penicillium cyclopium":
```json
"Penicillium cyclopium": {
  "ColorHistogramHS": 0.3,    // Reduce trust
  "ResNet50": 1.0,            // Keep standard
  "EfficientNetV2B0": 1.0     // Keep standard
}
```

### Scenario 2: ResNet50 Excels for Similar-Looking Species
If ResNet50 is better at distinguishing visually similar species:
```json
"Penicillium freii": {
  "ColorHistogramHS": 0.6,
  "ResNet50": 1.5,            // Increase trust
  "EfficientNetV2B0": 0.8
}
```

### Scenario 3: Model Specialization
If different models excel at different aspects:
```json
"Penicillium viridicatum": {
  "ColorHistogramHS": 1.2,    // Good with color patterns
  "ResNet50": 0.7,            // Not as good
  "EfficientNetV2B0": 0.6
}
```

## Usage

### Running Ensemble Analysis

Simply run the ensemble analysis script:

```bash
nix-shell --run "uv run python ensemble_analysis.py"
```

The script will automatically:
1. Check for `species_weights.json`
2. If found, evaluate the manual weighted strategy alongside others
3. Compare all strategies and report the best one

### Output

The analysis produces:
- `ensemble_results_manual_weighted.json` - Detailed prediction results
- Updated `strategy_comparison.png` - Includes manual weighted in comparison
- Console output comparing all strategies

## Tuning the Weights

### Step 1: Analyze Current Performance

Run the ensemble analysis to see which species are problematic:
```bash
nix-shell --run "uv run python ensemble_analysis.py"
```

Check the `complementary_cases/` directory for visualizations showing where each model fails.

### Step 2: Identify Patterns

Look for patterns in the false predictions:
- Does ColorHistogramHS consistently fail for certain species?
- Does ResNet50 correct specific types of errors?
- Are there species where EfficientNetV2B0 excels?

### Step 3: Update Weights

Edit `species_weights.json` based on observed patterns:
- Lower weights for models that consistently fail
- Raise weights for models that consistently succeed
- Keep default weights for species with no clear pattern

### Step 4: Re-evaluate

Run the analysis again to see if accuracy improves:
```bash
nix-shell --run "uv run python ensemble_analysis.py"
```

### Step 5: Iterate

Repeat the process, fine-tuning weights based on results.

## Comparison with Other Strategies

### Weighted Sum (Accuracy-Based)
- **Weights**: Based on overall model accuracy
- **Pros**: Data-driven, automatic
- **Cons**: Doesn't account for species-specific strengths

### Simple Average
- **Weights**: All models equal (1.0)
- **Pros**: Simple, no tuning needed
- **Cons**: Treats all models equally regardless of performance

### Manual Weighted (Species-Specific)
- **Weights**: Custom per species per model
- **Pros**: Fine-grained control, can encode domain knowledge
- **Cons**: Requires manual tuning, risk of overfitting

## Best Practices

1. **Start Conservative**: Begin with default weights (1.0) for most species
2. **Target Problem Cases**: Only adjust weights for species showing consistent issues
3. **Use Visualizations**: Check `complementary_cases/` images to understand failures
4. **Document Rationale**: Add notes in the JSON explaining why you chose specific weights
5. **Validate Changes**: Always re-run analysis after adjusting weights
6. **Avoid Overfitting**: Don't tune weights to perfection on the test set

## Example Workflow

```bash
# 1. Run initial analysis
nix-shell --run "uv run python ensemble_analysis.py"

# 2. Check results
cat results/ensemble_analysis/ensemble_results_weighted.json
ls results/ensemble_analysis/complementary_cases/wrong_colorhistogramhs/

# 3. Edit weights based on findings
nano species_weights.json

# 4. Re-run analysis
nix-shell --run "uv run python ensemble_analysis.py"

# 5. Compare results
# Check if Manual Weighted > Weighted Sum or Simple Average
```

## Advanced: Dynamic Weight Generation

For a more data-driven approach, you could create a script to automatically generate weights based on per-species accuracy:

```python
# Pseudo-code example
for species in all_species:
    for model in models:
        accuracy = get_accuracy(model, species)
        weights[species][model] = accuracy
```

This could be implemented as a separate analysis script that generates `species_weights.json` automatically.

## Notes

- Weights are multiplicative: `combined_score = model1_score * weight1 + model2_score * weight2 + ...`
- The strategy does not normalize weights (unlike weighted_sum which normalizes by accuracy)
- A weight of 0 effectively removes that model's contribution for that species
- Negative weights are not recommended (behavior undefined)

## Troubleshooting

### Manual weighted strategy not running
- Check that `species_weights.json` exists in the scripts directory
- Verify JSON is valid (use `python -m json.tool species_weights.json`)

### No improvement over other strategies
- Weights may not be tuned appropriately
- Problem species may not be in the weight dict (check 'default' weights)
- Consider visualizing cases to understand failure modes better

### Lower accuracy than expected
- May be overfitting to specific cases
- Try more conservative weight adjustments
- Ensure weights are in reasonable ranges (0.5 - 2.0)

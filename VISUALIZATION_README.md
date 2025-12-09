# Prediction Visualization Documentation

## Overview

The `visualize_prediction.py` module provides comprehensive visualization for species prediction results, showing K nearest neighbors per environment in a clear grid layout.

## Features

### Multi-Environment Grid Layout
- **Query Images**: First column shows query images from each environment
- **Neighbors per Environment**: Each row displays K neighbors for that environment's query
- **Color Coding**:
  - **Green borders**: Species matches ground truth
  - **Red borders**: Species doesn't match ground truth (false predictions)
  
### Comprehensive Metadata Display
- **Header Section**:
  - Prediction status (CORRECT/FALSE)
  - Ground truth species
  - Predicted species with confidence
  - Strategy information (E1/E2/E3, AVG/UNI)
  - Feature extractor and K value
  
- **Per Image**:
  - Environment name
  - Species name
  - Similarity score
  - Image ID
  - Rank number (for neighbors)

## Function Reference

### `visualize_prediction_by_environment()`

Creates a single visualization for one prediction result.

**Parameters:**
- `prediction_result`: Dict from `predict_segment_group()`
- `segmented_image_dir`: Path to segmented images directory
- `output_path`: Where to save the visualization
- `k`: Number of neighbors to display (default: 7)
- `thumbnail_size`: Image size (default: (150, 150))
- `text_color`: RGB tuple for text (default: black)
- `bg_color`: RGB tuple for background (default: white)
- `border_width`: Width of colored borders (default: 8)

**Example:**
```python
from visualize_prediction import visualize_prediction_by_environment

visualize_prediction_by_environment(
    prediction_result=result,
    segmented_image_dir="../Dataset/myco_segmented",
    output_path="./visualization.jpg",
    k=7
)
```

### `batch_visualize_predictions()`

Creates visualizations for multiple prediction results.

**Parameters:**
- `prediction_results`: List of prediction result dicts
- `segmented_image_dir`: Path to segmented images directory
- `output_dir`: Directory to save visualizations
- `k`: Number of neighbors to display (default: 7)
- `filter_correct`: 
  - `True`: Only correct predictions
  - `False`: Only false predictions
  - `None`: All predictions
- `max_visualizations`: Maximum number to create

**Example:**
```python
from visualize_prediction import batch_visualize_predictions

output_paths = batch_visualize_predictions(
    prediction_results=results,
    segmented_image_dir="../Dataset/myco_segmented",
    output_dir="./visualizations",
    k=7,
    filter_correct=False,  # Only false predictions
    max_visualizations=10
)
```

## Usage Examples

### Example 1: Visualize False Predictions from Evaluation

```python
from qdrant_client import QdrantClient
from evaluate_species import run_species_evaluation
from feature_extractors import ResNet50Extractor
from visualize_prediction import batch_visualize_predictions

# Run evaluation
client = QdrantClient(url="http://localhost:6333")
results, _ = run_species_evaluation(
    client=client,
    collection_name="myco_features",
    feature_extractor=ResNet50Extractor(),
    k=7,
    environment=None,  # E1 strategy
    strategy="avg",
    output_dir="./results"
)

# Visualize false predictions
batch_visualize_predictions(
    prediction_results=results,
    segmented_image_dir="../Dataset/myco_segmented",
    output_dir="./results/false_predictions",
    k=7,
    filter_correct=False,
    max_visualizations=10
)
```

### Example 2: Compare Different Strategies

```python
from feature_extractors import HOGExtractor

strategies = {
    "E1": None,
    "E2": "all",
    "E3_CYA": "CYA"
}

for name, env in strategies.items():
    results, _ = run_species_evaluation(
        client=client,
        collection_name="myco_features",
        feature_extractor=HOGExtractor(),
        k=7,
        environment=env,
        strategy="avg",
        output_dir=f"./results/{name}"
    )
    
    batch_visualize_predictions(
        prediction_results=results,
        segmented_image_dir="../Dataset/myco_segmented",
        output_dir=f"./results/viz_{name}",
        k=7,
        filter_correct=False
    )
```

### Example 3: Command-Line Usage

```bash
# Visualize from saved JSON results
python visualize_prediction.py \
    --result-file results/predictions.json \
    --segmented-dir ../Dataset/myco_segmented \
    --output-dir ./visualizations \
    --k 7 \
    --filter-false \
    --max-viz 10
```

### Example 4: Single Strain Visualization

```python
from evaluate_species import predict_segment_group, collect_testset

# Collect test set for specific strain
test_sets = collect_testset(
    client=client,
    collection_name="myco_features",
    strain="DTO 123-A1",
    environment_strategy="E1"
)

# Predict
result = predict_segment_group(
    client=client,
    collection_name="myco_features",
    test_group=test_sets[0],
    strain="DTO 123-A1",
    feature_extractor=ResNet50Extractor(),
    k=7,
    environment=None,
    strategy="avg"
)

# Visualize
visualize_prediction_by_environment(
    prediction_result=result,
    segmented_image_dir="../Dataset/myco_segmented",
    output_path="./viz_DTO_123-A1.jpg",
    k=7
)
```

## Layout Description

### Grid Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  CORRECT/FALSE PREDICTION                                       │
│  Ground Truth: Species A                                        │
│  Predicted: Species B (Confidence: 0.875)                      │
│  Strategy: E1 | Aggregation: AVG | Feature: resnet50 | K=7    │
├─────────────────────────────────────────────────────────────────┤
│ [Query] [Neighbor #1] [Neighbor #2] ... [Neighbor #7]          │  ← Environment 1
│  CYA     CYA          CYA                CYA                    │
│         Species A     Species B          Species A              │
├─────────────────────────────────────────────────────────────────┤
│ [Query] [Neighbor #1] [Neighbor #2] ... [Neighbor #7]          │  ← Environment 2
│  MEA     MEA          MEA                MEA                    │
│         Species A     Species A          Species C              │
├─────────────────────────────────────────────────────────────────┤
│   ... (one row per environment) ...                             │
└─────────────────────────────────────────────────────────────────┘
```

### Color Coding

- **Query Image Border**:
  - Green: Overall prediction is correct
  - Red: Overall prediction is false

- **Neighbor Image Border**:
  - Green: Neighbor's species matches ground truth
  - Red: Neighbor's species doesn't match ground truth

## Strategy Explanations

### E1 (Same Environment)
- Creates 6 test sets, each with one image per environment
- Each query searches within its own environment
- Visualization shows neighbors from same environment per row
- Good for evaluating environment-specific performance

### E2 (All Environments)
- Creates 6 test sets, each with one image per environment
- Each query searches across ALL environments
- Visualization shows neighbors from any environment per row
- Good for evaluating cross-environment generalization

### E3 (Specific Environment)
- Creates 6 test sets from one specific environment
- Each test set has 1 image from that environment
- Queries search within that environment only
- Visualization shows only one row (single environment)
- Good for evaluating performance on specific growth conditions

## Integration with Comprehensive Evaluation

The visualization works seamlessly with `run_comprehensive_eval.py`:

```python
# In your comprehensive evaluation
from visualize_prediction import batch_visualize_predictions

# After running evaluation
results = run_comprehensive_evaluation(
    client=client,
    collection_name="myco_features",
    k=7,
    output_dir="./results"
)

# Visualize false predictions for each configuration
for config, config_results in results.items():
    batch_visualize_predictions(
        prediction_results=config_results,
        segmented_image_dir="../Dataset/myco_segmented",
        output_dir=f"./results/{config}/visualizations",
        k=7,
        filter_correct=False,
        max_visualizations=5
    )
```

## Output Files

Visualizations are saved as JPEG files with naming convention:
- `{number}_{strain_name}_{status}.jpg`
- Example: `001_DTO_123-A1_false.jpg`

## Performance Tips

1. **Limit visualizations**: Use `max_visualizations` parameter for large result sets
2. **Filter strategically**: Use `filter_correct` to focus on false predictions
3. **Adjust K**: Lower K values create smaller, faster visualizations
4. **Image size**: Reduce `thumbnail_size` for faster processing

## Troubleshooting

### Missing Images
**Problem**: Warning "Failed to load image"  
**Solution**: Verify `segmented_image_dir` path and image file names

### Font Issues
**Problem**: Text appears as boxes  
**Solution**: Fonts fallback to default automatically, but install Liberation fonts for best results

### Memory Issues
**Problem**: Out of memory with many visualizations  
**Solution**: Use `max_visualizations` parameter and process in batches

## See Also

- `evaluate_species.py`: Main evaluation script
- `query_utils.py`: Query and visualization utilities
- `run_comprehensive_eval.py`: Comprehensive evaluation runner
- `example_visualize_prediction.py`: Working examples

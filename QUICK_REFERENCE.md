# Quick Reference: visualize_prediction.py

## Import
```python
from visualize_prediction import visualize_prediction_by_environment, batch_visualize_predictions
```

## Single Visualization
```python
visualize_prediction_by_environment(
    prediction_result=result,           # From predict_segment_group()
    segmented_image_dir="../Dataset/myco_segmented",
    output_path="./viz.jpg",
    k=7                                 # Number of neighbors
)
```

## Batch Visualization
```python
# All predictions
batch_visualize_predictions(
    prediction_results=results,
    segmented_image_dir="../Dataset/myco_segmented",
    output_dir="./visualizations",
    k=7
)

# Only false predictions
batch_visualize_predictions(
    prediction_results=results,
    segmented_image_dir="../Dataset/myco_segmented",
    output_dir="./visualizations_false",
    k=7,
    filter_correct=False,               # False = false predictions
    max_visualizations=10               # Limit output
)
```

## With Evaluation
```python
from evaluate_species import run_species_evaluation
from feature_extractors import ResNet50Extractor

# Run evaluation
results, _ = run_species_evaluation(
    client=client,
    collection_name="myco_features",
    feature_extractor=ResNet50Extractor(),
    k=7,
    environment=None,                    # E1
    strategy="avg",
    output_dir="./results"
)

# Visualize
batch_visualize_predictions(
    prediction_results=results,
    segmented_image_dir="../Dataset/myco_segmented",
    output_dir="./results/visualizations",
    k=7,
    filter_correct=False
)
```

## Command Line
```bash
# From JSON file
python visualize_prediction.py \
    --result-file results.json \
    --segmented-dir ../Dataset/myco_segmented \
    --output-dir ./viz \
    --k 7 \
    --filter-false

# Quick test
python test_visualization.py

# Examples
python example_visualize_prediction.py --example batch
```

## Color Coding
- **Query Border**: Green (correct) / Red (false)
- **Neighbor Border**: Green (matches ground truth) / Red (doesn't match)

## Layout
```
Header: Status, Ground Truth, Predicted, Strategy
─────────────────────────────────────────────────
Row 1: [Query CYA] [N1] [N2] [N3] [N4] [N5] [N6] [N7]
Row 2: [Query MEA] [N1] [N2] [N3] [N4] [N5] [N6] [N7]
Row 3: [Query YES] [N1] [N2] [N3] [N4] [N5] [N6] [N7]
...
```

## Filter Options
```python
filter_correct=None   # All predictions
filter_correct=True   # Only correct
filter_correct=False  # Only false
```

## Strategies
- **E1**: `environment=None` → Same environment per query
- **E2**: `environment="all"` → All environments
- **E3**: `environment="CYA"` → Specific environment

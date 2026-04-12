# EDA: Diverse Data Reorganization

## Overview

Exploratory data analysis of new fungal species images from `Dataset/new_data` and reorganization into hierarchical structure at `Dataset/diverse_data/`.

## Source Data

- **Source:** `Dataset/new_data/` organized by letter ranges (A-L, M-R, S-Z)
- **Total images:** 652
- **Species count:** 45

## Target Structure

```
Dataset/diverse_data/
└── images/
    └── {species}/
        └── {environment}/
            ├── {Strain}_{angle}_{id}.jpg          (preprocessed image)
            ├── {Strain}_{angle}_{id}_bboxes.jpg   (with bbox visualization)
            └── {Strain}_{angle}_{id}_seg{n}.jpg   (segmented colonies)
└── diverse_data_metadata.json (with bbox coordinates and segment paths)
```

## Processing Results

| Metric | Count |
|--------|-------|
| Images processed | 651 |
| Bbox visualization images | 651 |
| Segmented colonies | 1,037 |
| Corrupted source files skipped | 1 |

## Environment Distribution

| Environment | Samples |
|-------------|---------|
| CYA | 146 |
| MEA | 146 |
| YES | 116 |
| DG18 | 122 |
| CREA | 108 |
| OA | 10 |
| CYAS | 2 |
| UNKNOWN | 2 |

## JSON Metadata Schema

Each image entry includes:
- `id`: Unique identifier
- `file_path`: Path to preprocessed image
- `bboxes_image`: Path to image with bounding boxes drawn
- `data`: Object containing:
  - `species`, `strain`, `environment`, `angle`
  - `bboxes`: Array of `{xmin, ymin, xmax, ymax}` objects
  - `num_colonies`: Number of colonies detected
  - `segment_paths`: Array of paths to segmented colony images

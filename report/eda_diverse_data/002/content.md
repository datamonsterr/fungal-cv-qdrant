# EDA: Canonical Dataset Restructure and Analysis

## Overview

Exploratory data analysis of the restructured canonical dataset for MycoAI fungal species project. Two source collections (curated_primary, incoming_low_quality) mapped to unified prepared hierarchy.

## Source Collections

| Metric | curated_primary | incoming_low_quality |
|--------|----------------|---------------------|
| Images | 435 | 577 |
| Species | 8 | 65 |
| Strains | 31 | 1 (all unknown) |
| Environments | 7 | 1 (all unknown) |
| Parse success | 100% | 0% (fallback) |

## Key Findings

1. **Total**: 1,012 images, 73 species, 32 strains, 8 environments
2. **Curated**: Metadata-rich, 7 environments balanced (60-64 each), 4-7 strains/species. P. cyclopium has 1 strain (limitation).
3. **Incoming**: Species diversity (65), but all metadata unresolved. Needs enrichment.
4. **Pipeline**: Unified src/prepare/dataset.py replaces reformat_dataset.py + reformat_dataset_yolo.py. Dual segmentation (KMeans + Contour).
5. **Consumers fixed**: Feature extractors, Qdrant upload, retrieval eval all use segment_path from metadata.

## Figures

- collection_distribution.png: Donut chart of 435 vs 577 images
- curated_species.png: 8 Penicillium species distribution
- incoming_species.png: Top 20 incoming species distribution
- curated_environments.png: 7 environment distribution
- curated_strains.png: 31 strain distribution
- parse_status.png: Metadata parse status by collection
- dataset_pipeline.png: Pipeline architecture diagram

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MYCOAI_ROOT | Auto-detected | Workspace root |
| DATASET_ROOT | $MYCOAI_ROOT/Dataset | Shared dataset folder |
| WEIGHTS_DIR | $MYCOAI_ROOT/weights | Model checkpoints |
| RESULTS_DIR | $MYCOAI_ROOT/results | Experiment outputs |

#!/usr/bin/env fish

source .venv/bin/activate.fish

echo "Running feature extraction generate_features"
python -m src.experiments.feature_extraction.generate_features

echo "Running utils upload_qdrant"
python -m src.utils.upload_qdrant \
  --features-json Dataset/segmented_features.json \
  --metadata-json Dataset/segmented_image_metadata.json \
  --collection myco_fungi_features_full

echo "Running cross_validation run"
python -m src.experiments.cross_validation.run --collection myco_fungi_features_full_finetuned

echo "Running cross_validation visualize"
python -m src.experiments.cross_validation.visualize

echo "Running kmeans_segmentation run"
python -m src.experiments.kmeans_segmentation.run

echo "Running retrieval run"
python -m src.experiments.retrieval.run

echo "Done"

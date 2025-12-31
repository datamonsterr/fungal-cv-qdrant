import argparse
import sys
import os
from pathlib import Path

# Add the project root directory to sys.path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    QDRANT_URL, COLLECTION_NAME, FEATURES_JSON_PATH, 
    SEGMENTED_METADATA_PATH, RESULTS_DIR, STRAIN_SPECIES_MAPPING_PATH
)
from qdrant_client import QdrantClient

def run_reformat(args):
    print("Running dataset reformatting...")
    from src.scripts.reformat_dataset import reformat_dataset
    reformat_dataset()
    print("Dataset reformatting complete.")

def run_reformat_hierarchical(args):
    print("Running hierarchical dataset reformatting...")
    from src.scripts.reformat_hierarchical_dataset import main as reformat_hierarchical
    reformat_hierarchical()
    print("Hierarchical dataset reformatting complete.")

def run_generate_mapping(args):
    print("Generating strain mapping...")
    from src.scripts.generate_strain_mapping import generate_strain_mapping
    generate_strain_mapping()
    print("Strain mapping generation complete.")

def run_extract(args):
    print("Running feature extraction...")
    from src.feature_extraction.generate_features import generate_features
    generate_features()
    
    print("Uploading features to Qdrant...")
    from src.database.upload_qdrant import upload_features_to_qdrant
    
    client = QdrantClient(url=QDRANT_URL)
    upload_features_to_qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        features_json_path=str(FEATURES_JSON_PATH),
        metadata_json_path=str(SEGMENTED_METADATA_PATH)
    )
    print("Feature extraction and upload complete.")

def run_train(args):
    print("Running model training...")
    from src.training.train_models import main as train_main
    train_main()
    print("Model training complete.")

def run_predict(args):
    print(f"Running prediction for strain: {args.strain}")
    from src.classification.prediction import predict
    from src.feature_extraction.feature_extractors import (
        ResNet50Extractor, MobileNetV2Extractor, EfficientNetV2B0Extractor,
        HOGExtractor, GaborExtractor, ColorHistogramExtractor
    )
    
    client = QdrantClient(url=QDRANT_URL)
    
    # Select extractor
    if args.extractor == "resnet50":
        extractor = ResNet50Extractor()
    elif args.extractor == "mobilenetv2":
        extractor = MobileNetV2Extractor()
    elif args.extractor == "efficientnetv2":
        extractor = EfficientNetV2B0Extractor()
    elif args.extractor == "hog":
        extractor = HOGExtractor()
    else:
        print(f"Unknown extractor: {args.extractor}")
        return

    result = predict(
        client=client,
        collection_name=COLLECTION_NAME,
        strain=args.strain,
        feature_extractor=extractor,
        k=args.k,
        without_siblings=not args.with_siblings
    )
    
    print(f"Prediction Result: {result['predicted_specy']} (Correct: {result['correct']})")
    print(f"Confidence: {result['predicted_confidence']:.4f}")

def run_evaluate(args):
    print("Running comprehensive evaluation...")
    from src.classification.evaluate_species import run_species_evaluation
    from src.feature_extraction.feature_extractors import (
        ResNet50Extractor, MobileNetV2Extractor, EfficientNetV2B0Extractor
    )
    
    client = QdrantClient(url=QDRANT_URL)
    
    # Default to ResNet50 for evaluation if not specified
    # Or loop through all? The original script might have looped.
    # Let's allow specifying one.
    
    if args.extractor == "resnet50":
        extractor = ResNet50Extractor()
    elif args.extractor == "mobilenetv2":
        extractor = MobileNetV2Extractor()
    elif args.extractor == "efficientnetv2":
        extractor = EfficientNetV2B0Extractor()
    else:
        extractor = ResNet50Extractor() # Default
        
    run_species_evaluation(
        client=client,
        collection_name=COLLECTION_NAME,
        feature_extractor=extractor,
        k=args.k,
        output_dir=str(RESULTS_DIR)
    )
    print("Evaluation complete.")

def main():
    parser = argparse.ArgumentParser(description="Fungal Species Classification Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Reformat
    parser_reformat = subparsers.add_parser("reformat", help="Reformat dataset structure")
    parser_reformat.set_defaults(func=run_reformat)

    # Reformat Hierarchical
    parser_reformat_h = subparsers.add_parser("reformat-hierarchical", help="Reformat dataset into hierarchical structure")
    parser_reformat_h.set_defaults(func=run_reformat_hierarchical)

    # Generate Mapping
    parser_mapping = subparsers.add_parser("generate-mapping", help="Generate strain to species mapping with test split")
    parser_mapping.set_defaults(func=run_generate_mapping)

    # Extract (and Upload)
    parser_extract = subparsers.add_parser("extract", help="Extract features and upload to Qdrant")
    parser_extract.set_defaults(func=run_extract)

    # Train
    parser_train = subparsers.add_parser("train", help="Train/Finetune classification models")
    parser_train.set_defaults(func=run_train)

    # Predict
    parser_predict = subparsers.add_parser("predict", help="Predict species for a strain")
    parser_predict.add_argument("--strain", type=str, required=True, help="Strain ID to predict")
    parser_predict.add_argument("--extractor", type=str, default="resnet50", help="Feature extractor to use")
    parser_predict.add_argument("--k", type=int, default=5, help="Number of neighbors")
    parser_predict.add_argument("--with-siblings", action="store_true", help="Include sibling segments in search")
    parser_predict.set_defaults(func=run_predict)

    # Evaluate
    parser_evaluate = subparsers.add_parser("evaluate", help="Run evaluation on test set")
    parser_evaluate.add_argument("--extractor", type=str, default="resnet50", help="Feature extractor to use")
    parser_evaluate.add_argument("--k", type=int, default=5, help="Number of neighbors")
    parser_evaluate.set_defaults(func=run_evaluate)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

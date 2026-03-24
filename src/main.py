import argparse
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from src.config import (
    COLLECTION_NAME,
    FEATURES_JSON_PATH,
    QDRANT_URL,
    RESULTS_DIR,
    SEGMENTED_IMAGE_DIR,
    SEGMENTED_METADATA_PATH,
)

# Add the project root directory to sys.path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def run_generate_cv_fold_mappings(args):
    print("Generating fold-specific strain mapping CSV files...")
    from src.scripts.generate_cv_fold_mappings import generate_fold_mapping_files

    generated = generate_fold_mapping_files(n_folds=args.n_folds)
    print(f"Generated {len(generated)} fold mapping files.")


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
        metadata_json_path=str(SEGMENTED_METADATA_PATH),
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
        EfficientNetB1Extractor,
        HOGExtractor,
        MobileNetV2Extractor,
        ResNet50Extractor,
    )

    client = QdrantClient(url=QDRANT_URL)

    # Select extractor
    if args.extractor == "resnet50":
        extractor = ResNet50Extractor()
    elif args.extractor == "mobilenetv2":
        extractor = MobileNetV2Extractor()
    elif args.extractor == "efficientnetb1":
        extractor = EfficientNetB1Extractor()
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
        without_siblings=not args.with_siblings,
    )

    print(
        f"Prediction Result: {result['predicted_specy']} (Correct: {result['correct']})"
    )
    print(f"Confidence: {result['predicted_confidence']:.4f}")


def run_predict_new(args):  # noqa: C901
    """Predict species for a new unseen strain from raw images."""
    import json
    from collections import Counter
    from datetime import datetime

    import cv2

    from src.classification.visualization.visualize_prediction import (
        visualize_prediction_by_environment,
    )
    from src.feature_extraction.feature_extractors import (
        ColorHistogramExtractor,
        ColorHistogramHSExtractor,
        EfficientNetB1Extractor,
        EfficientNetB1TripletExtractor,
        GaborExtractor,
        HOGExtractor,
        MobileNetV2Extractor,
        ResNet50Extractor,
    )
    from src.preprocessing.kmeans import segment_kmeans
    from src.preprocessing.preprocess import process_image

    strain_path = Path(args.path)
    if not strain_path.exists():
        print(f"Error: Path {args.path} does not exist")
        return

    # Extract strain name from path
    strain_name = strain_path.name

    # Select extractor
    if args.extractor == "resnet50":
        extractor = ResNet50Extractor()
    elif args.extractor == "mobilenetv2":
        extractor = MobileNetV2Extractor()
    elif args.extractor == "efficientnetb1":
        extractor = EfficientNetB1Extractor()
    elif args.extractor == "efficientnetb1_triplet":
        extractor = EfficientNetB1TripletExtractor()
    elif args.extractor == "hog":
        extractor = HOGExtractor()
    elif args.extractor == "gabor":
        extractor = GaborExtractor()
    elif args.extractor == "colorhistogram":
        extractor = ColorHistogramExtractor()
    elif args.extractor == "colorhistogramhs":
        extractor = ColorHistogramHSExtractor()
    else:
        print(f"Unknown extractor: {args.extractor}")
        return

    print(f"Processing new strain: {strain_name}")
    print(f"Using extractor: {extractor.name}")
    print(f"K neighbors: {args.k}")

    client = QdrantClient(url=QDRANT_URL)

    # Collect all images from environment subdirectories
    raw_results = []
    processed_count = 0

    for env_dir in sorted(strain_path.iterdir()):
        if not env_dir.is_dir():
            continue

        environment = env_dir.name
        print(f"\nProcessing environment: {environment}")

        for img_file in env_dir.glob("*.jpg"):
            print(f"  Processing: {img_file.name}")

            # Read and preprocess image
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"    Warning: Could not read {img_file}")
                continue

            # Apply preprocessing (crop Petri dish)
            processed_img = process_image(img)  # type: ignore[arg-type]

            # Segment using K-means to find bounding boxes
            bboxes = segment_kmeans(str(img_file))

            if not bboxes:
                print(f"    Warning: No segments found in {img_file.name}")
                continue

            print(f"    Found {len(bboxes)} segments")

            # Process each segment
            for seg_idx, bbox in enumerate(bboxes):
                # Crop segment
                segment = processed_img[
                    bbox["ymin"] : bbox["ymax"], bbox["xmin"] : bbox["xmax"]
                ]

                if segment.size == 0:
                    continue

                # Extract features from segment
                features = extractor.extract(segment)

                # Search in Qdrant
                neighbors = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=features.tolist(),
                    using=extractor.name.lower(),
                    limit=args.k,
                    with_payload=True,
                ).points

                # Format neighbors
                formatted_neighbors = []
                for neighbor in neighbors:
                    if neighbor.payload is not None:
                        formatted_neighbors.append(
                            {
                                "image_id": neighbor.payload.get("image_id"),
                                "score": neighbor.score,
                                "distance": 1.0 - neighbor.score,
                                "strain": neighbor.payload.get("strain"),
                                "environment": neighbor.payload.get("environment"),
                                "angle": neighbor.payload.get("angle"),
                                "specy": neighbor.payload.get("specy"),
                                "parent_id": neighbor.payload.get("parent_id"),
                                "segment_index": neighbor.payload.get("segment_index"),
                                "bbox": neighbor.payload.get("bbox"),
                            }
                        )

                raw_results.append(
                    {
                        "query_image_id": f"{img_file.stem}_seg{seg_idx}",
                        "query_environment": environment,
                        "query_file": str(img_file),
                        "segment_index": seg_idx,
                        "bbox": bbox,
                        "neighbors": formatted_neighbors,
                    }
                )

                processed_count += 1

    if not raw_results:
        print("\nError: No segments were processed successfully")
        return

    print(f"\nProcessed {processed_count} segments total")

    # Aggregate predictions across all segments
    species_scores = Counter()
    species_counts = Counter()

    for result in raw_results:
        for neighbor in result["neighbors"]:
            specy = neighbor.get("specy")
            score = neighbor.get("score", 0.0)

            if specy and specy != "unknown":
                species_scores[specy] += score
                species_counts[specy] += 1

    # Calculate aggregated scores
    total_neighbors = sum(species_counts.values())
    aggregated_results = []

    for specy, total_score in species_scores.items():
        if args.strategy == "avg":
            final_score = total_score / total_neighbors if total_neighbors > 0 else 0
        elif args.strategy == "uni":
            count = species_counts[specy]
            final_score = count / total_neighbors if total_neighbors > 0 else 0
        else:
            final_score = total_score

        aggregated_results.append({"specy": specy, "score": final_score})

    aggregated_results.sort(key=lambda x: x["score"], reverse=True)

    # Create prediction result structure
    prediction_result = {
        "strain": strain_name,
        "ground_truth": "unknown",  # We don't know the true species
        "predicted_specy": (
            aggregated_results[0]["specy"] if aggregated_results else "unknown"
        ),
        "predicted_confidence": (
            aggregated_results[0]["score"] if aggregated_results else 0.0
        ),
        "correct": None,  # Unknown since we don't have ground truth
        "aggregated_results": aggregated_results,
        "raw_results": raw_results,
        "feature_extractor": extractor.name,
        "strategy": args.strategy,
    }

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"predict_new_{strain_name}_{timestamp}"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save JSON results
    json_path = output_dir / "prediction_results.json"
    with open(json_path, "w") as f:
        json.dump(prediction_result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Prediction Results for: {strain_name}")
    print(f"{'='*60}")
    print(f"Top Predicted Species: {prediction_result['predicted_specy']}")
    print(f"Confidence: {prediction_result['predicted_confidence']:.4f}")
    print("\nTop 5 Species Rankings:")
    for i, result in enumerate(aggregated_results[:5], 1):
        print(f"  {i}. {result['specy']}: {result['score']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - JSON: {json_path.name}")

    # Generate visualization
    viz_path = output_dir / "prediction_visualization.jpg"
    try:
        visualize_prediction_by_environment(
            prediction_result=prediction_result,
            segmented_image_dir=str(SEGMENTED_IMAGE_DIR),
            output_path=str(viz_path),
            k=args.k,
        )
        print(f"  - Visualization: {viz_path.name}")
    except Exception as e:
        print(f"  Warning: Could not generate visualization: {e}")

    print(f"{'='*60}")


def run_evaluate(args):  # noqa: C901
    print("Running comprehensive evaluation...")
    from datetime import datetime

    from src.classification.evaluate_species import run_species_evaluation
    from src.feature_extraction.feature_extractors import (
        ColorHistogramExtractor,
        ColorHistogramHSExtractor,
        EfficientNetB1Extractor,
        EfficientNetB1FinetunedExtractor,
        EfficientNetB1TripletExtractor,
        GaborExtractor,
        HOGExtractor,
        MobileNetV2Extractor,
        MobileNetV2FinetunedExtractor,
        ResNet50Extractor,
        ResNet50FinetunedExtractor,
        ViT256DinoExtractor,
        ViTCellVitX20Extractor,
        ViTCellVitX40Extractor,
        ViTFinetunedExtractor,
        ViTSAMBExtractor,
        ViTSAMHExtractor,
        ViTSAMLExtractor,
    )

    client = QdrantClient(url=QDRANT_URL)

    # Use custom collection if specified, otherwise use default
    collection_name = (
        args.collection
        if hasattr(args, "collection") and args.collection
        else COLLECTION_NAME
    )

    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{timestamp}_k{args.k}"
    run_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nResults will be saved to: {run_dir}\n")

    # Map strategy names
    strategy_map = {
        "uni": "uni",
        "weighted": "avg",
        "score": "avg",
        "avg": "avg",
    }

    # Define all strategies and environments
    all_strategies = ["uni", "weighted"]
    all_environments = [
        None,
        "all",
        "E3_MEA",
        "E3_DG18",
        "E3_CREA",
        "E3_CYA",
        "E3_CYA30",
        "E3_CYAS",
        "E3_YES",
        "E4_MEA",
        "E4_DG18",
        "E4_CREA",
        "E4_CYA",
        "E4_CYA30",
        "E4_CYAS",
        "E4_YES",
    ]

    # Determine which strategies to run
    if args.strategy.lower() == "all":
        strategies_to_run = all_strategies
    else:
        strategies_to_run = [args.strategy.lower()]

    # Determine which environments to run
    if args.environment and args.environment.lower() == "all":
        environments_to_run = all_environments
    else:
        environments_to_run = [args.environment]

    # Handle 'all' extractors option
    if args.extractor == "all":
        extractors = [
            ("ResNet50", ResNet50Extractor()),
            ("MobileNetV2", MobileNetV2Extractor()),
            ("EfficientNetB1", EfficientNetB1Extractor()),
            ("HOG", HOGExtractor()),
            ("Gabor", GaborExtractor()),
            ("ColorHistogram", ColorHistogramExtractor()),
            ("ColorHistogramHS", ColorHistogramHSExtractor()),
        ]
    elif args.extractor == "all-finetuned":
        extractors = [
            ("ResNet50_finetuned", ResNet50FinetunedExtractor()),
            ("MobileNetV2_finetuned", MobileNetV2FinetunedExtractor()),
            ("EfficientNetB1_finetuned", EfficientNetB1FinetunedExtractor()),
        ]
    else:
        # Single extractor
        if args.extractor == "resnet50":
            extractor = ResNet50Extractor()
            extractor_name = "ResNet50"
        elif args.extractor == "resnet50_finetuned":
            extractor = ResNet50FinetunedExtractor()
            extractor_name = "ResNet50_finetuned"
        elif args.extractor == "mobilenetv2":
            extractor = MobileNetV2Extractor()
            extractor_name = "MobileNetV2"
        elif args.extractor == "mobilenetv2_finetuned":
            extractor = MobileNetV2FinetunedExtractor()
            extractor_name = "MobileNetV2_finetuned"
        elif args.extractor == "efficientnetb1":
            extractor = EfficientNetB1Extractor()
            extractor_name = "EfficientNetB1"
        elif args.extractor == "efficientnetb1_finetuned":
            extractor = EfficientNetB1FinetunedExtractor()
            extractor_name = "EfficientNetB1_finetuned"
        elif args.extractor == "efficientnetb1_triplet":
            extractor = EfficientNetB1TripletExtractor()
            extractor_name = "EfficientNetB1_triplet"
        elif args.extractor == "hog":
            extractor = HOGExtractor()
            extractor_name = "HOG"
        elif args.extractor == "gabor":
            extractor = GaborExtractor()
            extractor_name = "Gabor"
        elif args.extractor == "colorhistogram":
            extractor = ColorHistogramExtractor()
            extractor_name = "ColorHistogram"
        elif args.extractor == "colorhistogramhs":
            extractor = ColorHistogramHSExtractor()
            extractor_name = "ColorHistogramHS"
        elif args.extractor == "vit256_dino":
            extractor = ViT256DinoExtractor()
            extractor_name = "ViT256Dino"
        elif args.extractor == "cellvit_x20":
            extractor = ViTCellVitX20Extractor()
            extractor_name = "ViTCellVitX20"
        elif args.extractor == "cellvit_x40":
            extractor = ViTCellVitX40Extractor()
            extractor_name = "ViTCellVitX40"
        elif args.extractor == "sam_vit_b":
            extractor = ViTSAMBExtractor()
            extractor_name = "ViTSAMB"
        elif args.extractor == "sam_vit_l":
            extractor = ViTSAMLExtractor()
            extractor_name = "ViTSAML"
        elif args.extractor == "sam_vit_h":
            extractor = ViTSAMHExtractor()
            extractor_name = "ViTSAMH"
        elif args.extractor == "vit_finetuned":
            extractor = ViTFinetunedExtractor()
            extractor_name = "ViT_finetuned"
        else:
            extractor = ResNet50Extractor()
            extractor_name = "ResNet50"

        extractors = [(extractor_name, extractor)]

    # Run all combinations
    total_runs = len(extractors) * len(strategies_to_run) * len(environments_to_run)
    current_run = 0

    for extractor_name, extractor in extractors:
        for strategy_name in strategies_to_run:
            for env_value in environments_to_run:
                current_run += 1

                agg_strategy = strategy_map.get(strategy_name, "avg")
                env_label = env_value or "E1"

                print(f"\n{'='*60}")
                print(f"Run {current_run}/{total_runs}")
                print(f"Extractor: {extractor_name}")
                print(f"Strategy: {strategy_name.upper()}")
                print(f"Environment: {env_label}")
                print(f"{'='*60}\n")

                output_dir = (
                    run_dir / f"{extractor_name.lower()}_{strategy_name}_{env_label}"
                )
                output_dir.mkdir(exist_ok=True, parents=True)

                run_species_evaluation(
                    client=client,
                    collection_name=collection_name,
                    feature_extractor=extractor,
                    k=args.k,
                    environment=env_value,
                    strategy=agg_strategy,
                    output_dir=str(output_dir),
                    generate_visualizations=True,
                )

    print("\n" + "=" * 60)
    print(f"All evaluations complete! ({total_runs} total runs)")
    print(f"Results saved to: {run_dir}")
    print("=" * 60)


def run_evaluate_all(args):
    """Run evaluation for all extractors, environments, and strategies, then generate CSV summary."""
    print("Running comprehensive evaluation for all configurations...")
    import csv
    import json
    from datetime import datetime

    from src.classification.evaluate_species import run_species_evaluation
    from src.feature_extraction.feature_extractors import (
        ColorHistogramExtractor,
        ColorHistogramHSExtractor,
        EfficientNetB1Extractor,
        EfficientNetB1FinetunedExtractor,
        EfficientNetB1TripletExtractor,
        GaborExtractor,
        HOGExtractor,
        MobileNetV2Extractor,
        MobileNetV2FinetunedExtractor,
        ResNet50Extractor,
        ResNet50FinetunedExtractor,
        ViT256DinoExtractor,
        ViTCellVitX20Extractor,
        ViTCellVitX40Extractor,
        ViTFinetunedExtractor,
        ViTSAMBExtractor,
        ViTSAMHExtractor,
        ViTSAMLExtractor,
    )

    client = QdrantClient(url=QDRANT_URL)

    # Use custom collection if specified, otherwise use default
    collection_name = (
        args.collection
        if hasattr(args, "collection") and args.collection
        else COLLECTION_NAME
    )

    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{timestamp}_k{args.k}"
    run_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nResults will be saved to: {run_dir}\n")

    # Define all extractors
    all_extractors = [
        ("ResNet50", ResNet50Extractor()),
        ("MobileNetV2", MobileNetV2Extractor()),
        ("EfficientNetB1", EfficientNetB1Extractor()),
        ("HOG", HOGExtractor()),
        ("Gabor", GaborExtractor()),
        ("ColorHistogram", ColorHistogramExtractor()),
        ("ColorHistogramHS", ColorHistogramHSExtractor()),
    ]

    # If using fine-tuned collection, use fine-tuned extractors instead of DL models
    if collection_name and "finetuned" in collection_name.lower():
        all_extractors = [
            ("ResNet50_finetuned", ResNet50FinetunedExtractor()),
            ("MobileNetV2_finetuned", MobileNetV2FinetunedExtractor()),
            ("EfficientNetB1_finetuned", EfficientNetB1FinetunedExtractor()),
            ("EfficientNetB1_triplet", EfficientNetB1TripletExtractor()),
            ("HOG", HOGExtractor()),
            ("Gabor", GaborExtractor()),
            ("ColorHistogram", ColorHistogramExtractor()),
            ("ColorHistogramHS", ColorHistogramHSExtractor()),
        ]

    # Define all strategies and environments
    all_strategies = [("uni", "uni"), ("weighted", "avg")]
    all_environments = [
        (None, "E1"),
        ("all", "all"),
        ("E3_MEA", "E3_MEA"),
        ("E3_DG18", "E3_DG18"),
        ("E3_CREA", "E3_CREA"),
        ("E3_CYA", "E3_CYA"),
        ("E3_CYA30", "E3_CYA30"),
        ("E3_CYAS", "E3_CYAS"),
        ("E3_YES", "E3_YES"),
        ("E4_MEA", "E4_MEA"),
        ("E4_DG18", "E4_DG18"),
        ("E4_CREA", "E4_CREA"),
        ("E4_CYA", "E4_CYA"),
        ("E4_CYA30", "E4_CYA30"),
        ("E4_CYAS", "E4_CYAS"),
        ("E4_YES", "E4_YES"),
    ]

    # Store results for CSV
    csv_results = []

    # Run all combinations
    total_runs = len(all_extractors) * len(all_strategies) * len(all_environments)
    current_run = 0

    for extractor_name, extractor in all_extractors:
        for strategy_name, agg_strategy in all_strategies:
            for env_value, env_label in all_environments:
                current_run += 1

                print(f"\n{'='*60}")
                print(f"Run {current_run}/{total_runs}")
                print(f"Extractor: {extractor_name}")
                print(f"Strategy: {strategy_name.upper()}")
                print(f"Environment: {env_label}")
                print(f"{'='*60}\n")

                output_dir = (
                    run_dir / f"{extractor_name.lower()}_{strategy_name}_{env_label}"
                )
                output_dir.mkdir(exist_ok=True, parents=True)

                run_species_evaluation(
                    client=client,
                    collection_name=collection_name,
                    feature_extractor=extractor,
                    k=args.k,
                    environment=env_value,
                    strategy=agg_strategy,
                    output_dir=str(output_dir),
                    generate_visualizations=True,  # Skip visualizations for speed
                )

                # Read the results JSON to extract accuracy
                results_json_path = output_dir / "evaluation_results.json"
                try:
                    if results_json_path.exists():
                        with open(results_json_path, "r") as f:
                            results = json.load(f)
                            accuracy = results.get("overall_accuracy", 0.0)
                            total_strains = results.get("total_strains", 0)
                            correct_predictions = results.get("correct_predictions", 0)

                            csv_results.append(
                                {
                                    "extractor": extractor_name,
                                    "strategy": strategy_name.upper(),
                                    "environment": env_label,
                                    "accuracy": accuracy,
                                    "correct": correct_predictions,
                                    "total": total_strains,
                                    "k": args.k,
                                }
                            )
                    else:
                        print(f"Warning: {results_json_path} not found")
                except Exception as e:
                    print(
                        f"Error reading results for {extractor_name}/{strategy_name}/{env_label}: {e}"
                    )

    # Generate CSV summary
    csv_path = run_dir / "accuracy_summary.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "extractor",
            "strategy",
            "environment",
            "accuracy",
            "correct",
            "total",
            "k",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in csv_results:
            writer.writerow(row)

    print("\n" + "=" * 60)
    print(f"All evaluations complete! ({total_runs} total runs)")
    print(f"Results saved to: {run_dir}")
    print(f"CSV summary: {csv_path}")
    print("=" * 60)

    # Print top 10 results
    print("\nTop 10 Configurations by Accuracy:")
    print("-" * 80)
    sorted_results = sorted(csv_results, key=lambda x: x["accuracy"], reverse=True)
    print(
        f"{'Rank':<6} {'Extractor':<18} {'Strategy':<12} {'Environment':<12} {'Accuracy':<10}"
    )
    print("-" * 80)
    for i, result in enumerate(sorted_results[:10], 1):
        print(
            f"{i:<6} {result['extractor']:<18} {result['strategy']:<12} {result['environment']:<12} {result['accuracy']:.4f}"
        )
    print("-" * 80)


def run_extract_finetuned(args):
    """Extract features using fine-tuned deep learning models."""
    print("Extracting features with fine-tuned models...")
    from src.scripts.extract_finetuned_features import main as extract_finetuned_main

    extract_finetuned_main(fold_index=args.fold)
    print("Fine-tuned feature extraction complete.")


def run_upload_finetuned(args):
    """Upload fine-tuned features to existing Qdrant collection."""
    print("Uploading fine-tuned features to Qdrant...")
    from src.scripts.upload_finetuned_features import main as upload_finetuned_main

    upload_finetuned_main(fold_index=args.fold)
    print("Fine-tuned feature upload complete.")


def run_extract_vit(args):
    """Extract features using Vision Transformer models."""
    print(f"Extracting ViT features with {args.weights_type} weights...")
    from src.scripts.extract_vit_features import extract_vit_features

    extract_vit_features(
        output_json_path=args.output,
        weights_type=args.weights_type,
    )
    print("ViT feature extraction complete.")


def run_upload_vit(args):
    """Upload ViT features to Qdrant collection."""
    print(f"Uploading ViT features to collection '{args.collection}'...")
    from src.scripts.upload_vit_features import upload_vit_features

    upload_vit_features(
        features_json_path=args.features,
        collection_name=args.collection,
    )
    print("ViT feature upload complete.")


def run_cross_validate(args):
    """Run 5-fold strain-level cross-validation."""
    print("Running 5-fold cross-validation...")
    from src.scripts.cross_validation import run_cross_validation

    collection = getattr(args, "collection", None)
    extractor = getattr(args, "extractor", "efficientnetb1_finetuned")
    run_cross_validation(
        collection_name=collection,
        extractor_key=extractor,
        use_fold_specific_assets=args.use_fold_specific_assets,
        collection_template=args.collection_template,
        weights_dir=args.weights_dir,
    )
    print("Cross-validation complete.")


def run_cross_validate_visualize(args):
    """Generate visualizations from cross-validation results."""
    print("Generating cross-validation visualizations...")
    from src.scripts.cv_visualize import run_visualizations

    run_visualizations()
    print("Visualization generation complete.")


def run_report(args):
    print("Running comprehensive report...")
    from src.experiments.comprehensive_report import run_comprehensive_report

    # Parse lists
    extractors = (
        args.extractors.split(",")
        if args.extractors
        else [
            "resnet50",
            "mobilenetv2",
            "efficientnetv2",
            "hog",
            "gabor",
            "colorhistogram",
        ]
    )
    env_strategies = (
        args.strategies.split(",") if args.strategies else ["all", "ob", "rev"]
    )  # Default assumptions
    agg_strategies = (
        args.agg_strategies.split(",") if args.agg_strategies else ["avg", "uni"]
    )

    # Clean up whitespace
    extractors = [e.strip().lower() for e in extractors]
    env_strategies = [e.strip().lower() for e in env_strategies]
    agg_strategies = [e.strip().lower() for e in agg_strategies]

    run_comprehensive_report(
        identifier=args.identifier,
        extractors=extractors,
        env_strategies=env_strategies,
        agg_strategies=agg_strategies,
        k=args.k,
    )
    print("Comprehensive report complete.")


def run_download_dataset(args: argparse.Namespace) -> None:
    """Handler for the download-dataset subcommand."""
    from src.scripts.gdrive_download import run_download

    exit_code = run_download(
        url=args.url,
        target=args.target,
        force=args.force,
        quiet=args.quiet,
        fuzzy=False,
    )
    if exit_code != 0:
        sys.exit(exit_code)


def main():
    parser = argparse.ArgumentParser(
        description="Fungal Species Classification Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Reformat
    parser_reformat = subparsers.add_parser(
        "reformat", help="Reformat dataset structure"
    )
    parser_reformat.set_defaults(func=run_reformat)

    # Reformat Hierarchical
    parser_reformat_h = subparsers.add_parser(
        "reformat-hierarchical",
        help="Reformat dataset into hierarchical structure",
    )
    parser_reformat_h.set_defaults(func=run_reformat_hierarchical)

    # Generate Mapping
    parser_mapping = subparsers.add_parser(
        "generate-mapping",
        help="Generate strain to species mapping with test split",
    )
    parser_mapping.set_defaults(func=run_generate_mapping)

    parser_cv_mapping = subparsers.add_parser(
        "generate-cv-fold-mappings",
        help="Generate fold-specific mapping CSV files with Test split per fold",
    )
    parser_cv_mapping.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds to generate (default: 5)",
    )
    parser_cv_mapping.set_defaults(func=run_generate_cv_fold_mappings)

    # Extract (and Upload)
    parser_extract = subparsers.add_parser(
        "extract", help="Extract features and upload to Qdrant"
    )
    parser_extract.set_defaults(func=run_extract)

    # Train
    parser_train = subparsers.add_parser(
        "train", help="Train/Finetune classification models"
    )
    parser_train.set_defaults(func=run_train)

    # Predict
    parser_predict = subparsers.add_parser(
        "predict", help="Predict species for a strain"
    )
    parser_predict.add_argument(
        "--strain", type=str, required=True, help="Strain ID to predict"
    )
    parser_predict.add_argument(
        "--extractor",
        type=str,
        default="resnet50",
        help="Feature extractor to use",
    )
    parser_predict.add_argument("--k", type=int, default=5, help="Number of neighbors")
    parser_predict.add_argument(
        "--with-siblings",
        action="store_true",
        help="Include sibling segments in search",
    )
    parser_predict.set_defaults(func=run_predict)

    # Predict New (for unseen strains)
    parser_predict_new = subparsers.add_parser(
        "predict-new",
        help="Predict species for a new unseen strain from raw images",
    )
    parser_predict_new.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to strain folder (strain_name/environment_name/*.jpg)",
    )
    parser_predict_new.add_argument(
        "--extractor",
        type=str,
        default="resnet50",
        help="Feature extractor to use (resnet50, mobilenetv2, efficientnetb1, hog, gabor, colorhistogram, colorhistogramhs)",
    )
    parser_predict_new.add_argument(
        "--k", type=int, default=7, help="Number of neighbors"
    )
    parser_predict_new.add_argument(
        "--strategy",
        type=str,
        default="avg",
        choices=["avg", "uni"],
        help="Aggregation strategy",
    )
    parser_predict_new.set_defaults(func=run_predict_new)

    # Evaluate
    parser_evaluate = subparsers.add_parser(
        "evaluate", help="Run evaluation on test set"
    )
    parser_evaluate.add_argument(
        "--extractor",
        type=str,
        default="resnet50",
        help="Feature extractor to use (use 'all' for all extractors, 'all-finetuned' for fine-tuned models only, or specific: resnet50, resnet50_finetuned, mobilenetv2, mobilenetv2_finetuned, etc.)",
    )
    parser_evaluate.add_argument("--k", type=int, default=5, help="Number of neighbors")
    parser_evaluate.add_argument(
        "--strategy",
        type=str,
        default="weighted",
        help="Aggregation strategy: all, uni (count-based), weighted/score/avg (similarity-weighted)",
    )
    parser_evaluate.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Environment strategy: all, E1 (same env), E2 (all env), E3_<env> (specific env), E4_<env> (exclude env). Default: E1",
    )
    parser_evaluate.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name to use (default: uses COLLECTION_NAME from config)",
    )
    parser_evaluate.set_defaults(func=run_evaluate)

    # Evaluate All
    parser_evaluate_all = subparsers.add_parser(
        "evaluate-all",
        help="Run evaluation for all extractors, environments, and strategies with CSV summary",
    )
    parser_evaluate_all.add_argument(
        "--k", type=int, default=7, help="Number of neighbors"
    )
    parser_evaluate_all.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name to use (default: uses COLLECTION_NAME from config)",
    )
    parser_evaluate_all.set_defaults(func=run_evaluate_all)

    # Report
    parser_report = subparsers.add_parser(
        "report", help="Generate comprehensive report"
    )
    parser_report.add_argument(
        "--identifier",
        type=str,
        required=True,
        help="Report identifier (folder name)",
    )
    parser_report.add_argument(
        "--extractors", type=str, help="Comma-separated list of extractors"
    )
    parser_report.add_argument(
        "--strategies",
        type=str,
        help="Comma-separated list of environment strategies",
    )
    parser_report.add_argument(
        "--agg-strategies",
        type=str,
        help="Comma-separated list of aggregation strategies",
    )
    parser_report.add_argument("--k", type=int, default=5, help="Number of neighbors")
    parser_report.set_defaults(func=run_report)

    # Extract Fine-tuned
    parser_extract_ft = subparsers.add_parser(
        "extract-finetuned",
        help="Extract features using fine-tuned deep learning models",
    )
    parser_extract_ft.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold index for fold-specific weight and output naming",
    )
    parser_extract_ft.set_defaults(func=run_extract_finetuned)

    # Upload Fine-tuned
    parser_upload_ft = subparsers.add_parser(
        "upload-finetuned",
        help="Upload fine-tuned features to existing Qdrant collection",
    )
    parser_upload_ft.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold index for fold-specific feature file and collection naming",
    )
    parser_upload_ft.set_defaults(func=run_upload_finetuned)

    # Extract ViT
    parser_extract_vit = subparsers.add_parser(
        "extract-vit",
        help="Extract features using Vision Transformer models",
    )
    parser_extract_vit.add_argument(
        "--weights-type",
        type=str,
        default="vit256_dino",
        choices=[
            "cellvit_x20",
            "cellvit_x40",
            "sam_vit_b",
            "sam_vit_l",
            "sam_vit_h",
            "vit256_dino",
        ],
        help="Type of ViT pretrained weights to use",
    )
    parser_extract_vit.add_argument(
        "--output",
        type=str,
        default="Dataset/vit_features.json",
        help="Output JSON file path",
    )
    parser_extract_vit.set_defaults(func=run_extract_vit)

    # Upload ViT
    parser_upload_vit = subparsers.add_parser(
        "upload-vit",
        help="Upload ViT features to Qdrant collection",
    )
    parser_upload_vit.add_argument(
        "--features",
        type=str,
        default="Dataset/vit_features.json",
        help="Path to ViT features JSON file",
    )
    parser_upload_vit.add_argument(
        "--collection",
        type=str,
        default="myco_fungi_features_vit",
        help="Name of the Qdrant collection",
    )
    parser_upload_vit.set_defaults(func=run_upload_vit)

    # Cross-Validate
    parser_cv = subparsers.add_parser(
        "cross-validate",
        help="Run 5-fold strain-level cross-validation (E1/E2 × uni/avg × K∈{3,5,7,9,11})",
    )
    parser_cv.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default: from config; use 'myco_fungi_features_full_finetuned' for finetuned)",
    )
    parser_cv.add_argument(
        "--extractor",
        type=str,
        default="efficientnetb1_finetuned",
        help="Extractor key to use (default: efficientnetb1_finetuned)",
    )
    parser_cv.add_argument(
        "--use-fold-specific-assets",
        action="store_true",
        help="Use fold-specific EfficientNetB1 weights and fold-specific collections",
    )
    parser_cv.add_argument(
        "--collection-template",
        type=str,
        default=None,
        help="Optional collection template, e.g. 'my_collection_fold{fold}'",
    )
    parser_cv.add_argument(
        "--weights-dir",
        type=str,
        default="weights",
        help="Directory containing fold-specific weights",
    )
    parser_cv.set_defaults(func=run_cross_validate)

    # Cross-Validate Visualize
    parser_cv_viz = subparsers.add_parser(
        "cross-validate-visualize",
        help="Generate visualizations from cross-validation results (reads report/week_1_2/cv_results.csv)",
    )
    parser_cv_viz.set_defaults(func=run_cross_validate_visualize)

    # Download Dataset
    parser_download = subparsers.add_parser(
        "download-dataset",
        help="Download a dataset from a shared Google Drive URL",
    )
    parser_download.add_argument(
        "--url",
        required=True,
        help="Shared Google Drive URL (file or folder)",
    )
    parser_download.add_argument(
        "--target",
        required=True,
        help="Local target directory to download into",
    )
    parser_download.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that already exist",
    )
    parser_download.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser_download.set_defaults(func=run_download_dataset)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

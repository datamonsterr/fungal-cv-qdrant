"""
Debug script to verify fine-tuned model feature extraction.
This helps diagnose why training accuracy is high but evaluation accuracy is low.
"""

import os

import numpy as np
from tensorflow.keras.models import Model, load_model

# Paths to trained models
RESNET50_WEIGHTS = "./weights/resnet50/20251229_214513/best_model.h5"
MOBILENETV2_WEIGHTS = "./weights/mobilenetv2/20251229_223112/best_model.h5"
EFFICIENTNETV2B0_WEIGHTS = "./weights/efficientnetv2b0/20251229_225347/best_model.h5"


def inspect_model_architecture(weights_path: str, model_name: str):
    """
    Inspect the architecture of a trained model.

    Args:
        weights_path: Path to model weights
        model_name: Name of the model for display
    """
    print("=" * 80)
    print(f"INSPECTING {model_name} ARCHITECTURE")
    print("=" * 80)

    if not os.path.exists(weights_path):
        print(f"❌ Model not found at: {weights_path}\n")
        return

    try:
        # Load the full model
        model = load_model(weights_path)

        print(f"\n✓ Model loaded from: {weights_path}")
        print(f"\nFull Model Architecture:")
        print("-" * 80)

        for i, layer in enumerate(model.layers):
            output_shape = (
                layer.output_shape if hasattr(layer, "output_shape") else "N/A"
            )
            print(
                f"Layer {i}: {layer.name:<40} {layer.__class__.__name__:<30} Output: {output_shape}"
            )

        print("\n" + "-" * 80)
        print(f"Total layers: {len(model.layers)}")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")

        # Find GlobalAveragePooling layer
        print("\n" + "-" * 80)
        print("Searching for GlobalAveragePooling2D layer...")
        gap_layer = None
        gap_index = -1
        for i, layer in enumerate(model.layers):
            if "global_average_pooling" in layer.name.lower():
                gap_layer = layer
                gap_index = i
                print(
                    f"✓ Found at Layer {i}: {layer.name} (Output shape: {layer.output_shape})"
                )
                break

        if gap_layer is None:
            print(
                "⚠️  GlobalAveragePooling2D layer not found by name, trying index 2..."
            )
            if len(model.layers) > 2:
                gap_layer = model.layers[2]
                gap_index = 2
                print(
                    f"Using Layer 2: {gap_layer.name} (Output shape: {gap_layer.output_shape})"
                )

        # Create feature extractor
        if gap_layer:
            feature_extractor = Model(inputs=model.input, outputs=gap_layer.output)
            print(f"\n✓ Feature extractor created!")
            print(f"  Input shape: {feature_extractor.input_shape}")
            print(f"  Output shape: {feature_extractor.output_shape}")
            print(f"  Feature dimension: {feature_extractor.output_shape[1]}")

            # Test with dummy input
            print("\n" + "-" * 80)
            print("Testing feature extraction with dummy input...")
            dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            features = feature_extractor.predict(dummy_input, verbose=0)
            print(f"✓ Feature extraction successful!")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {features.shape}")
            print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")
            print(f"  Feature mean: {features.mean():.4f}")
            print(f"  Feature std: {features.std():.4f}")

        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback

        traceback.print_exc()
        print()


def compare_feature_extraction():
    """Compare ImageNet vs fine-tuned feature extraction."""
    print("=" * 80)
    print("COMPARING IMAGENET VS FINE-TUNED FEATURES")
    print("=" * 80)
    print()

    import cv2
    from feature_extractors import ResNet50Extractor

    # Find a test image
    test_image_path = "../Dataset/segmented_image"
    test_images = []
    if os.path.exists(test_image_path):
        import glob

        test_images = glob.glob(os.path.join(test_image_path, "*.jpg"))[:1]

    if not test_images:
        print("⚠️  No test images found in ../Dataset/segmented_image/")
        print("Skipping feature comparison.\n")
        return

    test_image_file = test_images[0]
    print(f"Using test image: {test_image_file}\n")

    # Load image
    image = cv2.imread(test_image_file)
    if image is None:
        print(f"❌ Failed to load image: {test_image_file}\n")
        return

    print(f"Image shape: {image.shape}")
    print()

    # Extract features with ImageNet weights
    print("-" * 80)
    print("1. Extracting features with ImageNet weights...")
    print("-" * 80)
    extractor_imagenet = ResNet50Extractor()
    features_imagenet = extractor_imagenet.extract(image)
    print(f"✓ Feature shape: {features_imagenet.shape}")
    print(
        f"  Feature range: [{features_imagenet.min():.4f}, {features_imagenet.max():.4f}]"
    )
    print(f"  Feature mean: {features_imagenet.mean():.4f}")
    print(f"  Feature std: {features_imagenet.std():.4f}")
    print()

    # Extract features with fine-tuned weights
    print("-" * 80)
    print("2. Extracting features with fine-tuned weights...")
    print("-" * 80)
    if os.path.exists(RESNET50_WEIGHTS):
        extractor_finetuned = ResNet50Extractor(weights_path=RESNET50_WEIGHTS)
        features_finetuned = extractor_finetuned.extract(image)
        print(f"✓ Feature shape: {features_finetuned.shape}")
        print(
            f"  Feature range: [{features_finetuned.min():.4f}, {features_finetuned.max():.4f}]"
        )
        print(f"  Feature mean: {features_finetuned.mean():.4f}")
        print(f"  Feature std: {features_finetuned.std():.4f}")
        print()

        # Compare
        print("-" * 80)
        print("3. Comparison")
        print("-" * 80)
        cosine_sim = np.dot(features_imagenet, features_finetuned) / (
            np.linalg.norm(features_imagenet) * np.linalg.norm(features_finetuned)
        )
        l2_distance = np.linalg.norm(features_imagenet - features_finetuned)

        print(f"Cosine similarity: {cosine_sim:.4f}")
        print(f"L2 distance: {l2_distance:.4f}")
        print()

        if cosine_sim > 0.95:
            print("⚠️  WARNING: Features are very similar (cosine sim > 0.95)")
            print("   This suggests fine-tuned weights may not be loading correctly!")
        else:
            print(
                "✓ Features are different, fine-tuned weights are being used correctly"
            )
        print()
    else:
        print(f"⚠️  Fine-tuned weights not found at: {RESNET50_WEIGHTS}")
        print()

    print("=" * 80 + "\n")


def main():
    """Main function."""
    print("\n")
    print("=" * 80)
    print("FINE-TUNED MODEL DIAGNOSTIC TOOL")
    print("=" * 80)
    print()
    print("This script helps diagnose issues with fine-tuned model feature extraction.")
    print()

    # Inspect each model
    models = [
        (RESNET50_WEIGHTS, "ResNet50"),
        (MOBILENETV2_WEIGHTS, "MobileNetV2"),
        (EFFICIENTNETV2B0_WEIGHTS, "EfficientNetV2B0"),
    ]

    for weights_path, model_name in models:
        inspect_model_architecture(weights_path, model_name)

    # Compare feature extraction
    compare_feature_extraction()

    # Summary
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print()
    print("What to do next:")
    print()
    print(
        "1. ✅ FIXED: Feature extractors now correctly extract from GlobalAveragePooling layer"
    )
    print()
    print(
        "2. 🔄 RE-EXTRACT FEATURES: You MUST re-extract features with fixed extractors:"
    )
    print("   $ uv run python feature_extractors.py")
    print()
    print("3. 🔄 RE-UPLOAD TO QDRANT: Upload the new features:")
    print("   $ uv run python upload_qdrant.py")
    print()
    print("4. ✅ RUN EVALUATION: Now evaluate with matching features:")
    print("   $ uv run python run_comprehensive_eval.py")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

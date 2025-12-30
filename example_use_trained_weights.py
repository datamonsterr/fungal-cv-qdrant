"""
Example script showing how to use fine-tuned feature extractors.
"""
import os
from feature_extractors import (
    ResNet50Extractor,
    MobileNetV2Extractor,
    EfficientNetV2B0Extractor
)

# Example 1: Using default ImageNet weights
print("Example 1: Using ImageNet weights (default)")
print("=" * 80)
extractor_default = ResNet50Extractor()
print(f"Extractor: {extractor_default.name}")
print(f"Feature dimension: {extractor_default._feature_dim}")
print()

# Example 2: Using fine-tuned weights
print("Example 2: Using fine-tuned weights")
print("=" * 80)

# Path to your trained model weights
# Format: weights/{model_name}/{timestamp}/best_model.h5
# Example: weights/resnet50/20241229_143022/best_model.h5

# Replace with your actual trained weights path
weights_path = "./weights/resnet50/20241229_143022/best_model.h5"

if os.path.exists(weights_path):
    extractor_finetuned = ResNet50Extractor(weights_path=weights_path)
    print(f"Extractor: {extractor_finetuned.name}")
    print(f"Feature dimension: {extractor_finetuned._feature_dim}")
    print(f"Using fine-tuned weights from: {weights_path}")
else:
    print(f"Weights not found at: {weights_path}")
    print("Train a model first using train_models.py")
    print("Then update the weights_path variable with the actual path")

print()

# Example 3: Using different models with fine-tuned weights
print("Example 3: All models with optional fine-tuned weights")
print("=" * 80)

model_configs = [
    {
        'name': 'ResNet50',
        'class': ResNet50Extractor,
        'weights': './weights/resnet50/20241229_143022/best_model.h5'
    },
    {
        'name': 'MobileNetV2',
        'class': MobileNetV2Extractor,
        'weights': './weights/mobilenetv2/20241229_143022/best_model.h5'
    },
    {
        'name': 'EfficientNetV2B0',
        'class': EfficientNetV2B0Extractor,
        'weights': './weights/efficientnetv2b0/20241229_143022/best_model.h5'
    }
]

for config in model_configs:
    print(f"\n{config['name']}:")
    print("-" * 40)
    
    # Try with fine-tuned weights
    if os.path.exists(config['weights']):
        extractor = config['class'](weights_path=config['weights'])
        print(f"✓ Loaded fine-tuned weights")
    else:
        # Fall back to ImageNet weights
        extractor = config['class']()
        print(f"⚠ Using ImageNet weights (fine-tuned weights not found)")
    
    print(f"  Feature dimension: {extractor._feature_dim}")

print("\n" + "=" * 80)
print("Usage in your code:")
print("=" * 80)
print("""
# Import the extractor
from feature_extractors import ResNet50Extractor

# Option 1: Use ImageNet weights (default)
extractor = ResNet50Extractor()

# Option 2: Use fine-tuned weights
extractor = ResNet50Extractor(weights_path='./weights/resnet50/20241229_143022/best_model.h5')

# Extract features from an image
import cv2
image = cv2.imread('path/to/image.jpg')
features = extractor.extract(image)
print(f"Feature vector shape: {features.shape}")
""")

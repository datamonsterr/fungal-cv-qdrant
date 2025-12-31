import os
import json
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional
from skimage.feature import hog
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights
)
from PIL import Image

def l2_normalize(features: np.ndarray) -> np.ndarray:
    """
    Apply L2 normalization to feature vector.
    
    Args:
        features: Feature vector as numpy array
        
    Returns:
        L2 normalized feature vector
    """
    norm = np.linalg.norm(features, ord=2)
    if norm == 0:
        return features
    return features / norm

class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from an image.
        
        Args:
            image: Input image as numpy array (BGR format from cv2)
            
        Returns:
            Feature vector as 1D numpy array
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        pass

class HOGExtractor(FeatureExtractor):
    """Histogram of Oriented Gradients (HOG) feature extractor."""
    
    def __init__(self, orientations: int = 9, pixels_per_cell: Tuple[int, int] = (8, 8),
                 cells_per_block: Tuple[int, int] = (2, 2), target_size: Tuple[int, int] = (128, 128)):
        super().__init__("HOG")
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.target_size = target_size
        self._feature_dim: Optional[int] = None
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features from image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        gray = cv2.resize(gray, self.target_size)
        
        # Extract HOG features
        features = hog(gray, orientations=self.orientations,
                      pixels_per_cell=self.pixels_per_cell,
                      cells_per_block=self.cells_per_block,
                      visualize=False, feature_vector=True)
        
        if self._feature_dim is None:
            self._feature_dim = len(features)
        
        # Apply L2 normalization
        features = l2_normalize(features)
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for HOG."""
        if self._feature_dim is None:
            # Calculate expected dimension
            dummy = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            self.extract(dummy)
        
        feature_dim = self._feature_dim if self._feature_dim is not None else 0
        return [f"hog_{i}" for i in range(feature_dim)]

class GaborExtractor(FeatureExtractor):
    """Gabor filter feature extractor."""
    
    def __init__(self, frequencies: Optional[List[float]] = None, thetas: Optional[List[float]] = None,
                 target_size: Tuple[int, int] = (128, 128)):
        super().__init__("Gabor")
        self.frequencies = frequencies or [0.1, 0.2, 0.3, 0.4]
        self.thetas = thetas or [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.target_size = target_size
        self._kernels: List[Any] = self._prepare_kernels()
    
    def _prepare_kernels(self) -> List[Any]:
        """Prepare Gabor kernels with different frequencies and orientations."""
        kernels: List[Any] = []
        for freq in self.frequencies:
            for theta in self.thetas:
                kernel = np.real(gabor_kernel(freq, theta=theta))
                kernels.append(kernel)
        return kernels
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract Gabor filter features from image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        gray = cv2.resize(gray, self.target_size)
        
        # Apply Gabor filters and compute statistics
        features: List[float] = []
        for kernel in self._kernels:
            filtered = ndi.convolve(gray, kernel, mode='wrap')
            # Extract mean and std as features
            features.extend([filtered.mean(), filtered.std()])
        
        # Apply L2 normalization
        return l2_normalize(np.array(features))
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for Gabor filters."""
        names: List[str] = []
        for i, _ in enumerate(self.frequencies):
            for j, _ in enumerate(self.thetas):
                names.append(f"gabor_f{i}_t{j}_mean")
                names.append(f"gabor_f{i}_t{j}_std")
        return names

class ColorHistogramExtractor(FeatureExtractor):
    """Color histogram feature extractor."""
    
    def __init__(self, bins: int = 32, target_size: Tuple[int, int] = (128, 128)):
        super().__init__("ColorHistogram")
        self.bins = bins
        self.target_size = target_size
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features from image."""
        # Resize image
        resized = cv2.resize(image, self.target_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Compute histogram for each channel
        features: List[float] = []
        for i in range(3):  # R, G, B channels
            hist = cv2.calcHist([rgb], [i], None, [self.bins], [0, 256])
            hist = hist.flatten()
            features.extend(hist)
        
        # Apply L2 normalization
        return l2_normalize(np.array(features))
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for color histogram."""
        names: List[str] = []
        for channel in ['r', 'g', 'b']:
            for i in range(self.bins):
                names.append(f"hist_{channel}_{i}")
        return names

class ColorHistogramHSExtractor(FeatureExtractor):
    """Color histogram feature extractor using HSV color space (H and S channels only)."""
    
    def __init__(self, bins: int = 32, target_size: Tuple[int, int] = (128, 128)):
        super().__init__("ColorHistogramHS")
        self.bins = bins
        self.target_size = target_size
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features from H and S channels of HSV image."""
        # Resize image
        resized = cv2.resize(image, self.target_size)
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        # Compute histogram for H and S channels only (skip V channel)
        features: List[float] = []
        for i in range(2):  # H and S channels (indices 0 and 1)
            if i == 0:  # Hue channel: range [0, 180] in OpenCV
                hist = cv2.calcHist([hsv], [i], None, [self.bins], [0, 180])
            else:  # Saturation channel: range [0, 256]
                hist = cv2.calcHist([hsv], [i], None, [self.bins], [0, 256])
            hist = hist.flatten()
            features.extend(hist)
        
        # Apply L2 normalization
        return l2_normalize(np.array(features))
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for HSV histogram (H and S channels)."""
        names: List[str] = []
        for channel in ['h', 's']:
            for i in range(self.bins):
                names.append(f"hist_{channel}_{i}")
        return names

class BaseDeepLearningExtractor(FeatureExtractor):
    """Base class for Deep Learning feature extractors using PyTorch."""
    
    def __init__(self, name: str, target_size: Tuple[int, int], weights_path: Optional[str] = None):
        super().__init__(name)
        self.target_size = target_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(weights_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self._feature_dim = self._get_feature_dim()

    @abstractmethod
    def _build_model(self, weights_path: Optional[str]) -> nn.Module:
        pass

    def _get_feature_dim(self) -> int:
        # Run a dummy input to get output dimension
        dummy_input = torch.zeros(1, 3, *self.target_size).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
        return output.shape[1]

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract features using the deep learning model."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_tensor = self.preprocess(rgb).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_tensor)
        
        # Move to CPU and convert to numpy
        features_np = features.cpu().numpy().flatten()
        
        # Apply L2 normalization
        return l2_normalize(features_np)

    def get_feature_names(self) -> List[str]:
        return [f"{self.name.lower()}_{i}" for i in range(self._feature_dim)]

class ResNet50Extractor(BaseDeepLearningExtractor):
    """ResNet50 feature extractor."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), weights_path: Optional[str] = None):
        super().__init__("ResNet50", target_size, weights_path)
        
    def _build_model(self, weights_path: Optional[str]) -> nn.Module:
        if weights_path and os.path.exists(weights_path):
            print(f"Loading fine-tuned ResNet50 weights from: {weights_path}")
            # Load full model and remove head
            # Assuming weights_path points to a state_dict of a model with a classifier
            # We need to reconstruct the architecture used during training
            model = resnet50(weights=None)
            # Replace fc layer to match training (if needed) or just load state_dict
            # For feature extraction, we want the backbone.
            # If loading a full model checkpoint:
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # If the checkpoint has a different fc layer, we might need to adjust
                # But here we want features *before* the fc layer.
                # ResNet50: (avgpool): AdaptiveAvgPool2d(output_size=(1, 1)), (fc): Linear(...)
                
                # Load weights (ignoring fc if mismatch, or loading all if matching)
                # A robust way is to load weights into the full model, then strip fc.
                # However, if the trained model had a different number of classes, loading strict=True will fail.
                
                # Let's assume we load what we can.
                model.load_state_dict(state_dict, strict=False)
                print("✓ Fine-tuned weights loaded (partial/full)")
            except Exception as e:
                print(f"Warning: Failed to load fine-tuned weights: {e}")
                print("Using ImageNet weights instead")
                model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            
        # Remove the classification head (fc layer)
        # ResNet50 structure ends with avgpool -> flatten -> fc
        # We want the output of avgpool, flattened.
        # We can replace fc with Identity
        model.fc = nn.Identity()
        return model

class MobileNetV2Extractor(BaseDeepLearningExtractor):
    """MobileNetV2 feature extractor."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), weights_path: Optional[str] = None):
        super().__init__("MobileNetV2", target_size, weights_path)
        
    def _build_model(self, weights_path: Optional[str]) -> nn.Module:
        if weights_path and os.path.exists(weights_path):
            print(f"Loading fine-tuned MobileNetV2 weights from: {weights_path}")
            model = mobilenet_v2(weights=None)
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                model.load_state_dict(state_dict, strict=False)
                print("✓ Fine-tuned weights loaded")
            except Exception as e:
                print(f"Warning: Failed to load fine-tuned weights: {e}")
                model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        else:
            model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            
        # MobileNetV2 classifier is a Sequential block.
        # The last layer is a Linear layer.
        # We want features from the pooling layer.
        # MobileNetV2 structure: features -> adaptive_avg_pool2d -> classifier
        # We can just use model.features and add pooling.
        
        class MobileNetV2Features(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.features = original_model.features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                
            def forward(self, x):
                x = self.features(x)
                x = self.pool(x)
                x = torch.flatten(x, 1)
                return x
                
        return MobileNetV2Features(model)

class EfficientNetV2B0Extractor(BaseDeepLearningExtractor):
    """EfficientNetV2-S feature extractor (closest to B0/Small)."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), weights_path: Optional[str] = None):
        super().__init__("EfficientNetV2B0", target_size, weights_path)
        
    def _build_model(self, weights_path: Optional[str]) -> nn.Module:
        if weights_path and os.path.exists(weights_path):
            print(f"Loading fine-tuned EfficientNetV2 weights from: {weights_path}")
            model = efficientnet_v2_s(weights=None)
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                model.load_state_dict(state_dict, strict=False)
                print("✓ Fine-tuned weights loaded")
            except Exception as e:
                print(f"Warning: Failed to load fine-tuned weights: {e}")
                model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        else:
            model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            
        # EfficientNetV2 classifier is a Sequential block.
        # We want features before the classifier.
        model.classifier = nn.Identity()
        return model

class ColorHistogramHSconcatResnet50(FeatureExtractor):
    """
    Concatenated feature extractor combining ColorHistogramHS and ResNet50.
    ColorHistogramHS features are weighted more heavily (3x) before concatenation.
    """
    
    def __init__(self, 
                 hist_weight: float = 3.0,
                 bins: int = 32, 
                 hist_target_size: Tuple[int, int] = (128, 128),
                 resnet_target_size: Tuple[int, int] = (224, 224)):
        super().__init__("ColorHistogramHSconcatResnet50")
        self.hist_weight = hist_weight
        self.hist_extractor = ColorHistogramHSExtractor(bins=bins, target_size=hist_target_size)
        self.resnet_extractor = ResNet50Extractor(target_size=resnet_target_size)
        self._feature_dim = 64 + 2048
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract concatenated features with weighted ColorHistogramHS."""
        hist_features = self.hist_extractor.extract(image)
        resnet_features = self.resnet_extractor.extract(image)
        
        weighted_hist = hist_features * self.hist_weight
        concat_features = np.concatenate([weighted_hist, resnet_features])
        
        return l2_normalize(concat_features)
    
    def get_feature_names(self) -> List[str]:
        hist_names = [f"weighted_hist_{name}" for name in self.hist_extractor.get_feature_names()]
        resnet_names = self.resnet_extractor.get_feature_names()
        return hist_names + resnet_names

def extract_features_from_dataset(
    segmented_image_path: str,
    metadata_path: str,
    output_json_path: str,
    extractors: List[FeatureExtractor]
) -> List[dict[str, Any]]:
    """
    Extract features from all segmented images and save to JSON.
    """
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    
    print(f"Found {len(metadata_list)} images in metadata")
    print(f"Applying {len(extractors)} feature extractors: {[e.name for e in extractors]}")
    
    results: List[dict[str, Any]] = []
    
    for idx, metadata in enumerate(metadata_list):
        image_id = metadata['id']
        image_path = os.path.join(segmented_image_path, f"{image_id}.jpg")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping...")
            continue
        
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            print(f"Warning: Failed to read {image_path}, skipping...")
            continue
        
        feature_data: dict[str, Any] = {
            'id': image_id,
            'features': {}
        }
        
        try:
            for extractor in extractors:
                features = extractor.extract(image)
                feature_data['features'][extractor.name.lower()] = {
                    'vector': features.tolist(),
                    'dimension': len(features)
                }
            
            results.append(feature_data)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(metadata_list)} images...")
        
        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            continue
    
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    total_features = 0
    if results:
        total_features = sum(feat['dimension'] for feat in results[0]['features'].values())
    
    print(f"\nFeature extraction complete!")
    print(f"Processed {len(results)} images")
    print(f"Feature types: {list(results[0]['features'].keys()) if results else []}")
    print(f"Total feature dimension: {total_features}")
    print(f"Results saved to: {output_json_path}")
    
    return results

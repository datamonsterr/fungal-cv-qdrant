import os
import json
import cv2
import ssl
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from skimage.feature import hog  # type: ignore
from skimage.filters import gabor_kernel  # type: ignore
from scipy import ndimage as ndi  # type: ignore
from tensorflow.keras.applications import ResNet50, EfficientNetV2B0, MobileNetV2  # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess  # type: ignore
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetv2_preprocess  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input  # type: ignore

ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore


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
        self._feature_dim: int | None = None
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features from image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        gray = cv2.resize(gray, self.target_size)
        
        # Extract HOG features
        features = hog(gray, orientations=self.orientations,  # type: ignore
                      pixels_per_cell=self.pixels_per_cell,
                      cells_per_block=self.cells_per_block,
                      visualize=False, feature_vector=True)
        
        if self._feature_dim is None:
            self._feature_dim = len(features)  # type: ignore
        
        # Apply L2 normalization
        features = l2_normalize(features)  # type: ignore
        return features  # type: ignore
    
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
    
    def __init__(self, frequencies: List[float] | None = None, thetas: List[float] | None = None,
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
                kernel = np.real(gabor_kernel(freq, theta=theta))  # type: ignore
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
            filtered = ndi.convolve(gray, kernel, mode='wrap')  # type: ignore
            # Extract mean and std as features
            features.extend([filtered.mean(), filtered.std()])  # type: ignore
        
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
            # Normalize histogram
            hist = hist / (hist.sum() + 1e-7)
            features.extend(hist)  # type: ignore
        
        # Apply L2 normalization
        return l2_normalize(np.array(features))
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for color histogram."""
        names: List[str] = []
        for channel in ['r', 'g', 'b']:
            for i in range(self.bins):
                names.append(f"hist_{channel}_{i}")
        return names


class ResNet50Extractor(FeatureExtractor):
    """ResNet50 feature extractor (without final classification layer)."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        super().__init__("ResNet50")
        self.target_size = target_size
        
        # Load ResNet50 without top layer
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # type: ignore
        self.model: Any = Model(inputs=base_model.input, outputs=base_model.output)  # type: ignore
        self._feature_dim = 2048  # ResNet50 output dimension
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract ResNet50 features from image."""
        # Resize image
        resized = cv2.resize(image, self.target_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Expand dimensions for batch
        img_array = np.expand_dims(rgb, axis=0)
        
        # Preprocess for ResNet50
        img_array = resnet_preprocess(img_array)  # type: ignore
        
        # Extract features
        features = self.model.predict(img_array, verbose=0)  # type: ignore
        
        # Apply L2 normalization
        return l2_normalize(features.flatten())
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for ResNet50."""
        return [f"resnet50_{i}" for i in range(self._feature_dim)]

class MobileNetV2Extractor(FeatureExtractor):
    """MobileNetV2 feature extractor (without final classification layer)."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        super().__init__("MobileNetV2")
        self.target_size = target_size
        
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')  # type: ignore
        
        self.model: Any = Model(inputs=base_model.input, outputs=base_model.output)  # type: ignore
        self._feature_dim = 1280  # MobileNetV2 output dimension
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract MobileNetV2 features from image."""
        # Resize image
        resized = cv2.resize(image, self.target_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Expand dimensions for batch
        img_array = np.expand_dims(rgb, axis=0)
        
        # Preprocess for MobileNetV2
        img_array = mobilenetv2_preprocess(img_array)  # type: ignore
        
        # Extract features
        features = self.model.predict(img_array, verbose=0)  # type: ignore
        
        # Apply L2 normalization
        return l2_normalize(features.flatten())
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for MobileNetV2."""
        return [f"mobilenetv2_{i}" for i in range(self._feature_dim)]


class EfficientNetV2B0Extractor(FeatureExtractor):
    """EfficientNetV2B0 feature extractor (without final classification layer)."""
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        super().__init__("EfficientNetV2B0")
        self.target_size = target_size

        # Ensure SSL context is set for model downloads
        ssl._create_default_https_context = ssl._create_unverified_context
        
        base_model = EfficientNetV2B0(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        self.model = base_model
        self._feature_dim = 1280

    def extract(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, self.target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(rgb, axis=0)
        img_array = efficientnetv2_preprocess(img_array)
        features = self.model.predict(img_array, verbose=0)
        # Apply L2 normalization
        return l2_normalize(features.flatten())

    def get_feature_names(self) -> List[str]:
        return [f"efficientnetv2b0_{i}" for i in range(self._feature_dim)]


def extract_features_from_dataset(
    segmented_image_path: str,
    metadata_path: str,
    output_json_path: str,
    extractors: List[FeatureExtractor]
) -> List[dict[str, Any]]:
    """
    Extract features from all segmented images and save to JSON.
    
    Args:
        segmented_image_path: Path to directory containing segmented images
        metadata_path: Path to JSON file with image metadata
        output_json_path: Path where the output JSON will be saved
        extractors: List of feature extractor instances to apply
    
    Returns:
        List of dictionaries with extracted features
    """
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    
    print(f"Found {len(metadata_list)} images in metadata")
    print(f"Applying {len(extractors)} feature extractors: {[e.name for e in extractors]}")
    
    # Prepare results storage
    results: List[dict[str, Any]] = []
    
    # Process each image
    for idx, metadata in enumerate(metadata_list):
        image_id = metadata['id']
        image_path = os.path.join(segmented_image_path, f"{image_id}.jpg")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping...")
            continue
        
        # Load image
        image = cv2.imread(image_path)
        # cv2.imread returns ndarray, but check for empty/invalid images
        if image.size == 0:  # type: ignore
            print(f"Warning: Failed to read {image_path}, skipping...")
            continue
        
        # Extract features using all extractors
        feature_data: dict[str, Any] = {
            'id': image_id,
            'features': {}
        }
        
        try:
            for extractor in extractors:
                features = extractor.extract(image)
                
                # Store features as a list under the extractor name
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
    
    # Save to JSON
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate total features
    total_features = 0
    if results:
        total_features = sum(feat['dimension'] for feat in results[0]['features'].values())
    
    print(f"\nFeature extraction complete!")
    print(f"Processed {len(results)} images")
    print(f"Feature types: {list(results[0]['features'].keys()) if results else []}")
    print(f"Total feature dimension: {total_features}")
    print(f"Results saved to: {output_json_path}")
    
    return results


def get_next_version(base_path: str, extension: str = ".json") -> str:
    """
    Get the next version number for a file path.
    
    Args:
        base_path: Base path without version number (e.g., "../Dataset/segmented_features")
        extension: File extension (default: ".json")
    
    Returns:
        Path with next version number (e.g., "../Dataset/segmented_features_2.json")
    """
    import glob
    import re
    
    # Find all existing versioned files
    dir_path = os.path.dirname(base_path)
    file_name = os.path.basename(base_path)
    pattern = f"{file_name}_[0-9]+{extension}"
    
    existing_files = glob.glob(os.path.join(dir_path, pattern))
    
    # Extract version numbers and find the maximum
    max_version = 0
    version_regex = re.compile(rf"{re.escape(file_name)}_(\d+){re.escape(extension)}")
    
    for file_path in existing_files:
        match = version_regex.search(os.path.basename(file_path))
        if match:
            version = int(match.group(1))
            max_version = max(max_version, version)
    
    # Return next version
    next_version = max_version + 1
    return f"{base_path}_{next_version}{extension}"


def main() -> None:
    """Main function to extract features from segmented images."""
    # Base paths - these are the actual existing files
    SEGMENT_IMAGE_PATH = "../Dataset/segmented_image"
    SEGMENT_METADATA_PATH = "../Dataset/segmented_image_metadata.json"
    BASE_OUTPUT_JSON_PATH = "../Dataset/segmented_features"
    
    # Get versioned output path (automatically increments version)
    OUTPUT_JSON_PATH = get_next_version(BASE_OUTPUT_JSON_PATH, extension=".json")
    
    print(f"Using image path: {SEGMENT_IMAGE_PATH}")
    print(f"Using metadata path: {SEGMENT_METADATA_PATH}")
    print(f"Output will be saved to: {OUTPUT_JSON_PATH}\n")
    
    # Initialize feature extractors
    extractors: List[FeatureExtractor] = [
        # HOG with reduced dimensions: ~945 instead of ~3780
        # Increased pixels_per_cell from (8,8) to (16,16)
        HOGExtractor(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)),
        GaborExtractor(frequencies=[0.1, 0.2, 0.3, 0.4], thetas=[0, np.pi/4, np.pi/2, 3*np.pi/4]),
        ColorHistogramExtractor(bins=32),
        ResNet50Extractor(),
        MobileNetV2Extractor(),
        EfficientNetV2B0Extractor()
    ]
    
    # Extract features
    results = extract_features_from_dataset(
        segmented_image_path=SEGMENT_IMAGE_PATH,
        metadata_path=SEGMENT_METADATA_PATH,
        output_json_path=OUTPUT_JSON_PATH,
        extractors=extractors
    )
    
    print(f"\nTotal images processed: {len(results)}")
    if results:
        print(f"Feature types per image: {list(results[0]['features'].keys())}")
        print(f"\nSample feature dimensions:")
        for feat_type, feat_data in results[0]['features'].items():
            print(f"  {feat_type}: {feat_data['dimension']} dimensions")


if __name__ == "__main__":
    main()

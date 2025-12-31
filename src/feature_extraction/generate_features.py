import json
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path

from src.config import (
    SEGMENTED_IMAGE_DIR,
    SEGMENTED_METADATA_PATH,
    FEATURES_JSON_PATH
)
from src.feature_extraction.feature_extractors import (
    ResNet50Extractor,
    MobileNetV2Extractor,
    EfficientNetV2B0Extractor,
    HOGExtractor,
    GaborExtractor,
    ColorHistogramExtractor,
    ColorHistogramHSExtractor
)

def generate_features(
    image_dir: Path = SEGMENTED_IMAGE_DIR,
    metadata_path: Path = SEGMENTED_METADATA_PATH,
    output_path: Path = FEATURES_JSON_PATH
) -> None:
    
    if not metadata_path.exists():
        print(f"Error: Metadata file {metadata_path} not found.")
        return

    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
        
    print(f"Found {len(metadata_list)} images in metadata.")
    
    # Initialize extractors
    extractors = [
        ResNet50Extractor(),
        MobileNetV2Extractor(),
        EfficientNetV2B0Extractor(),
        HOGExtractor(),
        GaborExtractor(),
        ColorHistogramExtractor(),
        ColorHistogramHSExtractor()
    ]
    
    features_data = []
    
    for item in tqdm(metadata_list, desc="Extracting features"):
        image_id = item['id']
        image_path = image_dir / f"{image_id}.jpg"
        
        if not image_path.exists():
            continue
            
        # Read image
        # Note: Deep learning extractors usually handle reading/transforming internally 
        # or expect a path/PIL image. 
        # My BaseDeepLearningExtractor.extract takes an image path.
        # The traditional ones take a numpy array (cv2 image).
        
        img_cv2 = cv2.imread(str(image_path))
        if img_cv2 is None:
            continue
            
        record = {
            'id': image_id,
            'features': {}
        }
        
        for extractor in extractors:
            try:
                if hasattr(extractor, 'extract'):
                    # Check signature or type of extractor
                    # My refactored extractors:
                    # DL ones take image_path (str)
                    # Traditional ones take image (numpy array)
                    
                    # This is a bit messy, let's check the base class or implementation
                    # In my refactor:
                    # BaseDeepLearningExtractor.extract(self, image_path: str)
                    # BaseFeatureExtractor.extract(self, image: np.ndarray)
                    
                    # I need to distinguish them.
                    # I can check if it inherits from BaseDeepLearningExtractor
                    
                    from src.feature_extraction.feature_extractors import BaseDeepLearningExtractor
                    
                    if isinstance(extractor, BaseDeepLearningExtractor):
                        vector = extractor.extract(str(image_path))
                    else:
                        vector = extractor.extract(img_cv2)
                        
                    # Convert to list for JSON serialization
                    if isinstance(vector, np.ndarray):
                        vector = vector.tolist()
                    elif isinstance(vector, torch.Tensor):
                        vector = vector.cpu().numpy().tolist()
                        
                    record['features'][extractor.name.lower()] = {
                        'vector': vector,
                        'dimension': len(vector)
                    }
            except Exception as e:
                print(f"Error extracting {extractor.name} for {image_id}: {e}")
                
        features_data.append(record)
        
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(features_data, f)
        
    print(f"Features saved to {output_path}")

if __name__ == "__main__":
    generate_features()

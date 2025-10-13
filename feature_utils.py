"""
Utility functions for working with JSON feature files.
"""
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple


def load_features(json_path: str) -> List[Dict[str, Any]]:
    """
    Load features from JSON file.
    
    Args:
        json_path: Path to JSON file containing features
        
    Returns:
        List of feature dictionaries
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def get_feature_vector(
    feature_data: Dict[str, Any],
    feature_types: List[str] | None = None
) -> np.ndarray:
    """
    Extract feature vector from a single image's feature data.
    
    Args:
        feature_data: Feature dictionary for one image
        feature_types: List of feature types to include (e.g., ['hog', 'gabor'])
                      If None, includes all feature types
                      
    Returns:
        Concatenated feature vector as numpy array
    """
    if feature_types is None:
        feature_types = list(feature_data['features'].keys())
    
    vectors: List[List[float]] = []
    for feat_type in feature_types:
        if feat_type in feature_data['features']:
            vectors.append(feature_data['features'][feat_type]['vector'])
    
    return np.concatenate(vectors) if vectors else np.array([])  # type: ignore


def features_to_dataframe(
    features: List[Dict[str, Any]],
    feature_types: List[str] | None = None
) -> pd.DataFrame:
    """
    Convert JSON features to pandas DataFrame.
    
    Args:
        features: List of feature dictionaries
        feature_types: List of feature types to include
                      If None, includes all feature types
                      
    Returns:
        DataFrame with columns: id, feature vectors (expanded)
    """
    rows: List[Dict[str, Any]] = []
    
    for feat_data in features:
        row: Dict[str, Any] = {'id': feat_data['id']}
        
        # Get concatenated vector
        vector = get_feature_vector(feat_data, feature_types)
        
        # Add each dimension as a column
        for i, val in enumerate(vector):
            row[f'feature_{i}'] = float(val)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def features_to_numpy(
    features: List[Dict[str, Any]],
    feature_types: List[str] | None = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert JSON features to numpy array.
    
    Args:
        features: List of feature dictionaries
        feature_types: List of feature types to include
                      If None, includes all feature types
                      
    Returns:
        Tuple of (feature_matrix, image_ids)
        - feature_matrix: 2D numpy array of shape (n_images, n_features)
        - image_ids: List of image IDs corresponding to rows
    """
    vectors: List[np.ndarray] = []
    ids: List[str] = []
    
    for feat_data in features:
        vector = get_feature_vector(feat_data, feature_types)
        vectors.append(vector)
        ids.append(feat_data['id'])
    
    return np.array(vectors), ids  # type: ignore


def get_feature_info(features: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Get information about feature dimensions.
    
    Args:
        features: List of feature dictionaries
        
    Returns:
        Dictionary mapping feature type to dimension
    """
    if not features:
        return {}
    
    info: Dict[str, int] = {}
    for feat_type, feat_data in features[0]['features'].items():
        info[feat_type] = int(feat_data['dimension'])
    
    return info


def filter_features(
    features: List[Dict[str, Any]],
    image_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    Filter features by image IDs.
    
    Args:
        features: List of feature dictionaries
        image_ids: List of image IDs to keep
        
    Returns:
        Filtered list of feature dictionaries
    """
    id_set = set(image_ids)
    return [f for f in features if f['id'] in id_set]


def combine_feature_types(
    features: List[Dict[str, Any]],
    feature_types: List[str]
) -> List[Dict[str, Any]]:
    """
    Create new feature list with only specified feature types.
    
    Args:
        features: List of feature dictionaries
        feature_types: List of feature types to keep
        
    Returns:
        New list with filtered feature types
    """
    result: List[Dict[str, Any]] = []
    
    for feat_data in features:
        new_features: Dict[str, Any] = {}
        for feat_type in feature_types:
            if feat_type in feat_data['features']:
                new_features[feat_type] = feat_data['features'][feat_type]
        
        result.append({
            'id': feat_data['id'],
            'features': new_features
        })
    
    return result


def save_features(features: List[Dict[str, Any]], json_path: str) -> None:
    """
    Save features to JSON file.
    
    Args:
        features: List of feature dictionaries
        json_path: Path where JSON will be saved
    """
    with open(json_path, 'w') as f:
        json.dump(features, f, indent=2)


def print_feature_summary(features: List[Dict[str, Any]]) -> None:
    """
    Print a summary of the features.
    
    Args:
        features: List of feature dictionaries
    """
    if not features:
        print("No features found.")
        return
    
    print(f"Total images: {len(features)}")
    print(f"\nFeature types and dimensions:")
    
    info = get_feature_info(features)
    total_dim = 0
    
    for feat_type, dim in info.items():
        print(f"  {feat_type:20s}: {dim:6d} dimensions")
        total_dim += dim
    
    print(f"\nTotal feature dimension: {total_dim}")


# Example usage
if __name__ == "__main__":
    # Load features
    features = load_features("../Dataset/segmented_features.json")
    
    # Print summary
    print("=== Feature Summary ===")
    print_feature_summary(features)
    
    # Get feature info
    print("\n=== Feature Info ===")
    info = get_feature_info(features)
    print(info)
    
    # Convert to numpy array (using only HOG and Gabor)
    print("\n=== Converting to NumPy (HOG + Gabor only) ===")
    X, ids = features_to_numpy(features, feature_types=['hog', 'gabor'])
    print(f"Shape: {X.shape}")
    print(f"First 3 IDs: {ids[:3]}")
    
    # Convert to DataFrame (all features)
    print("\n=== Converting to DataFrame (all features) ===")
    df = features_to_dataframe(features)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:5]}...")
    
    # Filter by specific feature types
    print("\n=== Filtering to ResNet50 only ===")
    resnet_only = combine_feature_types(features, ['resnet50'])
    print_feature_summary(resnet_only)

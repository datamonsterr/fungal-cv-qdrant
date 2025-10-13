"""
Utility functions for querying Qdrant vector database.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Optional 
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue 
from feature_extractors import FeatureExtractor


def get_image_features(
    image_path: str,
    extractor: FeatureExtractor
) -> np.ndarray:
    """
    Extract features from an image using the specified extractor.
    
    Args:
        image_path: Path to the image file
        extractor: Feature extractor instance
        
    Returns:
        Feature vector as numpy array
    """
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        raise ValueError(f"Failed to read image from {image_path}")
    
    features = extractor.extract(image)
    return features


def build_filter(
    environment: Optional[str] = None,
    angle: Optional[str] = None,
    strain: Optional[str] = None,
    specy: Optional[str] = None,
    parent_id: Optional[str] = None
) -> Optional[Filter]:
    """
    Build a Qdrant filter based on metadata conditions.
    
    Args:
        environment: Filter by environment (e.g., "CYA", "MEA")
        angle: Filter by viewing angle ("ob" or "rev")
        strain: Filter by strain (e.g., "DTO 123-A1")
        specy: Filter by species name
        parent_id: Filter by parent image ID
        
    Returns:
        Qdrant Filter object or None if no filters specified
    """
    conditions = []
    
    if environment is not None:
        conditions.append(
            FieldCondition(key="environment", match=MatchValue(value=environment))
        )
    
    if angle is not None:
        conditions.append(
            FieldCondition(key="angle", match=MatchValue(value=angle))
        )
    
    if strain is not None:
        conditions.append(
            FieldCondition(key="strain", match=MatchValue(value=strain))
        )
    
    if specy is not None:
        conditions.append(
            FieldCondition(key="specy", match=MatchValue(value=specy))
        )
    
    if parent_id is not None:
        conditions.append(
            FieldCondition(key="parent_id", match=MatchValue(value=parent_id))
        )
    
    if not conditions:
        return None
    
    return Filter(must=conditions)


def find_nearest_neighbors_by_id(
    client: QdrantClient,
    collection_name: str,
    query_image_id: str,
    feature_type: str,
    num_neighbors: int = 10,
    environment: Optional[str] = None,
    angle: Optional[str] = None,
    strain: Optional[str] = None,
    specy: Optional[str] = None,
    exclude_self: bool = True
) -> List[Dict[str, Any]]:
    """
    Find nearest neighbors using an image ID already in the collection.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to query
        query_image_id: ID of the image to use as query
        feature_type: Type of features to use ("hog", "gabor", "colorhistogram", "resnet50")
        num_neighbors: Number of nearest neighbors to return
        environment: Filter by environment
        angle: Filter by viewing angle
        strain: Filter by strain
        specy: Filter by species
        exclude_self: Whether to exclude the query image from results
        
    Returns:
        List of dictionaries containing neighbor information
    """
    # First, retrieve the query image's features
    search_result = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="image_id", match=MatchValue(value=query_image_id))]
        ),
        limit=1,
        with_vectors=True
    )
    
    if not search_result[0]:
        raise ValueError(f"Image with ID {query_image_id} not found in collection")
    
    query_point = search_result[0][0]
    query_vector = query_point.vector.get(feature_type)  # type: ignore
    
    if query_vector is None:
        available_types = list(query_point.vector.keys())  # type: ignore
        raise ValueError(
            f"Feature type '{feature_type}' not found. Available types: {available_types}"
        )
    
    # Build filter for search
    search_filter = build_filter(
        environment=environment,
        angle=angle,
        strain=strain,
        specy=specy
    )
    
    # Adjust limit if excluding self
    search_limit = num_neighbors + 1 if exclude_self else num_neighbors
    
    # Search for nearest neighbors
    results = client.search(
        collection_name=collection_name,
        query_vector=(feature_type, query_vector),
        query_filter=search_filter,
        limit=search_limit,
        with_payload=True
    )
    
    # Process results
    neighbors = []
    for result in results:
        # Skip self if requested
        if exclude_self and result.payload.get('image_id') == query_image_id:  # type: ignore
            continue
        
        neighbor_data = {
            'image_id': result.payload.get('image_id'),  # type: ignore
            'score': result.score,
            'distance': 1.0 - result.score,  # Assuming cosine similarity
            'strain': result.payload.get('strain'),  # type: ignore
            'environment': result.payload.get('environment'),  # type: ignore
            'angle': result.payload.get('angle'),  # type: ignore
            'specy': result.payload.get('specy'),  # type: ignore
            'parent_id': result.payload.get('parent_id'),  # type: ignore
            'segment_index': result.payload.get('segment_index'),  # type: ignore
            'bbox': result.payload.get('bbox'),  # type: ignore
        }
        neighbors.append(neighbor_data)
        
        if len(neighbors) >= num_neighbors:
            break
    
    return neighbors


def find_nearest_neighbors_by_image(
    client: QdrantClient,
    collection_name: str,
    image_path: str,
    extractor: FeatureExtractor,
    feature_type: str,
    num_neighbors: int = 10,
    environment: Optional[str] = None,
    angle: Optional[str] = None,
    strain: Optional[str] = None,
    specy: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Find nearest neighbors using a new image file (not in collection).
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to query
        image_path: Path to the query image file
        extractor: Feature extractor instance matching the feature_type
        feature_type: Type of features to use ("hog", "gabor", "colorhistogram", "resnet50")
        num_neighbors: Number of nearest neighbors to return
        environment: Filter by environment
        angle: Filter by viewing angle
        strain: Filter by strain
        specy: Filter by species
        
    Returns:
        List of dictionaries containing neighbor information
    """
    # Extract features from the query image
    query_vector = get_image_features(image_path, extractor)
    
    # Build filter for search
    search_filter = build_filter(
        environment=environment,
        angle=angle,
        strain=strain,
        specy=specy
    )
    
    # Search for nearest neighbors
    results = client.search(
        collection_name=collection_name,
        query_vector=(feature_type, query_vector.tolist()),
        query_filter=search_filter,
        limit=num_neighbors,
        with_payload=True
    )
    
    # Process results
    neighbors = []
    for result in results:
        neighbor_data = {
            'image_id': result.payload.get('image_id'),  # type: ignore
            'score': result.score,
            'distance': 1.0 - result.score,  # Assuming cosine similarity
            'strain': result.payload.get('strain'),  # type: ignore
            'environment': result.payload.get('environment'),  # type: ignore
            'angle': result.payload.get('angle'),  # type: ignore
            'specy': result.payload.get('specy'),  # type: ignore
            'parent_id': result.payload.get('parent_id'),  # type: ignore
            'segment_index': result.payload.get('segment_index'),  # type: ignore
            'bbox': result.payload.get('bbox'),  # type: ignore
        }
        neighbors.append(neighbor_data)
    
    return neighbors


def get_image_metadata(
    client: QdrantClient,
    collection_name: str,
    image_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve metadata for a specific image by ID.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        image_id: ID of the image to retrieve metadata for
        
    Returns:
        Dictionary with image metadata or None if not found
    """
    search_result = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="image_id", match=MatchValue(value=image_id))]
        ),
        limit=1,
        with_payload=True
    )
    
    if not search_result[0]:
        return None
    
    point = search_result[0][0]
    payload = point.payload  # type: ignore
    
    return {
        'image_id': payload.get('image_id'),
        'strain': payload.get('strain'),
        'environment': payload.get('environment'),
        'angle': payload.get('angle'),
        'specy': payload.get('specy'),
        'parent_id': payload.get('parent_id'),
        'segment_index': payload.get('segment_index'),
        'bbox': payload.get('bbox'),
    }


def get_collection_stats(
    client: QdrantClient,
    collection_name: str
) -> Dict[str, Any]:
    """
    Get statistics about the collection.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        
    Returns:
        Dictionary with collection statistics
    """
    collection_info = client.get_collection(collection_name=collection_name)
    
    # Get sample point to determine feature types
    sample = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_vectors=True,
        with_payload=True
    )
    
    stats = {
        'total_points': collection_info.points_count,
        'vector_types': [],
        'vector_dimensions': {},
    }
    
    if sample[0]:
        point = sample[0][0]
        if hasattr(point, 'vector') and isinstance(point.vector, dict):
            stats['vector_types'] = list(point.vector.keys())
            for vec_name, vec_data in point.vector.items():
                if isinstance(vec_data, list):
                    stats['vector_dimensions'][vec_name] = len(vec_data)
    
    return stats


def print_neighbors(neighbors: List[Dict[str, Any]], show_bbox: bool = False) -> None:
    """
    Pretty print nearest neighbors results.
    
    Args:
        neighbors: List of neighbor dictionaries
        show_bbox: Whether to show bounding box information
    """
    if not neighbors:
        print("No neighbors found.")
        return
    
    print(f"\nFound {len(neighbors)} neighbors:")
    print("-" * 100)
    
    for i, neighbor in enumerate(neighbors, 1):
        print(f"{i}. Image ID: {neighbor['image_id']}")
        print(f"   Score: {neighbor['score']:.4f} | Distance: {neighbor['distance']:.4f}")
        print(f"   Species: {neighbor['specy']} | Strain: {neighbor['strain']}")
        print(f"   Environment: {neighbor['environment']} | Angle: {neighbor['angle']}")
        print(f"   Parent ID: {neighbor['parent_id']} | Segment: {neighbor['segment_index']}")
        
        if show_bbox and neighbor['bbox']:
            bbox = neighbor['bbox']
            print(f"   BBox: ({bbox.get('xmin')}, {bbox.get('ymin')}) - ({bbox.get('xmax')}, {bbox.get('ymax')})")
        
        print("-" * 100)


def visualize_neighbors(
    query_image_path: str,
    neighbors: List[Dict[str, Any]],
    segmented_image_dir: str,
    output_path: str,
    max_neighbors: int = 5,
    thumbnail_size: tuple[int, int] = (200, 200),
    text_color: tuple[int, int, int] = (0, 0, 0),
    bg_color: tuple[int, int, int] = (255, 255, 255),
    border_color: tuple[int, int, int] = (200, 200, 200),
    query_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create a visualization showing the query image and its k nearest neighbors.
    
    Args:
        query_image_path: Path to the query image
        neighbors: List of neighbor dictionaries from find_nearest_neighbors_*
        segmented_image_dir: Directory containing segmented images
        output_path: Path to save the output visualization
        max_neighbors: Maximum number of neighbors to display
        thumbnail_size: Size to resize each image to (width, height)
        text_color: RGB color for text (default: black)
        bg_color: RGB color for background (default: white)
        border_color: RGB color for image borders (default: light gray)
        query_metadata: Optional metadata dictionary for the query image
    """
    import os
    from PIL import Image, ImageDraw, ImageFont
    
    # Limit neighbors
    neighbors = neighbors[:max_neighbors]
    
    # Load query image
    query_img = cv2.imread(query_image_path)
    if query_img is None or query_img.size == 0:
        raise ValueError(f"Failed to read query image from {query_image_path}")
    
    # Resize query image
    query_img_resized = cv2.resize(query_img, thumbnail_size)
    
    # Calculate layout dimensions
    num_images = len(neighbors) + 1  # +1 for query image
    text_height = 140  # Height for text below each image
    border_width = 10
    padding = 20
    
    img_width = thumbnail_size[0]
    img_height = thumbnail_size[1]
    
    # Calculate grid layout
    images_per_row = min(4, num_images)
    num_rows = (num_images + images_per_row - 1) // images_per_row
    
    # Canvas dimensions
    canvas_width = images_per_row * (img_width + padding) + padding
    canvas_height = num_rows * (img_height + text_height + padding) + padding
    
    # Create canvas with white background
    canvas_bgr = np.full((canvas_height, canvas_width, 3), bg_color, dtype=np.uint8)
    
    # Convert to PIL for better text rendering
    from PIL import Image, ImageDraw, ImageFont
    canvas_pil = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas_pil)
    
    # Try to load Arial font, fall back to default if not available
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 14)
        font_normal = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 11)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 10)
    except:
        try:
            font_title = ImageFont.truetype("arial.ttf", 14)
            font_normal = ImageFont.truetype("arial.ttf", 11)
            font_small = ImageFont.truetype("arial.ttf", 10)
        except:
            font_title = ImageFont.load_default()
            font_normal = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # Helper function to add text
    def put_text(text: str, position: tuple[int, int], font: Any, 
                 color: Optional[tuple[int, int, int]] = None) -> int:
        """Put text and return next y position."""
        if color is None:
            color = text_color
        x, y = position
        draw.text((x, y), text, font=font, fill=color)
        bbox = draw.textbbox((x, y), text, font=font)
        return int(bbox[3] + 5)
    
    # Place query image
    x_offset = padding
    y_offset = padding
    
    # Add green border to query image
    query_with_border = cv2.copyMakeBorder(
        query_img_resized, border_width, border_width, border_width, border_width,
        cv2.BORDER_CONSTANT, value=(0, 255, 0)  # Green border for query
    )
    
    # Convert to PIL and paste on canvas
    query_pil = Image.fromarray(cv2.cvtColor(query_with_border, cv2.COLOR_BGR2RGB))
    canvas_pil.paste(query_pil, (x_offset, y_offset))
    
    # Add text below query image
    y_end = y_offset + query_with_border.shape[0]
    text_y = y_end + 10
    text_x = x_offset + 5
    
    # Title
    text_y = put_text("QUERY IMAGE", (text_x, text_y), font_title, (0, 128, 0))
    
    # Add query metadata if provided
    if query_metadata:
        # Image ID
        image_id = query_metadata.get('image_id', 'unknown')
        if len(image_id) > 28:
            image_id = image_id[:25] + "..."
        text_y = put_text(f"ID: {image_id}", (text_x, text_y), font_small)
        
        # Species
        specy = query_metadata.get('specy', 'unknown')
        if len(specy) > 28:
            specy = specy[:25] + "..."
        text_y = put_text(f"Species: {specy}", (text_x, text_y), font_normal)
        
        # Strain
        strain = query_metadata.get('strain', 'unknown')
        if len(strain) > 28:
            strain = strain[:25] + "..."
        text_y = put_text(f"Strain: {strain}", (text_x, text_y), font_normal)
        
        # Environment
        environment = query_metadata.get('environment', 'unknown')
        text_y = put_text(f"Env: {environment}", (text_x, text_y), font_normal)
        
        # Angle
        angle = query_metadata.get('angle', 'unknown')
        text_y = put_text(f"Angle: {angle}", (text_x, text_y), font_normal)
    
    # Move to next position
    col = 1
    row = 0
    
    # Place neighbor images
    for idx, neighbor in enumerate(neighbors):
        # Calculate position
        if col >= images_per_row:
            col = 0
            row += 1
        
        x_offset = padding + col * (img_width + padding)
        y_offset = padding + row * (img_height + text_height + padding)
        
        # Load neighbor image
        neighbor_img_path = os.path.join(segmented_image_dir, f"{neighbor['image_id']}.jpg")
        neighbor_img = cv2.imread(neighbor_img_path)
        
        if neighbor_img is None or neighbor_img.size == 0:
            # If image not found, create placeholder
            neighbor_img = np.full((thumbnail_size[1], thumbnail_size[0], 3), (180, 180, 180), dtype=np.uint8)
        else:
            neighbor_img = cv2.resize(neighbor_img, thumbnail_size)
        
        # Add border to neighbor image
        neighbor_with_border = cv2.copyMakeBorder(
            neighbor_img, border_width, border_width, border_width, border_width,
            cv2.BORDER_CONSTANT, value=border_color
        )
        
        # Convert to PIL and paste on canvas
        y_end = y_offset + neighbor_with_border.shape[0]
        x_end = x_offset + neighbor_with_border.shape[1]
        
        if y_end <= canvas_height and x_end <= canvas_width:
            neighbor_pil = Image.fromarray(cv2.cvtColor(neighbor_with_border, cv2.COLOR_BGR2RGB))
            canvas_pil.paste(neighbor_pil, (x_offset, y_offset))
            
            # Add text below neighbor image
            text_y = y_end + 10
            text_x = x_offset + 5
            
            # Rank number
            text_y = put_text(f"#{idx + 1}", (text_x, text_y), font_title, (200, 0, 0))
            
            # Image ID
            image_id = neighbor.get('image_id', 'unknown')
            if len(image_id) > 28:
                image_id = image_id[:25] + "..."
            text_y = put_text(f"ID: {image_id}", (text_x, text_y), font_small)
            
            # Species
            specy = neighbor.get('specy', 'unknown')
            if len(specy) > 28:
                specy = specy[:25] + "..."
            text_y = put_text(f"Species: {specy}", (text_x, text_y), font_normal)
            
            # Strain
            strain = neighbor.get('strain', 'unknown')
            if len(strain) > 28:
                strain = strain[:25] + "..."
            text_y = put_text(f"Strain: {strain}", (text_x, text_y), font_normal)
            
            # Environment
            environment = neighbor.get('environment', 'unknown')
            text_y = put_text(f"Env: {environment}", (text_x, text_y), font_normal)
            
            angle = neighbor.get('angle', 'unknown')
            text_y = put_text(f"Angle: {angle}", (text_x, text_y), font_normal)
            
            # Similarity score
            score = neighbor.get('score', 0.0)
            text_y = put_text(f"Score: {score:.4f}", (text_x, text_y), font_title, (0, 100, 0))
        
        col += 1
    
    # Convert back to OpenCV format and save
    canvas_bgr = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, canvas_bgr)
    print(f"Visualization saved to: {output_path}")

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.experiments.feature_extraction.feature_extractors import FeatureExtractor


def get_image_features(image_path: str, extractor: FeatureExtractor) -> np.ndarray:
    """
    Extract features from an image using the specified extractor.
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
    parent_id: Optional[str] = None,
    exclude_environment: Optional[str] = None,
    exclude_strain: Optional[str] = None,
) -> Optional[Filter]:
    """
    Build a Qdrant filter based on metadata conditions.
    """
    conditions = []
    exclude_conditions = []

    if environment is not None:
        conditions.append(
            FieldCondition(key="environment", match=MatchValue(value=environment))
        )

    if exclude_environment is not None:
        exclude_conditions.append(
            FieldCondition(
                key="environment", match=MatchValue(value=exclude_environment)
            )
        )

    if exclude_strain is not None:
        exclude_conditions.append(
            FieldCondition(key="strain", match=MatchValue(value=exclude_strain))
        )

    if angle is not None:
        conditions.append(FieldCondition(key="angle", match=MatchValue(value=angle)))

    if strain is not None:
        conditions.append(FieldCondition(key="strain", match=MatchValue(value=strain)))

    if specy is not None:
        conditions.append(FieldCondition(key="specy", match=MatchValue(value=specy)))

    if parent_id is not None:
        conditions.append(
            FieldCondition(key="parent_id", match=MatchValue(value=parent_id))
        )

    if not conditions and not exclude_conditions:
        return None

    if exclude_conditions:
        return Filter(
            must=conditions if conditions else None, must_not=exclude_conditions
        )
    else:
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
    exclude_self: bool = True,
    exclude_environment: Optional[str] = None,
    exclude_strain: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Find nearest neighbors using an image ID already in the collection.
    """
    search_result = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="image_id", match=MatchValue(value=query_image_id))
            ]
        ),
        limit=1,
        with_vectors=True,
    )

    if not search_result[0]:
        raise ValueError(f"Image with ID {query_image_id} not found in collection")

    query_point = search_result[0][0]
    query_vector = query_point.vector.get(feature_type)

    if query_vector is None:
        available_types = list(query_point.vector.keys())
        raise ValueError(
            f"Feature type '{feature_type}' not found. Available types: {available_types}"
        )

    search_filter = build_filter(
        environment=environment,
        angle=angle,
        strain=strain,
        specy=specy,
        exclude_environment=exclude_environment,
        exclude_strain=exclude_strain,
    )

    search_limit = num_neighbors + 1 if exclude_self else num_neighbors

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=feature_type,
        query_filter=search_filter,
        limit=search_limit,
        with_payload=True,
    )
    results = response.points

    neighbors = []
    for result in results:
        if exclude_self and result.payload.get("image_id") == query_image_id:
            continue

        neighbor_data = {
            "image_id": result.payload.get("image_id"),
            "score": result.score,
            "distance": 1.0 - result.score,
            "strain": result.payload.get("strain"),
            "environment": result.payload.get("environment"),
            "angle": result.payload.get("angle"),
            "specy": result.payload.get("specy"),
            "parent_id": result.payload.get("parent_id"),
            "segment_index": result.payload.get("segment_index"),
            "bbox": result.payload.get("bbox"),
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
    specy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Find nearest neighbors using a new image file (not in collection).
    """
    query_vector = get_image_features(image_path, extractor)

    search_filter = build_filter(
        environment=environment, angle=angle, strain=strain, specy=specy
    )

    results = client.search(
        collection_name=collection_name,
        query_vector=(feature_type, query_vector.tolist()),
        query_filter=search_filter,
        limit=num_neighbors,
        with_payload=True,
    )

    neighbors = []
    for result in results:
        neighbor_data = {
            "image_id": result.payload.get("image_id"),
            "score": result.score,
            "distance": 1.0 - result.score,
            "strain": result.payload.get("strain"),
            "environment": result.payload.get("environment"),
            "angle": result.payload.get("angle"),
            "specy": result.payload.get("specy"),
            "parent_id": result.payload.get("parent_id"),
            "segment_index": result.payload.get("segment_index"),
            "bbox": result.payload.get("bbox"),
        }
        neighbors.append(neighbor_data)

    return neighbors


def get_image_metadata(
    client: QdrantClient, collection_name: str, image_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve metadata for a specific image by ID.
    """
    search_result = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="image_id", match=MatchValue(value=image_id))]
        ),
        limit=1,
        with_payload=True,
    )

    if not search_result[0]:
        return None

    point = search_result[0][0]
    payload = point.payload

    return {
        "image_id": payload.get("image_id"),
        "strain": payload.get("strain"),
        "environment": payload.get("environment"),
        "angle": payload.get("angle"),
        "specy": payload.get("specy"),
        "parent_id": payload.get("parent_id"),
        "segment_index": payload.get("segment_index"),
        "bbox": payload.get("bbox"),
    }


def get_collection_stats(client: QdrantClient, collection_name: str) -> Dict[str, Any]:
    """
    Get statistics about the collection.
    """
    collection_info = client.get_collection(collection_name=collection_name)

    sample = client.scroll(
        collection_name=collection_name, limit=1, with_vectors=True, with_payload=True
    )

    stats = {
        "total_points": collection_info.points_count,
        "vector_types": [],
        "vector_dimensions": {},
    }

    if sample[0]:
        point = sample[0][0]
        if hasattr(point, "vector") and isinstance(point.vector, dict):
            stats["vector_types"] = list(point.vector.keys())
            for vec_name, vec_data in point.vector.items():
                if isinstance(vec_data, list):
                    stats["vector_dimensions"][vec_name] = len(vec_data)

    return stats


def print_neighbors(neighbors: List[Dict[str, Any]], show_bbox: bool = False) -> None:
    """
    Pretty print nearest neighbors results.
    """
    if not neighbors:
        print("No neighbors found.")
        return

    print(f"\nFound {len(neighbors)} neighbors:")
    print("-" * 100)

    for i, neighbor in enumerate(neighbors, 1):
        print(f"{i}. Image ID: {neighbor['image_id']}")
        print(
            f"   Score: {neighbor['score']:.4f} | Distance: {neighbor['distance']:.4f}"
        )
        print(f"   Species: {neighbor['specy']} | Strain: {neighbor['strain']}")
        print(f"   Environment: {neighbor['environment']} | Angle: {neighbor['angle']}")
        print(
            f"   Parent ID: {neighbor['parent_id']} | Segment: {neighbor['segment_index']}"
        )

        if show_bbox and neighbor["bbox"]:
            bbox = neighbor["bbox"]
            print(
                f"   BBox: ({bbox.get('xmin')}, {bbox.get('ymin')}) - ({bbox.get('xmax')}, {bbox.get('ymax')})"
            )

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
    query_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Create a visualization showing the query image and its k nearest neighbors.
    """
    import os

    from PIL import Image, ImageDraw, ImageFont

    neighbors = neighbors[:max_neighbors]

    query_img = cv2.imread(query_image_path)
    if query_img is None or query_img.size == 0:
        raise ValueError(f"Failed to read query image from {query_image_path}")

    query_img_resized = cv2.resize(query_img, thumbnail_size)

    num_images = len(neighbors) + 1
    text_height = 140
    border_width = 10
    padding = 20

    img_width = thumbnail_size[0]
    img_height = thumbnail_size[1]

    images_per_row = min(4, num_images)
    num_rows = (num_images + images_per_row - 1) // images_per_row

    canvas_width = images_per_row * (img_width + padding) + padding
    canvas_height = num_rows * (img_height + text_height + padding) + padding

    canvas_bgr = np.full((canvas_height, canvas_width, 3), bg_color, dtype=np.uint8)

    canvas_pil = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas_pil)

    try:
        font_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 14
        )
        font_normal = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 11
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 10
        )
    except Exception:
        try:
            font_title = ImageFont.truetype("arial.ttf", 14)
            font_normal = ImageFont.truetype("arial.ttf", 11)
            font_small = ImageFont.truetype("arial.ttf", 10)
        except Exception:
            font_title = ImageFont.load_default()
            font_normal = ImageFont.load_default()
            font_small = ImageFont.load_default()

    def put_text(
        text: str,
        position: tuple[int, int],
        font: Any,
        color: Optional[tuple[int, int, int]] = None,
    ) -> int:
        if color is None:
            color = text_color
        x, y = position
        draw.text((x, y), text, font=font, fill=color)
        bbox = draw.textbbox((x, y), text, font=font)
        return int(bbox[3] + 5)

    x_offset = padding
    y_offset = padding

    query_with_border = cv2.copyMakeBorder(
        query_img_resized,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_CONSTANT,
        value=(0, 255, 0),
    )

    query_pil = Image.fromarray(cv2.cvtColor(query_with_border, cv2.COLOR_BGR2RGB))
    canvas_pil.paste(query_pil, (x_offset, y_offset))

    y_end = y_offset + query_with_border.shape[0]
    text_y = y_end + 10
    text_x = x_offset + 5

    text_y = put_text("QUERY IMAGE", (text_x, text_y), font_title, (0, 128, 0))

    if query_metadata:
        image_id = query_metadata.get("image_id", "unknown")
        if len(image_id) > 28:
            image_id = image_id[:25] + "..."
        text_y = put_text(f"ID: {image_id}", (text_x, text_y), font_small)

        specy = query_metadata.get("specy", "unknown")
        if len(specy) > 28:
            specy = specy[:25] + "..."
        text_y = put_text(f"Species: {specy}", (text_x, text_y), font_normal)

        strain = query_metadata.get("strain", "unknown")
        if len(strain) > 28:
            strain = strain[:25] + "..."
        text_y = put_text(f"Strain: {strain}", (text_x, text_y), font_normal)

        environment = query_metadata.get("environment", "unknown")
        text_y = put_text(f"Env: {environment}", (text_x, text_y), font_normal)

        angle = query_metadata.get("angle", "unknown")
        text_y = put_text(f"Angle: {angle}", (text_x, text_y), font_normal)

    col = 1
    row = 0

    for idx, neighbor in enumerate(neighbors):
        if col >= images_per_row:
            col = 0
            row += 1

        x_offset = padding + col * (img_width + padding)
        y_offset = padding + row * (img_height + text_height + padding)

        neighbor_img_path = os.path.join(
            segmented_image_dir, f"{neighbor['image_id']}.jpg"
        )
        neighbor_img = cv2.imread(neighbor_img_path)

        if neighbor_img is None or neighbor_img.size == 0:
            neighbor_img = np.full(
                (thumbnail_size[1], thumbnail_size[0], 3),
                (180, 180, 180),
                dtype=np.uint8,
            )
        else:
            neighbor_img = cv2.resize(neighbor_img, thumbnail_size)

        neighbor_with_border = cv2.copyMakeBorder(
            neighbor_img,
            border_width,
            border_width,
            border_width,
            border_width,
            cv2.BORDER_CONSTANT,
            value=border_color,
        )

        y_end = y_offset + neighbor_with_border.shape[0]
        x_end = x_offset + neighbor_with_border.shape[1]

        if y_end <= canvas_height and x_end <= canvas_width:
            neighbor_pil = Image.fromarray(
                cv2.cvtColor(neighbor_with_border, cv2.COLOR_BGR2RGB)
            )
            canvas_pil.paste(neighbor_pil, (x_offset, y_offset))

            text_y = y_end + 10
            text_x = x_offset + 5

            text_y = put_text(f"#{idx + 1}", (text_x, text_y), font_title, (200, 0, 0))

            image_id = neighbor.get("image_id", "unknown")
            if len(image_id) > 28:
                image_id = image_id[:25] + "..."
            text_y = put_text(f"ID: {image_id}", (text_x, text_y), font_small)

            specy = neighbor.get("specy", "unknown")
            if len(specy) > 28:
                specy = specy[:25] + "..."
            text_y = put_text(f"Species: {specy}", (text_x, text_y), font_normal)

            strain = neighbor.get("strain", "unknown")
            if len(strain) > 28:
                strain = strain[:25] + "..."
            text_y = put_text(f"Strain: {strain}", (text_x, text_y), font_normal)

            environment = neighbor.get("environment", "unknown")
            text_y = put_text(f"Env: {environment}", (text_x, text_y), font_normal)

            angle = neighbor.get("angle", "unknown")
            text_y = put_text(f"Angle: {angle}", (text_x, text_y), font_normal)

            score = neighbor.get("score", 0.0)
            text_y = put_text(
                f"Score: {score:.4f}", (text_x, text_y), font_title, (0, 100, 0)
            )

        col += 1

    canvas_bgr = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, canvas_bgr)
    print(f"Visualization saved to: {output_path}")


def visualize_false_prediction(
    query_image_path: str,
    neighbors: List[Dict[str, Any]],
    segmented_image_dir: str,
    output_path: str,
    ground_truth_species: str,
    predicted_species: str,
    max_neighbors: int = 7,
    thumbnail_size: tuple[int, int] = (200, 200),
    query_metadata: Optional[Dict[str, Any]] = None,
    filter_same_strain: bool = True,
) -> None:
    """
    Create a visualization for false predictions showing query image and k nearest neighbors.
    """
    import os

    from PIL import Image, ImageDraw, ImageFont

    if filter_same_strain and query_metadata:
        query_strain = query_metadata.get("strain")
        if query_strain:
            neighbors = [n for n in neighbors if n.get("strain") != query_strain]

    neighbors = neighbors[:max_neighbors]

    query_img = cv2.imread(query_image_path)
    if query_img is None or query_img.size == 0:
        raise ValueError(f"Failed to read query image from {query_image_path}")

    query_img_resized = cv2.resize(query_img, thumbnail_size)

    num_images = len(neighbors) + 1
    text_height = 160
    border_width = 10
    padding = 20

    img_width = thumbnail_size[0]
    img_height = thumbnail_size[1]

    images_per_row = min(4, num_images)
    num_rows = (num_images + images_per_row - 1) // images_per_row

    title_height = 80
    canvas_width = images_per_row * (img_width + padding) + padding
    canvas_height = (
        title_height + num_rows * (img_height + text_height + padding) + padding
    )

    canvas_bgr = np.full(
        (canvas_height, canvas_width, 3), (255, 255, 255), dtype=np.uint8
    )

    canvas_pil = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas_pil)

    try:
        font_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 18
        )
        font_subtitle = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 14
        )
        font_normal = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 11
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 10
        )
    except Exception:
        try:
            font_title = ImageFont.truetype("arial.ttf", 18)
            font_subtitle = ImageFont.truetype("arial.ttf", 14)
            font_normal = ImageFont.truetype("arial.ttf", 11)
            font_small = ImageFont.truetype("arial.ttf", 10)
        except Exception:
            font_title = ImageFont.load_default()
            font_subtitle = ImageFont.load_default()
            font_normal = ImageFont.load_default()
            font_small = ImageFont.load_default()

    title_y = 10
    draw.text(
        (20, title_y), "FALSE PREDICTION ANALYSIS", font=font_title, fill=(255, 0, 0)
    )
    title_y += 30
    draw.text(
        (20, title_y),
        f"Ground Truth: {ground_truth_species}",
        font=font_subtitle,
        fill=(0, 100, 0),
    )
    title_y += 25
    draw.text(
        (20, title_y),
        f"Predicted: {predicted_species}",
        font=font_subtitle,
        fill=(255, 0, 0),
    )

    def put_text(
        text: str,
        position: tuple[int, int],
        font: Any,
        color: Optional[tuple[int, int, int]] = None,
    ) -> int:
        if color is None:
            color = (0, 0, 0)
        x, y = position
        draw.text((x, y), text, font=font, fill=color)
        bbox = draw.textbbox((x, y), text, font=font)
        return int(bbox[3] + 5)

    x_offset = padding
    y_offset = title_height + padding

    query_with_border = cv2.copyMakeBorder(
        query_img_resized,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 255),
    )

    query_pil = Image.fromarray(cv2.cvtColor(query_with_border, cv2.COLOR_BGR2RGB))
    canvas_pil.paste(query_pil, (x_offset, y_offset))

    y_end = y_offset + query_with_border.shape[0]
    text_y = y_end + 10
    text_x = x_offset + 5

    text_y = put_text("QUERY IMAGE", (text_x, text_y), font_subtitle, (200, 0, 0))

    if query_metadata:
        image_id = query_metadata.get("image_id", "unknown")
        if len(image_id) > 28:
            image_id = image_id[:25] + "..."
        text_y = put_text(f"ID: {image_id}", (text_x, text_y), font_small)

        specy = query_metadata.get("specy", ground_truth_species)
        if len(specy) > 28:
            specy = specy[:25] + "..."
        text_y = put_text(f"Species: {specy}", (text_x, text_y), font_normal)

        strain = query_metadata.get("strain", "unknown")
        if len(strain) > 28:
            strain = strain[:25] + "..."
        text_y = put_text(f"Strain: {strain}", (text_x, text_y), font_normal)

        environment = query_metadata.get("environment", "unknown")
        text_y = put_text(f"Env: {environment}", (text_x, text_y), font_normal)

        angle = query_metadata.get("angle", "unknown")
        text_y = put_text(f"Angle: {angle}", (text_x, text_y), font_normal)

    col = 1
    row = 0

    for idx, neighbor in enumerate(neighbors):
        if col >= images_per_row:
            col = 0
            row += 1

        x_offset = padding + col * (img_width + padding)
        y_offset = title_height + padding + row * (img_height + text_height + padding)

        neighbor_img_path = os.path.join(
            segmented_image_dir, f"{neighbor['image_id']}.jpg"
        )
        neighbor_img = cv2.imread(neighbor_img_path)

        if neighbor_img is None or neighbor_img.size == 0:
            neighbor_img = np.full(
                (thumbnail_size[1], thumbnail_size[0], 3),
                (180, 180, 180),
                dtype=np.uint8,
            )
        else:
            neighbor_img = cv2.resize(neighbor_img, thumbnail_size)

        neighbor_species = neighbor.get("specy", "unknown")
        if neighbor_species == ground_truth_species:
            border_color_bgr = (0, 255, 0)
            border_color_rgb = (0, 200, 0)
        else:
            border_color_bgr = (0, 0, 255)
            border_color_rgb = (200, 0, 0)

        neighbor_with_border = cv2.copyMakeBorder(
            neighbor_img,
            border_width,
            border_width,
            border_width,
            border_width,
            cv2.BORDER_CONSTANT,
            value=border_color_bgr,
        )

        y_end = y_offset + neighbor_with_border.shape[0]
        x_end = x_offset + neighbor_with_border.shape[1]

        if y_end <= canvas_height and x_end <= canvas_width:
            neighbor_pil = Image.fromarray(
                cv2.cvtColor(neighbor_with_border, cv2.COLOR_BGR2RGB)
            )
            canvas_pil.paste(neighbor_pil, (x_offset, y_offset))

            text_y = y_end + 10
            text_x = x_offset + 5

            text_y = put_text(
                f"#{idx + 1}", (text_x, text_y), font_subtitle, border_color_rgb
            )

            image_id = neighbor.get("image_id", "unknown")
            if len(image_id) > 28:
                image_id = image_id[:25] + "..."
            text_y = put_text(f"ID: {image_id}", (text_x, text_y), font_small)

            specy = neighbor.get("specy", "unknown")
            if len(specy) > 28:
                specy = specy[:25] + "..."
            text_y = put_text(
                f"Species: {specy}", (text_x, text_y), font_normal, border_color_rgb
            )

            strain = neighbor.get("strain", "unknown")
            if len(strain) > 28:
                strain = strain[:25] + "..."
            text_y = put_text(f"Strain: {strain}", (text_x, text_y), font_normal)

            environment = neighbor.get("environment", "unknown")
            text_y = put_text(f"Env: {environment}", (text_x, text_y), font_normal)

            angle = neighbor.get("angle", "unknown")
            text_y = put_text(f"Angle: {angle}", (text_x, text_y), font_normal)

            score = neighbor.get("score", 0.0)
            text_y = put_text(
                f"Score: {score:.4f}", (text_x, text_y), font_subtitle, (0, 100, 0)
            )

        col += 1

    canvas_bgr = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, canvas_bgr)
    print(f"Visualization saved to: {output_path}")

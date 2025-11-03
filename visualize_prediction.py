"""
Visualize prediction results from JSON files.
Displays the prediction outcome, query images, and top neighbors with detailed information.
"""
import os
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_prediction_result(json_path: str) -> Dict[str, Any]:
    """Load prediction result from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_image(image_path: str, target_size: tuple = (200, 200)) -> Optional[np.ndarray]:
    """
    Load and resize image.
    
    Args:
        image_path: Path to image file
        target_size: Target size (width, height)
        
    Returns:
        Loaded and resized image or None if not found
    """
    if not os.path.exists(image_path):
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def get_fonts() -> Tuple:
    """
    Load fonts for text rendering.
    
    Returns:
        Tuple of (font_title, font_normal, font_small)
    """
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
    
    return font_title, font_normal, font_small


def create_header_image(
    result: Dict[str, Any],
    width: int,
    height: int = 150
) -> Image.Image:
    """
    Create header image with prediction information.
    
    Args:
        result: Prediction result dictionary
        width: Image width
        height: Image height
        
    Returns:
        PIL Image with header information
    """
    # Extract metadata
    strain = result.get('strain', 'Unknown')
    ground_truth = result.get('ground_truth', 'Unknown')
    predicted_specy = result.get('predicted_specy', 'Unknown')
    confidence = result.get('predicted_confidence', 0.0)
    is_correct = result.get('correct', False)
    feature_extractor = result.get('feature_extractor', 'Unknown')
    k = result.get('k', 0)
    environment = result.get('environment', 'Unknown')
    
    # Create image
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Get fonts
    font_title, font_normal, font_small = get_fonts()
    
    # Helper function to put text
    def put_text(text: str, position: Tuple[int, int], font, color=(0, 0, 0)) -> int:
        """Put text and return next y position."""
        x, y = position
        draw.text((x, y), text, font=font, fill=color)
        bbox = draw.textbbox((x, y), text, font=font)
        return int(bbox[3] + 5)
    
    # Draw header content
    y = 10
    x_left = 10
    
    # Strain and settings
    y = put_text(f"Strain: {strain}", (x_left, y), font_title)
    y = put_text(f"Feature: {feature_extractor}, K: {k}, Env: {environment}", (x_left, y), font_small)
    y += 5
    
    # Result status
    result_text = "CORRECT ✓" if is_correct else "INCORRECT ✗"
    result_color = (0, 150, 0) if is_correct else (200, 0, 0)
    y = put_text(f"Result: {result_text}", (x_left, y), font_title, result_color)
    
    # Ground truth and prediction
    y = put_text(f"Ground Truth: {ground_truth}", (x_left, y), font_normal)
    y = put_text(f"Predicted: {predicted_specy}", (x_left, y), font_normal)
    y = put_text(f"Confidence: {confidence:.4f}", (x_left, y), font_normal)
    
    return img


def create_image_with_label(
    img_array: np.ndarray,
    text_lines: List[str],
    border_color: Tuple[int, int, int] = (128, 128, 128),
    border_width: int = 5,
    label_height: int = 100,
    is_query: bool = False
) -> Image.Image:
    """
    Create an image with border and text label below.
    
    Args:
        img_array: Image as numpy array (BGR)
        text_lines: Lines of text to display below image
        border_color: Border color (R, G, B)
        border_width: Border width in pixels
        label_height: Height of label area
        is_query: Whether this is a query image (changes formatting)
        
    Returns:
        PIL Image with border and label
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Add border
    img_with_border = Image.new(
        'RGB',
        (img_pil.width + 2 * border_width, img_pil.height + 2 * border_width),
        color=border_color
    )
    img_with_border.paste(img_pil, (border_width, border_width))
    
    # Create label area
    label_img = Image.new('RGB', (img_with_border.width, label_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(label_img)
    
    # Get fonts
    font_title, font_normal, font_small = get_fonts()
    
    # Draw text lines
    y = 5
    x = 5
    
    for idx, line in enumerate(text_lines):
        if idx == 0 and is_query:
            # Query title in bold green
            draw.text((x, y), line, font=font_title, fill=(0, 128, 0))
        elif idx == 0:
            # Rank number in bold
            draw.text((x, y), line, font=font_title, fill=(0, 0, 0))
        elif "Score:" in line or "Confidence:" in line:
            # Score in green
            draw.text((x, y), line, font=font_normal, fill=(0, 100, 0))
        elif "ID:" in line:
            # ID in smaller font
            draw.text((x, y), line, font=font_small, fill=(80, 80, 80))
        else:
            # Normal text
            draw.text((x, y), line, font=font_normal, fill=(0, 0, 0))
        
        bbox = draw.textbbox((x, y), line, font=font_normal)
        y = int(bbox[3] + 3)
    
    # Combine image and label
    final_img = Image.new(
        'RGB',
        (img_with_border.width, img_with_border.height + label_height),
        color=(255, 255, 255)
    )
    final_img.paste(img_with_border, (0, 0))
    final_img.paste(label_img, (0, img_with_border.height))
    
    return final_img


def visualize_prediction_result(
    result: Dict[str, Any],
    segmented_image_dir: str = "../Dataset/segmented_image",
    output_path: Optional[str] = None,
    max_query_images: int = 3,
    top_k_neighbors: int = 10,
    image_size: tuple = (200, 200)
) -> np.ndarray:
    """
    Create visualization for a prediction result.
    
    Args:
        result: Prediction result dictionary
        segmented_image_dir: Directory containing segmented images
        output_path: Optional path to save the visualization
        max_query_images: Maximum number of query images to display
        top_k_neighbors: Number of top neighbors to display per query image
        image_size: Size for each image (width, height)
        
    Returns:
        Visualization image as numpy array
    """
    ground_truth = result.get('ground_truth', 'Unknown')
    
    # Calculate canvas dimensions
    images_per_row = top_k_neighbors + 1  # query + neighbors
    image_with_label_height = image_size[1] + 100 + 10  # image + label + border
    
    # Create header
    header_width = images_per_row * (image_size[0] + 20)
    header = create_header_image(result, header_width)
    
    # Process query images and their neighbors
    raw_results = result.get('raw_results', [])
    row_images = []
    
    for idx, query_result in enumerate(raw_results[:max_query_images]):
        query_image_id = query_result.get('query_image_id', '')
        query_parent_id = query_result.get('query_parent_id', '')
        query_environment = query_result.get('query_environment', '')
        neighbors = query_result.get('neighbors', [])
        
        # Load query image
        query_image_path = os.path.join(segmented_image_dir, f"{query_image_id}.jpg")
        query_img = load_image(query_image_path, image_size)
        
        if query_img is None:
            # Create placeholder
            query_img = np.full((*image_size[::-1], 3), (200, 200, 200), dtype=np.uint8)
            cv2.putText(
                query_img, "Not Found", (10, image_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
        
        # Create query image with label
        query_label_lines = [
            f"QUERY #{idx + 1}",
            f"ID: {query_image_id[:20]}...",
            f"Env: {query_environment}",
        ]
        query_with_label = create_image_with_label(
            query_img,
            query_label_lines,
            border_color=(0, 128, 255),  # Blue border for query
            border_width=5,
            is_query=True
        )
        
        # Process neighbors
        neighbor_images_pil = [query_with_label]
        
        for rank, neighbor in enumerate(neighbors[:top_k_neighbors]):
            neighbor_image_id = neighbor.get('image_id', '')
            neighbor_strain = neighbor.get('strain', 'Unknown')
            neighbor_specy = neighbor.get('specy', 'Unknown')
            neighbor_env = neighbor.get('environment', 'Unknown')
            neighbor_score = neighbor.get('score', 0.0)
            
            # Load neighbor image
            neighbor_image_path = os.path.join(segmented_image_dir, f"{neighbor_image_id}.jpg")
            neighbor_img = load_image(neighbor_image_path, image_size)
            
            if neighbor_img is None:
                neighbor_img = np.full((*image_size[::-1], 3), (200, 200, 200), dtype=np.uint8)
                cv2.putText(
                    neighbor_img, "Not Found", (10, image_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                )
            
            # Determine border color
            border_color = (0, 200, 0) if neighbor_specy == ground_truth else (128, 128, 128)
            
            # Create neighbor image with label
            neighbor_label_lines = [
                f"#{rank + 1} Score: {neighbor_score:.3f}",
                f"ID: {neighbor_image_id[:20]}...",
                f"Strain: {neighbor_strain[:22]}",
                f"Specy: {neighbor_specy[:22]}",
                f"Env: {neighbor_env}",
            ]
            neighbor_with_label = create_image_with_label(
                neighbor_img,
                neighbor_label_lines,
                border_color=border_color,
                border_width=5
            )
            
            neighbor_images_pil.append(neighbor_with_label)
        
        # Pad with empty images if needed
        while len(neighbor_images_pil) < images_per_row:
            empty_img = Image.new(
                'RGB',
                (neighbor_images_pil[0].width, neighbor_images_pil[0].height),
                color=(240, 240, 240)
            )
            neighbor_images_pil.append(empty_img)
        
        # Combine images horizontally
        row_width = sum(img.width for img in neighbor_images_pil) + 10 * (len(neighbor_images_pil) - 1)
        row_height = max(img.height for img in neighbor_images_pil)
        row_img = Image.new('RGB', (row_width, row_height), color=(255, 255, 255))
        
        x_offset = 0
        for img in neighbor_images_pil:
            row_img.paste(img, (x_offset, 0))
            x_offset += img.width + 10
        
        row_images.append(row_img)
    
    # Combine all rows vertically
    if row_images:
        total_height = header.height + 10 + sum(img.height for img in row_images) + 10 * len(row_images)
        total_width = max(header.width, max(img.width for img in row_images))
        
        final_img = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
        
        # Paste header
        final_img.paste(header, (0, 0))
        
        # Paste rows
        y_offset = header.height + 10
        for row_img in row_images:
            final_img.paste(row_img, (0, y_offset))
            y_offset += row_img.height + 10
    else:
        final_img = header
    
    # Convert to numpy array
    final_array = cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        cv2.imwrite(output_path, final_array)
        print(f"Visualization saved to: {output_path}")
    
    return final_array


def main():
    parser = argparse.ArgumentParser(
        description="Visualize prediction results from JSON files"
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to prediction result JSON file"
    )
    parser.add_argument(
        "--segmented-dir",
        type=str,
        default="../Dataset/segmented_image",
        help="Directory containing segmented images (default: ../Dataset/segmented_image)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for visualization (default: auto-generated in results/visualizations/)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=3,
        help="Maximum number of query images to display (default: 3)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top neighbors to display (default: 10)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=200,
        help="Size for each image in pixels (default: 200)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the visualization in a window"
    )
    
    args = parser.parse_args()
    
    # Load prediction result
    print(f"Loading prediction result from: {args.json_path}")
    result = load_prediction_result(args.json_path)
    
    # Generate output path if not provided
    if args.output is None:
        json_filename = Path(args.json_path).stem
        output_dir = "./results/visualizations"
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{json_filename}_visualization.jpg")
    
    # Create visualization
    print(f"Creating visualization...")
    print(f"  Strain: {result.get('strain')}")
    print(f"  Ground truth: {result.get('ground_truth')}")
    print(f"  Predicted: {result.get('predicted_specy')}")
    print(f"  Correct: {result.get('correct')}")
    
    visualization = visualize_prediction_result(
        result=result,
        segmented_image_dir=args.segmented_dir,
        output_path=args.output,
        max_query_images=args.max_queries,
        top_k_neighbors=args.top_k,
        image_size=(args.image_size, args.image_size)
    )
    
    # Display if requested
    if args.show:
        cv2.imshow("Prediction Visualization", visualization)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"\n✓ Visualization complete!")


if __name__ == "__main__":
    main()

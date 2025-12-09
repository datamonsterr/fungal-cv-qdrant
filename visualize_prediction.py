"""
Visualization functions for species prediction results.
Displays K nearest neighbors per environment in a grid layout.
"""
import os
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


def visualize_prediction_by_environment(
    prediction_result: Dict[str, Any],
    segmented_image_dir: str,
    output_path: str,
    k: int = 7,
    thumbnail_size: Tuple[int, int] = (150, 150),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    border_width: int = 8
) -> None:
    """
    Create a visualization showing query images and their K nearest neighbors per environment.
    
    Layout:
    - Title section with metadata (ground truth, predicted species, strategy info)
    - Grid layout: Each row represents one environment
    - First column: Query image from that environment
    - Columns 2-(k+1): K nearest neighbors for that query
    - Green borders: Ground truth species matches
    - Red borders: False predictions
    
    Args:
        prediction_result: Result dictionary from predict_segment_group()
        segmented_image_dir: Directory containing segmented images
        output_path: Path to save the output visualization
        k: Number of nearest neighbors to display per query (default: 7)
        thumbnail_size: Size to resize each image to (width, height)
        text_color: RGB color for text
        bg_color: RGB color for background
        border_width: Width of colored border around images
    """
    # Extract metadata from prediction result
    ground_truth = prediction_result['ground_truth']
    predicted_specy = prediction_result['predicted_specy']
    is_correct = prediction_result['correct']
    confidence = prediction_result['predicted_confidence']
    feature_extractor = prediction_result['feature_extractor']
    aggregation_strategy = prediction_result['strategy'].upper()
    raw_results = prediction_result['raw_results']
    
    # Determine environment strategy
    env_filter = prediction_result.get('environment')
    if env_filter is None:
        env_strategy = "E1 (Same Environment)"
    elif env_filter.lower() == "all":
        env_strategy = "E2 (All Environments)"
    else:
        env_strategy = f"E3 ({env_filter})"
    
    # Sort raw_results by environment for consistent ordering
    raw_results_sorted = sorted(raw_results, key=lambda x: x.get('query_environment', ''))
    
    num_environments = len(raw_results_sorted)
    if num_environments == 0:
        raise ValueError("No raw results found in prediction_result")
    
    # Layout parameters
    img_width, img_height = thumbnail_size
    text_height = 120  # Height for text below each image
    header_height = 120  # Height for title section
    padding = 15
    row_spacing = 30  # Extra space between rows
    
    # Calculate canvas dimensions
    images_per_row = k + 1  # Query + K neighbors
    canvas_width = images_per_row * (img_width + padding) + padding
    canvas_height = (header_height + 
                     num_environments * (img_height + text_height + padding + row_spacing) + 
                     padding)
    
    # Create canvas
    canvas_bgr = np.full((canvas_height, canvas_width, 3), bg_color, dtype=np.uint8)
    canvas_pil = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas_pil)
    
    # Load fonts
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 16)
        font_subtitle = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 13)
        font_normal = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 11)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 9)
    except:
        try:
            font_title = ImageFont.truetype("arial.ttf", 16)
            font_subtitle = ImageFont.truetype("arial.ttf", 13)
            font_normal = ImageFont.truetype("arial.ttf", 11)
            font_small = ImageFont.truetype("arial.ttf", 9)
        except:
            font_title = ImageFont.load_default()
            font_subtitle = ImageFont.load_default()
            font_normal = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # Helper function to add text
    def put_text(text: str, position: Tuple[int, int], font: Any, 
                 color: Optional[Tuple[int, int, int]] = None) -> int:
        if color is None:
            color = text_color
        draw.text(position, text, font=font, fill=color)
        bbox = draw.textbbox(position, text, font=font)
        return bbox[3]  # Return bottom y coordinate
    
    # Draw header section
    header_y = 15
    
    # Title with color based on correctness
    title_color = (0, 150, 0) if is_correct else (200, 0, 0)
    status = "CORRECT PREDICTION" if is_correct else "FALSE PREDICTION"
    header_y = put_text(status, (padding, header_y), font_title, title_color) + 8
    
    # Prediction details
    header_y = put_text(f"Ground Truth: {ground_truth}", (padding, header_y), font_subtitle, (0, 100, 0)) + 5
    pred_color = (0, 100, 0) if is_correct else (200, 0, 0)
    header_y = put_text(f"Predicted: {predicted_specy} (Confidence: {confidence:.3f})", 
                       (padding, header_y), font_subtitle, pred_color) + 5
    
    # Strategy info
    strategy_text = f"Strategy: {env_strategy} | Aggregation: {aggregation_strategy} | Feature: {feature_extractor} | K={k}"
    put_text(strategy_text, (padding, header_y), font_normal, (80, 80, 80))
    
    # Draw separator line
    line_y = header_height - 10
    draw.line([(padding, line_y), (canvas_width - padding, line_y)], fill=(150, 150, 150), width=2)
    
    # Process each environment (each row)
    current_y = header_height + padding
    images_loaded = 0
    
    for env_idx, raw_result in enumerate(raw_results_sorted):
        query_image_id = raw_result['query_image_id']
        query_environment = raw_result.get('query_environment', 'unknown')
        neighbors = raw_result['neighbors'][:k]  # Limit to k neighbors
        
        # Load and prepare query image
        query_image_path = os.path.join(segmented_image_dir, f"{query_image_id}.jpg")
        query_img = cv2.imread(query_image_path)
        
        if query_img is None or query_img.size == 0:
            # Try without .jpg extension in case it's already included
            alt_path = os.path.join(segmented_image_dir, query_image_id)
            if os.path.exists(alt_path):
                query_img = cv2.imread(alt_path)
            
            if query_img is None or query_img.size == 0:
                print(f"Warning: Failed to load query image from:")
                print(f"  - {query_image_path}")
                print(f"  - {alt_path}")
                current_y += img_height + text_height + padding + row_spacing
                continue
        
        query_img_resized = cv2.resize(query_img, thumbnail_size)
        images_loaded += 1
        
        # Determine border color for query (always use prediction result color)
        border_color_bgr = (0, 255, 0) if is_correct else (0, 0, 255)  # Green if correct, red if false
        
        # Add border to query image
        query_with_border = cv2.copyMakeBorder(
            query_img_resized, border_width, border_width, border_width, border_width,
            cv2.BORDER_CONSTANT, value=border_color_bgr
        )
        
        # Place query image (first column)
        x_offset = padding
        query_pil = Image.fromarray(cv2.cvtColor(query_with_border, cv2.COLOR_BGR2RGB))
        canvas_pil.paste(query_pil, (x_offset, current_y))
        
        # Add text below query image
        text_y = current_y + query_with_border.shape[0] + 5
        text_x = x_offset + 5
        
        put_text(f"QUERY", (text_x, text_y), font_subtitle, (80, 80, 200))
        text_y += 18
        put_text(f"Env: {query_environment}", (text_x, text_y), font_small, (80, 80, 80))
        text_y += 14
        put_text(f"ID: {query_image_id[:16]}...", (text_x, text_y), font_small, (100, 100, 100))
        
        # Place neighbor images (columns 2 to k+1)
        for neighbor_idx, neighbor in enumerate(neighbors):
            col = neighbor_idx + 1  # Start from column 1 (0 is query)
            x_offset = padding + col * (img_width + padding)
            
            neighbor_image_id = neighbor['image_id']
            neighbor_specy = neighbor.get('specy', 'unknown')
            neighbor_strain = neighbor.get('strain', 'unknown')
            neighbor_env = neighbor.get('environment', 'unknown')
            neighbor_score = neighbor.get('score', 0.0)
            
            # Load neighbor image
            neighbor_image_path = os.path.join(segmented_image_dir, f"{neighbor_image_id}.jpg")
            neighbor_img = cv2.imread(neighbor_image_path)
            
            if neighbor_img is None or neighbor_img.size == 0:
                # Try without .jpg extension
                alt_path = os.path.join(segmented_image_dir, neighbor_image_id)
                if os.path.exists(alt_path):
                    neighbor_img = cv2.imread(alt_path)
                
                if neighbor_img is None or neighbor_img.size == 0:
                    print(f"Warning: Failed to load neighbor image from:")
                    print(f"  - {neighbor_image_path}")
                    print(f"  - {alt_path}")
                    continue
            
            neighbor_img_resized = cv2.resize(neighbor_img, thumbnail_size)
            images_loaded += 1
            
            # Determine border color (green if matches ground truth, red otherwise)
            if neighbor_specy == ground_truth:
                neighbor_border_color = (0, 255, 0)  # Green
                neighbor_text_color = (0, 150, 0)
            else:
                neighbor_border_color = (0, 0, 255)  # Red
                neighbor_text_color = (200, 0, 0)
            
            # Add border to neighbor image
            neighbor_with_border = cv2.copyMakeBorder(
                neighbor_img_resized, border_width, border_width, border_width, border_width,
                cv2.BORDER_CONSTANT, value=neighbor_border_color
            )
            
            # Place neighbor image
            neighbor_pil = Image.fromarray(cv2.cvtColor(neighbor_with_border, cv2.COLOR_BGR2RGB))
            canvas_pil.paste(neighbor_pil, (x_offset, current_y))
            
            # Add text below neighbor image
            text_y = current_y + neighbor_with_border.shape[0] + 5
            text_x = x_offset + 5
            
            put_text(f"#{neighbor_idx + 1}", (text_x, text_y), font_subtitle, (80, 80, 80))
            text_y += 18
            put_text(f"Species: {neighbor_specy[:18]}", (text_x, text_y), font_small, neighbor_text_color)
            text_y += 14
            put_text(f"Env: {neighbor_env}", (text_x, text_y), font_small, (80, 80, 80))
            text_y += 14
            put_text(f"Score: {neighbor_score:.3f}", (text_x, text_y), font_small, (80, 80, 80))
            text_y += 14
            put_text(f"ID: {neighbor_image_id[:12]}...", (text_x, text_y), font_small, (120, 120, 120))
        
        # Move to next row
        current_y += img_height + text_height + padding + row_spacing
    
    # Check if any images were loaded
    if images_loaded == 0:
        raise ValueError(f"No images could be loaded from {segmented_image_dir}. Please check the directory path and image IDs.")
    
    # Convert back to OpenCV and save
    canvas_bgr = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, canvas_bgr)
    print(f"Visualization saved to: {output_path} ({images_loaded} images loaded)")


def batch_visualize_predictions(
    prediction_results: List[Dict[str, Any]],
    segmented_image_dir: str,
    output_dir: str,
    k: int = 7,
    filter_correct: Optional[bool] = None,
    max_visualizations: Optional[int] = None
) -> List[str]:
    """
    Create visualizations for a batch of prediction results.
    
    Args:
        prediction_results: List of prediction result dictionaries
        segmented_image_dir: Directory containing segmented images
        output_dir: Directory to save visualizations
        k: Number of nearest neighbors to display
        filter_correct: If True, only visualize correct predictions; if False, only false predictions;
                       if None, visualize all
        max_visualizations: Maximum number of visualizations to create
        
    Returns:
        List of paths to created visualization files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter results if requested
    if filter_correct is not None:
        filtered_results = [r for r in prediction_results if r['correct'] == filter_correct]
    else:
        filtered_results = prediction_results
    
    # Limit number of visualizations
    if max_visualizations is not None:
        filtered_results = filtered_results[:max_visualizations]
    
    output_paths = []
    
    print(f"\nCreating {len(filtered_results)} visualizations...")
    for idx, result in enumerate(filtered_results, 1):
        strain = result['strain']
        status = "correct" if result['correct'] else "false"
        
        # Create filename
        filename = f"{idx:03d}_{strain.replace(' ', '_')}_{status}.jpg"
        output_path = os.path.join(output_dir, filename)
        
        try:
            visualize_prediction_by_environment(
                prediction_result=result,
                segmented_image_dir=segmented_image_dir,
                output_path=output_path,
                k=k
            )
            output_paths.append(output_path)
            print(f"  [{idx}/{len(filtered_results)}] {filename}")
        except Exception as e:
            print(f"  [{idx}/{len(filtered_results)}] Failed to create {filename}: {e}")
    
    return output_paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize species prediction results")
    parser.add_argument("--result-file", required=True, help="Path to prediction results JSON file")
    parser.add_argument("--segmented-dir", required=True, help="Directory containing segmented images")
    parser.add_argument("--output-dir", default="./visualizations", help="Output directory for visualizations")
    parser.add_argument("--k", type=int, default=7, help="Number of neighbors to display")
    parser.add_argument("--filter-correct", action="store_true", help="Only visualize correct predictions")
    parser.add_argument("--filter-false", action="store_true", help="Only visualize false predictions")
    parser.add_argument("--max-viz", type=int, help="Maximum number of visualizations to create")
    
    args = parser.parse_args()
    
    # Load results
    import json
    with open(args.result_file, 'r') as f:
        results = json.load(f)
    
    # Determine filter
    filter_correct = None
    if args.filter_correct:
        filter_correct = True
    elif args.filter_false:
        filter_correct = False
    
    # Create visualizations
    output_paths = batch_visualize_predictions(
        prediction_results=results,
        segmented_image_dir=args.segmented_dir,
        output_dir=args.output_dir,
        k=args.k,
        filter_correct=filter_correct,
        max_visualizations=args.max_viz
    )
    
    print(f"\n✓ Created {len(output_paths)} visualizations in {args.output_dir}/")

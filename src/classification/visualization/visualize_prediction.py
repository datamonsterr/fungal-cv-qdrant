import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import RESULTS_DIR, SEGMENTED_IMAGE_DIR


def generate_distinct_color(
    species_name: str, ground_truth: str
) -> Tuple[int, int, int]:
    """
    Generate a distinct color for a species based on its name.
    Ground truth species always gets green.
    Other species get distinct colors from a predefined palette.
    """
    if species_name == ground_truth:
        return (0, 255, 0)  # Green for ground truth

    # Predefined color palette (avoiding green for ground truth)
    COLOR_PALETTE = [
        (255, 0, 0),  # Red
        (0, 0, 255),  # Blue
        (255, 165, 0),  # Orange
        (148, 0, 211),  # Dark Violet
        (255, 20, 147),  # Deep Pink
        (0, 191, 255),  # Deep Sky Blue
        (255, 215, 0),  # Gold
        (220, 20, 60),  # Crimson
        (138, 43, 226),  # Blue Violet
        (255, 105, 180),  # Hot Pink
        (70, 130, 180),  # Steel Blue
        (255, 69, 0),  # Orange Red
        (186, 85, 211),  # Medium Orchid
        (30, 144, 255),  # Dodger Blue
        (255, 140, 0),  # Dark Orange
    ]

    # Use hash to consistently assign the same color to the same species
    hash_val = abs(hash(species_name))
    color_index = hash_val % len(COLOR_PALETTE)

    return COLOR_PALETTE[color_index]


def visualize_prediction_by_environment(
    prediction_result: Dict[str, Any],
    segmented_image_dir: str,
    output_path: str,
    k: int = 7,
    thumbnail_size: Tuple[int, int] = (150, 150),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    border_width: int = 8,
) -> None:
    """
    Create a visualization showing query images and their K nearest neighbors per environment.
    """
    # Extract metadata
    ground_truth = prediction_result["ground_truth"]
    predicted_specy = prediction_result["predicted_specy"]
    is_correct = prediction_result["correct"]
    confidence = prediction_result["predicted_confidence"]
    feature_extractor = prediction_result["feature_extractor"]
    aggregation_strategy = prediction_result["strategy"].upper()
    raw_results = prediction_result["raw_results"]
    aggregated_results = prediction_result.get("aggregated_results", [])

    # Create a species-to-rank mapping for quick lookup
    species_rank_map = {}
    for rank, agg_result in enumerate(aggregated_results, start=1):
        species_rank_map[agg_result["specy"]] = rank

    # Sort raw_results by environment
    raw_results_sorted = sorted(
        raw_results, key=lambda x: x.get("query_environment", "")
    )

    num_environments = len(raw_results_sorted)
    if num_environments == 0:
        print("No raw results to visualize.")
        return

    # Layout parameters
    img_width, img_height = thumbnail_size
    text_height = 90
    header_height = 250  # Increased for ranking legend
    padding = 15
    row_spacing = 100

    images_per_row = k + 1
    canvas_width = images_per_row * (img_width + padding) + padding
    canvas_height = header_height + num_environments * (
        img_height + text_height + row_spacing
    )

    # Create canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    # Load fonts (try to load a nice font, fallback to default)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        text_font = ImageFont.truetype("DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except IOError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # --- Header Section ---
    title_text = f"Strain: {prediction_result['strain']} | Ground Truth: {ground_truth}"
    pred_text = (
        f"Predicted: {predicted_specy} ({confidence:.2f}) | Correct: {is_correct}"
    )
    info_text = f"Extractor: {feature_extractor} | Strategy: {aggregation_strategy}"

    draw.text((padding, 20), title_text, fill=text_color, font=title_font)
    draw.text(
        (padding, 60),
        pred_text,
        fill=(0, 128, 0) if is_correct else (255, 0, 0),
        font=title_font,
    )
    draw.text((padding, 100), info_text, fill=text_color, font=text_font)

    # Draw Ranking Legend
    draw.text((padding, 130), "Top Species Ranking:", fill=text_color, font=text_font)
    legend_x = padding
    legend_y = 150
    for i, res in enumerate(aggregated_results[:5]):  # Show top 5
        specy = res["specy"]
        score = res["score"]
        color = generate_distinct_color(specy, ground_truth)
        legend_text = f"{i+1}. {specy} ({score:.2f})"

        # Draw colored box
        draw.rectangle([legend_x, legend_y, legend_x + 15, legend_y + 15], fill=color)
        draw.text(
            (legend_x + 20, legend_y), legend_text, fill=text_color, font=small_font
        )

        legend_x += 250
        if legend_x > canvas_width - 200:
            legend_x = padding
            legend_y += 20

    # --- Grid Section ---
    current_y = header_height

    for row_idx, result in enumerate(raw_results_sorted):
        query_id = result["query_image_id"]
        environment = result.get("query_environment", "unknown")
        neighbors = result["neighbors"]

        # Draw Environment Label
        draw.text(
            (padding, current_y - 30),
            f"Environment: {environment}",
            fill=text_color,
            font=title_font,
        )

        # 1. Draw Query Image
        query_path = os.path.join(segmented_image_dir, f"{query_id}.jpg")
        if os.path.exists(query_path):
            try:
                img = Image.open(query_path)
                img = img.resize(thumbnail_size)
                canvas.paste(img, (padding, current_y))

                # Draw border (Green if correct prediction for this query? Or just black for query)
                # Let's use black for query
                draw.rectangle(
                    [padding, current_y, padding + img_width, current_y + img_height],
                    outline=(0, 0, 0),
                    width=4,
                )

                draw.text(
                    (padding, current_y + img_height + 5),
                    "Query",
                    fill=text_color,
                    font=text_font,
                )
                draw.text(
                    (padding, current_y + img_height + 25),
                    f"ID: {query_id}",
                    fill=text_color,
                    font=small_font,
                )

            except Exception as e:
                print(f"Error loading query image {query_path}: {e}")

        # 2. Draw Neighbors
        for i, neighbor in enumerate(neighbors):
            if i >= k:
                break

            n_id = neighbor.get("image_id") or neighbor.get("id")
            n_specy = neighbor.get("specy", "unknown")
            n_score = neighbor.get("score", 0.0)
            n_strain = neighbor.get("strain", "unknown")

            x_pos = padding + (i + 1) * (img_width + padding)

            n_path = os.path.join(segmented_image_dir, f"{n_id}.jpg")
            if os.path.exists(n_path):
                try:
                    img = Image.open(n_path)
                    img = img.resize(thumbnail_size)
                    canvas.paste(img, (x_pos, current_y))

                    # Border color based on species match
                    border_color = generate_distinct_color(n_specy, ground_truth)

                    draw.rectangle(
                        [x_pos, current_y, x_pos + img_width, current_y + img_height],
                        outline=border_color,
                        width=border_width,
                    )

                    # Text info
                    draw.text(
                        (x_pos, current_y + img_height + 5),
                        f"#{i+1} {n_specy}",
                        fill=text_color,
                        font=text_font,
                    )
                    draw.text(
                        (x_pos, current_y + img_height + 25),
                        f"Score: {n_score:.4f}",
                        fill=text_color,
                        font=small_font,
                    )
                    draw.text(
                        (x_pos, current_y + img_height + 40),
                        f"Strain: {n_strain}",
                        fill=text_color,
                        font=small_font,
                    )

                except Exception as e:
                    print(f"Error loading neighbor image {n_path}: {e}")

        current_y += img_height + text_height + row_spacing

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    print(f"Saved visualization to {output_path}")


def batch_visualize_predictions(
    prediction_results: List[Dict[str, Any]],
    segmented_image_dir: str,
    output_dir: str,
    k: int = 7,
    filter_correct: Optional[bool] = None,
    max_visualizations: Optional[int] = None,
) -> List[str]:
    """
    Batch visualize predictions.
    """
    saved_paths = []
    count = 0

    for result in prediction_results:
        if filter_correct is not None:
            if result["correct"] != filter_correct:
                continue

        if max_visualizations and count >= max_visualizations:
            break

        strain = result["strain"]
        test_set_index = result.get("test_set_index", "")

        if test_set_index != "":
            filename = f"pred_{strain}_set{test_set_index}.jpg"
        else:
            filename = f"pred_{strain}.jpg"

        output_path = os.path.join(output_dir, filename)

        visualize_prediction_by_environment(
            prediction_result=result,
            segmented_image_dir=segmented_image_dir,
            output_path=output_path,
            k=k,
        )

        saved_paths.append(output_path)
        count += 1

    return saved_paths

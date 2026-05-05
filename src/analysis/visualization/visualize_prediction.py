import os
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


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


def _get_thumbnail_size(k: int) -> Tuple[int, int]:
    if k >= 11:
        return (96, 96)
    if k >= 9:
        return (108, 108)
    if k >= 7:
        return (124, 124)
    return (150, 150)


def _text_size(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
) -> Tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: Tuple[int, int],
    max_width: int,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
    line_spacing: int = 4,
) -> int:
    x_pos, y_pos = position
    words = text.split()
    lines: List[str] = []
    current_line = ""

    for word in words:
        candidate = word if not current_line else f"{current_line} {word}"
        candidate_width, _ = _text_size(draw, candidate, font)
        if candidate_width <= max_width or not current_line:
            current_line = candidate
            continue

        lines.append(current_line)
        current_line = word

    if current_line:
        lines.append(current_line)

    line_height = _text_size(draw, "Ag", font)[1]
    for line in lines:
        draw.text((x_pos, y_pos), line, fill=fill, font=font)
        y_pos += line_height + line_spacing

    return y_pos


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    aggregated_results: List[Dict[str, Any]],
    ground_truth: str,
    start_x: int,
    start_y: int,
    canvas_width: int,
    text_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    text_color: Tuple[int, int, int],
    padding: int,
) -> int:
    draw.text(
        (start_x, start_y),
        "Top Species Ranking:",
        fill=text_color,
        font=text_font,
    )
    y_pos = start_y + _text_size(draw, "Ag", text_font)[1] + 8
    box_size = 15
    item_gap_x = 18
    item_gap_y = 10
    max_item_width = max(220, (canvas_width - 2 * padding) // 2)
    item_x = start_x
    item_y = y_pos
    max_y = item_y

    for i, res in enumerate(aggregated_results[:5]):
        specy = res["specy"]
        score = res["score"]
        color = generate_distinct_color(specy, ground_truth)
        legend_text = f"{i + 1}. {specy} ({score:.2f})"
        text_w, text_h = _text_size(draw, legend_text, small_font)
        item_width = min(max_item_width, text_w + 30)

        if item_x + item_width > canvas_width - padding:
            item_x = start_x
            item_y = max_y + item_gap_y

        draw.rectangle(
            [item_x, item_y + 2, item_x + box_size, item_y + 2 + box_size],
            fill=color,
        )
        draw.text(
            (item_x + box_size + 8, item_y),
            legend_text,
            fill=text_color,
            font=small_font,
        )

        item_x += item_width + item_gap_x
        max_y = max(max_y, item_y + text_h)

    return max_y + 16


def _draw_image_card(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image_path: str,
    position: Tuple[int, int],
    thumbnail_size: Tuple[int, int],
    border_color: Tuple[int, int, int],
    border_width: int,
    lines: List[Tuple[str, ImageFont.ImageFont]],
    text_color: Tuple[int, int, int],
    line_spacing: int = 4,
) -> None:
    x_pos, y_pos = position
    img_width, img_height = thumbnail_size

    if os.path.exists(image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(thumbnail_size)
            canvas.paste(img, (x_pos, y_pos))
        except Exception as exc:
            print(f"Error loading image {image_path}: {exc}")
    else:
        draw.rectangle(
            [x_pos, y_pos, x_pos + img_width, y_pos + img_height],
            outline=(160, 160, 160),
            width=2,
        )
        draw.text((x_pos + 8, y_pos + 8), "Missing image", fill=text_color)

    draw.rectangle(
        [x_pos, y_pos, x_pos + img_width, y_pos + img_height],
        outline=border_color,
        width=border_width,
    )

    text_y = y_pos + img_height + 6
    for text, font in lines:
        draw.text((x_pos, text_y), text, fill=text_color, font=font)
        text_y += _text_size(draw, text, font)[1] + line_spacing


def _resolve_image_path(
    item: Dict[str, Any],
    default_dir: str,
    id_key: str,
) -> str:
    explicit_path = item.get("image_path") or item.get("query_image_path")
    if explicit_path:
        return explicit_path

    image_id = item.get(id_key) or item.get("id") or ""
    return os.path.join(default_dir, f"{image_id}.jpg")


def visualize_prediction_by_environment(
    prediction_result: Dict[str, Any],
    segmented_image_dir: str,
    output_path: str,
    k: int = 7,
    thumbnail_size: Optional[Tuple[int, int]] = None,
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

    # Sort raw_results by environment
    raw_results_sorted = sorted(
        raw_results, key=lambda x: x.get("query_environment", "")
    )

    num_environments = len(raw_results_sorted)
    if num_environments == 0:
        print("No raw results to visualize.")
        return

    resolved_thumbnail_size = thumbnail_size or _get_thumbnail_size(k)
    img_width, img_height = resolved_thumbnail_size
    padding = 15
    card_gap = 15
    row_spacing = 24
    env_label_gap = 34
    text_height = 64
    target_canvas_width = 1600
    card_width = img_width + padding
    columns = max(2, min(k + 1, max(2, (target_canvas_width - padding) // card_width)))
    canvas_width = padding + columns * card_width

    # Load fonts (try to load a nice font, fallback to default)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        text_font = ImageFont.truetype("DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except IOError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    title_text = f"Strain: {prediction_result['strain']} | Ground Truth: {ground_truth}"
    pred_text = (
        f"Predicted: {predicted_specy} ({confidence:.2f}) | Correct: {is_correct}"
    )
    info_text = f"Extractor: {feature_extractor} | Strategy: {aggregation_strategy}"

    measure_canvas = Image.new("RGB", (canvas_width, 200), bg_color)
    measure_draw = ImageDraw.Draw(measure_canvas)

    header_y = 20
    header_y = _draw_wrapped_text(
        measure_draw,
        title_text,
        (padding, header_y),
        canvas_width - 2 * padding,
        title_font,
        text_color,
    )
    header_y = _draw_wrapped_text(
        measure_draw,
        pred_text,
        (padding, header_y + 4),
        canvas_width - 2 * padding,
        title_font,
        (0, 128, 0) if is_correct else (255, 0, 0),
    )
    header_y = _draw_wrapped_text(
        measure_draw,
        info_text,
        (padding, header_y + 4),
        canvas_width - 2 * padding,
        text_font,
        text_color,
    )
    header_bottom = _draw_legend(
        measure_draw,
        aggregated_results,
        ground_truth,
        padding,
        header_y + 8,
        canvas_width,
        text_font,
        small_font,
        text_color,
        padding,
    )

    cards_per_environment = k + 1
    env_rows = ceil(cards_per_environment / columns)
    env_block_height = (
        env_label_gap + env_rows * (img_height + text_height + card_gap) + row_spacing
    )
    canvas_height = header_bottom + 16 + num_environments * env_block_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    header_y = 20
    header_y = _draw_wrapped_text(
        draw,
        title_text,
        (padding, header_y),
        canvas_width - 2 * padding,
        title_font,
        text_color,
    )
    header_y = _draw_wrapped_text(
        draw,
        pred_text,
        (padding, header_y + 4),
        canvas_width - 2 * padding,
        title_font,
        (0, 128, 0) if is_correct else (255, 0, 0),
    )
    header_y = _draw_wrapped_text(
        draw,
        info_text,
        (padding, header_y + 4),
        canvas_width - 2 * padding,
        text_font,
        text_color,
    )
    current_y = (
        _draw_legend(
            draw,
            aggregated_results,
            ground_truth,
            padding,
            header_y + 8,
            canvas_width,
            text_font,
            small_font,
            text_color,
            padding,
        )
        + 16
    )

    for result in raw_results_sorted:
        query_id = result["query_image_id"]
        environment = result.get("query_environment", "unknown")
        neighbors = result["neighbors"]

        draw.text(
            (padding, current_y - 30),
            f"Environment: {environment}",
            fill=text_color,
            font=title_font,
        )

        cards: List[Dict[str, Any]] = [
            {
                "image_path": _resolve_image_path(
                    result,
                    segmented_image_dir,
                    "query_image_id",
                ),
                "border_color": (0, 0, 0),
                "lines": [("Query", text_font), (f"ID: {query_id}", small_font)],
            }
        ]

        for i, neighbor in enumerate(neighbors[:k], start=1):
            n_specy = neighbor.get("specy", "unknown")
            n_score = neighbor.get("score", 0.0)
            n_strain = neighbor.get("strain", "unknown")
            cards.append(
                {
                    "image_path": _resolve_image_path(
                        neighbor,
                        segmented_image_dir,
                        "image_id",
                    ),
                    "border_color": generate_distinct_color(n_specy, ground_truth),
                    "lines": [
                        (f"#{i} {n_specy}", text_font),
                        (f"Score: {n_score:.4f}", small_font),
                        (f"Strain: {n_strain}", small_font),
                    ],
                }
            )

        grid_y = current_y
        for card_index, card in enumerate(cards):
            row_index = card_index // columns
            column_index = card_index % columns
            x_pos = padding + column_index * card_width
            y_pos = grid_y + row_index * (img_height + text_height + card_gap)

            _draw_image_card(
                canvas,
                draw,
                card["image_path"],
                (x_pos, y_pos),
                resolved_thumbnail_size,
                card["border_color"],
                border_width if card_index > 0 else 4,
                card["lines"],
                text_color,
            )

        current_y += env_block_height

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

from typing import Any, Literal, overload

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

type BBox = dict[str, int]
type Image = Any

CLUSTER_COLOURS = np.array(
    [
        (0, 80, 255),
        (255, 80, 80),
        (80, 220, 80),
        (220, 80, 220),
        (80, 220, 220),
    ],
    dtype=np.uint8,
)


def _default_plate_mask(shape: tuple[int, int]) -> NDArray[np.uint8]:
    height, width = shape
    radius = int(round(min(height, width) * 0.45))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.circle(mask, (width // 2, height // 2), radius, 255, -1)
    return mask


def _local_k2_shrink(
    bbox: BBox,
    hsv_image: Image,
    original_bgr: Image,
) -> BBox | None:
    """Shrink a bloated bounding box using local K=2 clustering on the Value channel.

    This strips light halos from bright small colonies without affecting overgrown
    textured colonies (whose bright pixels are distributed everywhere, so the box
    stays large after K=2).
    """
    x1, y1, x2, y2 = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
    # Crop the ROI from the unblurred HSV image
    roi_hsv = hsv_image[y1:y2, x1:x2]
    if roi_hsv.size == 0:
        return None

    v_channel = roi_hsv[:, :, 2].astype(np.float32).reshape(-1, 1)

    # Run local K=2 to separate bright core from dimmer halo
    try:
        labels = KMeans(n_clusters=2, random_state=0, n_init=5).fit_predict(v_channel)
    except Exception:
        return None

    # Identify which cluster is brighter
    cluster0_mean = float(v_channel[labels == 0].mean())
    cluster1_mean = float(v_channel[labels == 1].mean())
    bright_label = 0 if cluster0_mean >= cluster1_mean else 1

    # Build a mask of just the bright pixels
    bright_mask = (labels == bright_label).reshape(roi_hsv.shape[:2]).astype(
        np.uint8
    ) * 255

    # Find contours on the bright mask and take the largest one
    contours, _ = cv.findContours(bright_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_contour = max(contours, key=cv.contourArea)
    cx, cy, cw, ch = cv.boundingRect(best_contour)

    # Pad slightly to keep the box from being overly tight
    pad = max(2, int(max(x2 - x1, y2 - y1) * 0.02))
    rx1 = max(0, x1 + cx - pad)
    ry1 = max(0, y1 + cy - pad)
    rx2 = min(original_bgr.shape[1], x1 + cx + cw + pad)
    ry2 = min(original_bgr.shape[0], y1 + cy + ch + pad)

    return {"xmin": rx1, "ymin": ry1, "xmax": rx2, "ymax": ry2}


def _odd_kernel(value: float, minimum: int, maximum: int) -> int:
    kernel = int(round(value))
    kernel = max(minimum, min(kernel, maximum))
    if kernel % 2 == 0:
        kernel += 1
    return min(kernel, maximum if maximum % 2 == 1 else maximum - 1)


def _choose_foreground_label(
    labels: NDArray[np.int32],
    mask: NDArray[np.bool_],
    hsv_image: Image,
) -> int:
    inside_labels = labels[mask]
    if inside_labels.size == 0:
        return 0

    outside_labels = labels[~mask]
    background_label = -1
    if outside_labels.size > 0:
        background_label = int(np.bincount(outside_labels).argmax())

    candidate_labels = [
        int(label)
        for label in np.unique(inside_labels)
        if int(label) != background_label
    ]
    if not candidate_labels:
        candidate_labels = [int(label) for label in np.unique(inside_labels)]

    inside_total = len(inside_labels)
    masked_hsv = hsv_image[mask]
    preferred: list[tuple[float, int]] = []
    fallback: list[tuple[float, int]] = []

    for label in candidate_labels:
        cluster_pixels = masked_hsv[inside_labels == label]
        if len(cluster_pixels) == 0:
            continue
        area_fraction = len(cluster_pixels) / inside_total
        value_score = float(np.percentile(cluster_pixels[:, 2], 95))
        saturation_score = float(np.percentile(cluster_pixels[:, 1], 75))
        score = value_score + 0.35 * saturation_score
        item = (score, label)
        if 0.005 <= area_fraction <= 0.25:
            preferred.append(item)
        fallback.append(item)

    if preferred:
        return max(preferred)[1]
    if fallback:
        return max(fallback)[1]
    return candidate_labels[0]


def _erode_to_core(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    size = max(mask.shape)
    close_size = _odd_kernel(size * 0.015, minimum=3, maximum=11)
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_size, close_size))
    closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_kernel, iterations=1)

    erode_size = _odd_kernel(size * 0.035, minimum=5, maximum=25)
    erode_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erode_size, erode_size))
    eroded = cv.morphologyEx(closed, cv.MORPH_ERODE, erode_kernel, iterations=1)

    dilate_size = _odd_kernel(size * 0.025, minimum=5, maximum=21)
    dilate_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (dilate_size, dilate_size)
    )
    restored = cv.morphologyEx(eroded, cv.MORPH_DILATE, dilate_kernel, iterations=1)

    return np.asarray(restored, dtype=np.uint8)


def calculate_bbox_quality(bboxes: list[BBox], mask: NDArray[np.uint8]) -> float:
    if len(bboxes) == 0:
        return 0.0

    areas = []
    fits = []

    for bbox in bboxes:
        width = max(1, bbox["xmax"] - bbox["xmin"])
        height = max(1, bbox["ymax"] - bbox["ymin"])
        area = width * height
        areas.append(area)

        box_mask = mask[bbox["ymin"] : bbox["ymax"], bbox["xmin"] : bbox["xmax"]]
        foreground_area = np.count_nonzero(box_mask)
        fit = foreground_area / area
        fits.append(fit)

    if len(areas) < 2:
        cv_penalty = 1.0
    else:
        mean_area = np.mean(areas)
        if mean_area == 0:
            cv_val = 1.0
        else:
            cv_val = float(np.std(areas)) / float(mean_area)
        cv_penalty = 1.0 - min(cv_val, 1.0)

    mean_fit = float(np.mean(fits))
    fit_score = 1.0 - min(abs(mean_fit - 0.785), 1.0)

    return float(cv_penalty * fit_score)


def _cluster_visual(
    labels: NDArray[np.int32],
    mask: NDArray[np.bool_],
) -> Image:
    visual = CLUSTER_COLOURS[labels]
    output = np.zeros_like(visual)
    output[mask] = visual[mask]
    return output


def _location_visual(
    points: NDArray[np.int32],
    labels: NDArray[np.int32],
    shape: tuple[int, int],
) -> Image:
    visual = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for point, label in zip(points, labels, strict=False):
        visual[point[0], point[1]] = CLUSTER_COLOURS[int(label) % len(CLUSTER_COLOURS)]
    return visual


def _sort_bboxes(bboxes: list[BBox]) -> list[BBox]:
    return sorted(bboxes, key=lambda bbox: (bbox["ymin"], bbox["xmin"]))


def get_bbox(labels: NDArray[np.int32], mat: NDArray[np.int32]) -> list[BBox]:
    cluster_count = int(labels.max()) + 1 if labels.size else 0
    clusters: list[list[NDArray[np.int32]]] = [[] for _ in range(cluster_count)]
    for i, label in enumerate(labels):
        clusters[int(label)].append(mat[i])

    bounding_boxes: list[BBox] = []
    for cluster in clusters:
        if not cluster:
            continue
        cluster_points = np.array(cluster, dtype=np.int32)
        y_min = int(np.min(cluster_points[:, 0]))
        y_max = int(np.max(cluster_points[:, 0]))
        x_min = int(np.min(cluster_points[:, 1]))
        x_max = int(np.max(cluster_points[:, 1]))
        bounding_boxes.append(
            {
                "xmin": x_min,
                "ymin": y_min,
                "xmax": x_max,
                "ymax": y_max,
            }
        )
    return bounding_boxes


def draw_bbox(img: Image, bboxes: list[BBox]) -> Image:
    img_with_boxes = img.copy()
    thickness = max(2, int(max(img.shape[:2]) * 0.005))
    for i, bbox in enumerate(bboxes):
        cv.rectangle(
            img_with_boxes,
            (bbox["xmin"], bbox["ymin"]),
            (bbox["xmax"], bbox["ymax"]),
            tuple(int(value) for value in CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]),
            thickness,
        )
    return np.asarray(img_with_boxes, dtype=np.uint8)


def refine_bboxes(
    bboxes: list[BBox],
    mask: NDArray[np.uint8],
    hsv_image: Image | None = None,
    original_bgr: Image | None = None,
) -> list[BBox]:
    refined = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        size = max(w, h)

        # 1. Make outside boundingbox black pixel and apply the white foreground mask
        roi = mask[y1:y2, x1:x2].copy()

        # 2. Parameters depend on size of the bounding box
        k_size = _odd_kernel(size * 0.05, minimum=3, maximum=45)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))

        # Disconnect plate edges by eroding
        eroded = cv.morphologyEx(roi, cv.MORPH_ERODE, kernel)

        # Contour detection
        contours, _ = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback to no erosion
            contours, _ = cv.findContours(roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            refined.append(bbox)
            continue

        best_contour = None
        max_score = -1.0

        for c in contours:
            area = cv.contourArea(c)
            cx, cy, cw, ch = cv.boundingRect(c)
            bbox_area = max(1, cw * ch)

            # 3. Final result should be: area ratio of white pixels ~ pi/4 (circle fills bounding box well)
            fill_ratio = area / float(bbox_area)
            fit_score = 1.0 - min(abs(fill_ratio - 0.785), 1.0)
            score = area * fit_score

            if score > max_score:
                max_score = score
                best_contour = c

        if best_contour is not None:
            best_cx, best_cy, best_cw, best_ch = cv.boundingRect(best_contour)
            best_fill_ratio = cv.contourArea(best_contour) / float(
                max(1, best_cw * best_ch)
            )
            fit_score = 1.0 - min(abs(best_fill_ratio - 0.785), 1.0)

            # Poor circular fit → likely a plate-edge artifact or bright-halo case.
            # Try local K=2 shrink on the unblurred HSV to strip the halo safely.
            if fit_score < 0.3 and hsv_image is not None and original_bgr is not None:
                shrunk = _local_k2_shrink(bbox, hsv_image, original_bgr)
                if shrunk is not None:
                    refined.append(shrunk)
                    continue

            # Standard erode+contour path (good for normal colonies)
            pad = (k_size // 2) + int(size * 0.02)
            rx1 = max(0, x1 + best_cx - pad)
            ry1 = max(0, y1 + best_cy - pad)
            rx2 = min(mask.shape[1], x1 + best_cx + best_cw + pad)
            ry2 = min(mask.shape[0], y1 + best_cy + best_ch + pad)
            refined.append({"xmin": rx1, "ymin": ry1, "xmax": rx2, "ymax": ry2})
        else:
            refined.append(bbox)

    return _sort_bboxes(refined)


@overload
def segment_kmeans_image(
    img: Image,
    *,
    plate_mask: NDArray[np.uint8] | None = None,
    return_debug: Literal[False] = False,
) -> tuple[list[BBox], float]: ...


@overload
def segment_kmeans_image(
    img: Image,
    *,
    plate_mask: NDArray[np.uint8] | None = None,
    return_debug: Literal[True],
) -> tuple[list[BBox], float, dict[str, Image]]: ...


def segment_kmeans_image(
    img: Image,
    *,
    plate_mask: NDArray[np.uint8] | None = None,
    return_debug: bool = False,
) -> tuple[list[BBox], float] | tuple[list[BBox], float, dict[str, Image]]:
    if img.size == 0:
        empty: dict[str, Image] = {}
        return ([], 0.0, empty) if return_debug else ([], 0.0)

    # 1. Median blur to flatten texture and make neighbor colors uniform (reduces variance)
    median_k = _odd_kernel(max(img.shape[:2]) * 0.03, minimum=15, maximum=99)
    median_bgr = cv.medianBlur(img, median_k)

    # 2. Convert to HSV
    hsv = cv.cvtColor(median_bgr, cv.COLOR_BGR2HSV)

    # 3. Gaussian blur to smooth out the steps from the median filter (VERY blur)
    gauss_k = _odd_kernel(max(img.shape[:2]) * 0.08, minimum=31, maximum=255)
    blur = cv.GaussianBlur(hsv, (gauss_k, gauss_k), 0)

    # Unblurred HSV for local K=2 shrink step (preserves bright-core vs dimmer-halo distinction)
    hsv_raw = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    if plate_mask is None:
        plate_mask = _default_plate_mask(img.shape[:2])
    mask = plate_mask.astype(bool)

    flat = blur.reshape(-1, 3).astype(np.float32)
    flat[~mask.reshape(-1)] = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)

    # Reduce n_clusters from 5 to 3.
    # The 3 clusters will naturally become: (1) Artificial Background, (2) Agar/Plate, (3) Fungi
    # This forces the internal variations of the fungi to group into a single cluster.
    labels = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(flat)
    label_image = labels.reshape(img.shape[:2]).astype(np.int32)
    color_dimension = _cluster_visual(label_image, mask)

    foreground_label = _choose_foreground_label(label_image, mask, hsv)
    foreground_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    foreground_mask[(label_image == foreground_label) & mask] = 255
    cleaned_foreground = _erode_to_core(foreground_mask)
    if not np.any(cleaned_foreground):
        cleaned_foreground = foreground_mask

    points = np.argwhere(cleaned_foreground == 255).astype(np.int32)
    if len(points) == 0:
        debug_images = {
            "color_dimension": color_dimension,
            "foreground_mask": cv.cvtColor(cleaned_foreground, cv.COLOR_GRAY2BGR),
            "location_clusters": np.zeros_like(img),
            "bbox_image": img.copy(),
        }
        return ([], 0.0, debug_images) if return_debug else ([], 0.0)

    cluster_count = min(3, len(points))
    if cluster_count == 1:
        location_labels = np.zeros(len(points), dtype=np.int32)
    else:
        location_labels = KMeans(
            n_clusters=cluster_count,
            random_state=0,
            n_init=10,
        ).fit_predict(points.astype(np.float32))

    bboxes = _sort_bboxes(get_bbox(location_labels.astype(np.int32), points))
    bboxes = refine_bboxes(bboxes, cleaned_foreground, hsv_raw, img)
    score = calculate_bbox_quality(bboxes, cleaned_foreground)
    location_clusters = _location_visual(
        points,
        location_labels.astype(np.int32),
        img.shape[:2],
    )
    bbox_image = draw_bbox(img, bboxes)

    if not return_debug:
        return bboxes, score

    debug_images = {
        "color_dimension": color_dimension,
        "foreground_mask": cv.cvtColor(cleaned_foreground, cv.COLOR_GRAY2BGR),
        "location_clusters": location_clusters,
        "bbox_image": bbox_image,
    }
    return bboxes, score, debug_images


def segment_kmeans(img_path: str) -> list[BBox]:
    img = cv.imread(img_path)
    if img is None:
        return []
    bboxes, _ = segment_kmeans_image(img)
    return bboxes

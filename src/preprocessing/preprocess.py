from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

type Circle = tuple[int, int, int]
type Image = Any

DEFAULT_EXPORT_SIZE = 3000
MAX_WORKING_SIZE = 1024
MIN_WORKING_SIZE = 512


@dataclass(frozen=True)
class PreprocessArtifacts:
    square_image: Image
    export_image: Image
    masked_export_image: Image
    working_image: Image
    masked_working_image: Image
    working_mask: NDArray[np.uint8]
    export_mask: NDArray[np.uint8]
    circle_working: Circle
    circle_export: Circle
    circle_detected: bool


def center_crop_square(image: Image) -> Image:
    height, width = image.shape[:2]
    side = min(height, width)
    top = max((height - side) // 2, 0)
    left = max((width - side) // 2, 0)
    return image[top : top + side, left : left + side].copy()


def choose_working_size(
    side_length: int,
    *,
    max_size: int = MAX_WORKING_SIZE,
    min_size: int = MIN_WORKING_SIZE,
) -> int:
    return min(max(side_length, min_size), max_size)


def _resize_square(image: Image, size: int) -> Image:
    source_side = image.shape[0]
    interpolation = cv2.INTER_AREA if source_side >= size else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


def _odd_kernel(value: float, minimum: int, maximum: int) -> int:
    kernel = int(round(value))
    kernel = max(minimum, min(kernel, maximum))
    if kernel % 2 == 0:
        kernel += 1
    return min(kernel, maximum if maximum % 2 == 1 else maximum - 1)


def _sanitize_circle(circle: Circle, side: int) -> Circle:
    cx, cy, radius = circle
    max_radius = max(1, side // 2 - 1)
    radius = max(1, min(radius, max_radius))
    max_center = side - radius - 1
    min_center = radius
    cx = min(max(cx, min_center), max(max_center, min_center))
    cy = min(max(cy, min_center), max(max_center, min_center))
    return (int(cx), int(cy), int(radius))


def default_plate_circle(side: int) -> Circle:
    center = side // 2
    radius = int(round(side * 0.45))
    return _sanitize_circle((center, center, radius), side)


def detect_plate_circle(image: Image) -> Circle | None:
    side = min(image.shape[:2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_kernel = _odd_kernel(side * 0.008, minimum=5, maximum=15)
    blurred = cv2.medianBlur(gray, blur_kernel)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(int(side * 0.75), 1),
        param1=90,
        param2=max(35, int(side * 0.08)),
        minRadius=max(int(side * 0.35), 1),
        maxRadius=max(int(side * 0.52), 1),
    )

    if circles is None:
        return None

    candidates = np.round(circles[0, :]).astype(int)
    largest = max(candidates, key=lambda value: value[2])
    return _sanitize_circle((int(largest[0]), int(largest[1]), int(largest[2])), side)


def build_plate_mask(shape: tuple[int, int], circle: Circle) -> NDArray[np.uint8]:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (circle[0], circle[1]), circle[2], 255, -1)
    return mask


def apply_plate_mask(image: Image, mask: NDArray[np.uint8]) -> Image:
    masked = image.copy()
    masked[mask == 0] = 0
    return masked


def scale_circle(circle: Circle, from_size: int, to_size: int) -> Circle:
    scale = to_size / from_size
    scaled = (
        int(round(circle[0] * scale)),
        int(round(circle[1] * scale)),
        int(round(circle[2] * scale)),
    )
    return _sanitize_circle(scaled, to_size)


def prepare_image(
    image: Image,
    *,
    export_size: int = DEFAULT_EXPORT_SIZE,
    max_working_size: int = MAX_WORKING_SIZE,
    min_working_size: int = MIN_WORKING_SIZE,
) -> PreprocessArtifacts:
    square_image = center_crop_square(image)
    square_side = square_image.shape[0]
    working_size = choose_working_size(
        square_side,
        max_size=max_working_size,
        min_size=min_working_size,
    )

    working_image = _resize_square(square_image, working_size)
    circle_working = detect_plate_circle(working_image)
    circle_detected = circle_working is not None
    if circle_working is None:
        circle_working = default_plate_circle(working_size)

    working_mask = build_plate_mask(working_image.shape[:2], circle_working)
    masked_working_image = apply_plate_mask(working_image, working_mask)

    export_image = _resize_square(square_image, export_size)
    circle_export = scale_circle(circle_working, working_size, export_size)
    export_mask = build_plate_mask(export_image.shape[:2], circle_export)
    masked_export_image = apply_plate_mask(export_image, export_mask)

    return PreprocessArtifacts(
        square_image=square_image,
        export_image=export_image,
        masked_export_image=masked_export_image,
        working_image=working_image,
        masked_working_image=masked_working_image,
        working_mask=working_mask,
        export_mask=export_mask,
        circle_working=circle_working,
        circle_export=circle_export,
        circle_detected=circle_detected,
    )


def process_image(image: Image, output_size: int = 256) -> Image:
    return prepare_image(image, export_size=output_size).masked_export_image

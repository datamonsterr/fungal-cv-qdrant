import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from src.config import HEIGHT, WIDTH

def process_image(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Crop the Petri dish circle and resize to 256×256."""
    img = cv2.resize(img, (HEIGHT * 2, WIDTH * 2))
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=w,
        param1=90,
        param2=90,
        minRadius=min(w // 2, h // 2) // 2,
        maxRadius=min(w // 2, h // 2),
    )
    cropped: Optional[NDArray[np.uint8]] = None
    if circles is not None:
        circles: NDArray[np.uint16] = np.around(circles).astype(np.uint16)
        for circle in circles[0]:
            x, y, r = int(circle[0]), int(circle[1]), int(circle[2])

            # Mask outside circle
            mask = np.zeros_like(img)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
            img = np.where(mask == 0, 0, img)

            # Crop around circle ensuring boundaries are valid
            y = max(r, min(y, img.shape[0] - r))
            x = max(r, min(x, img.shape[1] - r))
            cropped = img[y - r : y + r, x - r : x + r, :]
            cropped = cv2.resize(cropped, (HEIGHT, WIDTH))
    return cropped if cropped is not None else img

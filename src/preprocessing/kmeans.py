from sklearn.cluster import KMeans
import numpy as np
from numpy.typing import NDArray
import cv2 as cv
import os
from typing import List, Dict, Any

def get_bbox(labels: NDArray[np.int32], mat: NDArray[np.int32]) -> List[Dict[str, int]]:
    clusters = [[] for _ in range(3)]
    for i, label in enumerate(labels):
        clusters[label].append(mat[i])

    bounding_boxes = []
    for cluster in clusters:
        if not cluster:
            continue
        cluster = np.array(cluster)
        y_min = int(np.min(cluster[:, 0]))
        y_max = int(np.max(cluster[:, 0]))
        x_min = int(np.min(cluster[:, 1]))
        x_max = int(np.max(cluster[:, 1]))
        bounding_boxes.append({
            "xmin": x_min,
            "ymin": y_min,
            "xmax": x_max,
            "ymax": y_max,
        })
    return bounding_boxes

def draw_bbox(img: NDArray[np.uint8], bboxes: List[Dict[str, int]]) -> NDArray[np.uint8]:
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    img_with_boxes = cv.cvtColor(img.copy(), cv.COLOR_HSV2RGB)
    for i, bbox in enumerate(bboxes):
        cv.rectangle(
            img_with_boxes,
            (bbox["xmin"], bbox["ymin"]),
            (bbox["xmax"], bbox["ymax"]),
            colors[i % len(colors)],
            2,
        )
    return img_with_boxes

def segment_kmeans(img_path: str) -> List[Dict[str, int]]:
    img = cv.imread(img_path)
    if img is None:
        return []
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    blur = cv.GaussianBlur(img, (9, 9), 0.95)

    mat = np.array(blur)
    h, w, _ = mat.shape
    n = h * w
    D = np.zeros((n, 3))

    crop_mask = np.zeros((h, w, 3), dtype=np.uint8)
    cv.circle(crop_mask, (128, 128), 115, [255, 255, 255], -1)

    for x in range(h):
        for y in range(w):
            if crop_mask[x, y, 0] == 255:
                D[x * w + y] = np.array([mat[x, y, 0], mat[x, y, 1], mat[x, y, 2]])
            else:
                D[x * w + y] = np.array([1000, 1000, 1000])

    kmeans = KMeans(n_clusters=3, random_state=0).fit(D)
    labels = kmeans.labels_

    masks = [np.zeros((h, w, 3)) for _ in range(3)]
    colors = [[0, 0, 255], [255, 0, 0], [0, 255, 0]]

    for x in range(h):
        for y in range(w):
            masks[labels[x * w + y]][x, y] = colors[labels[x * w + y]]

    mask = sum(masks)

    corners = np.array([
        [0, 0], [0, w], [h, 0], [h, w], [h // 2, 0], [0, w // 2], [h // 2, w], [h, w // 2]
    ])
    directions = np.array([
        [1, 1], [1, -1], [-1, 1], [-1, -1], [0, 1], [1, 0], [0, -1], [-1, 0]
    ])

    votes = np.zeros(8)
    for _ in range(120):
        corners += directions
        for j, corner in enumerate(corners):
            x, y = corner
            color = np.array(mask[int(x), int(y)]).argmax(axis=0)
            if color != 0 and votes[j] == 0:
                votes[j] = color

    avg_vote = int(np.round(np.average(votes)))
    if avg_vote == 0:
        seg_id = 1
    elif avg_vote == 1:
        seg_id = 0
    elif avg_vote == 2:
        seg_id = 2
    else:
        # Fallback or raise
        seg_id = 0

    new_mask = np.zeros((h, w))
    for x in range(h):
        for y in range(w):
            if masks[seg_id][x, y].any():
                new_mask[x, y] = 1

    P = np.argwhere(new_mask == 1)
    if len(P) == 0:
        return []
        
    kmeans_P = KMeans(n_clusters=3, random_state=0).fit(P)
    labels_P = kmeans_P.labels_

    bboxes = get_bbox(labels_P, P)
    return bboxes

# YOLOv26 Species Detection: Finetuning & Comparison Report

**MycoAI Research** | May 7, 2026

---

## Abstract

YOLOv26 nano segmentation model finetuned on 435-image fungal dataset with **8 Penicillium species classes** using manual Roboflow labels on Vast.ai GPU (NVIDIA RTX 2060, 12GB). Final best checkpoint achieves **mAP@50 = 81.56%**, **mAP@50-95 = 71.50%** on held-out test split. Compared against KMeans segmentation on 25 sampled prepared images: YOLOv26 averages **2.96 detections/image** vs KMeans **3.0**, with learned confidence-scored boxes.

---

## 1. Introduction

Species-aware colony detection is useful for downstream MycoAI retrieval and metadata enrichment. Existing pipeline uses KMeans segmentation (`src/preprocessing/kmeans.py`) to extract up to 3 colony regions per image. YOLOv26 adds learned detection with confidence scores and species-conditioned training labels.

This report covers:
- species-labeled dataset selection
- YOLOv26n finetuning on Vast.ai
- quantitative + qualitative comparison of YOLOv26 vs KMeans
- prepared-dataset inference artifact generation

---

## 2. Methodology

### 2.1 Dataset

| Property | Value |
|----------|-------|
| Source | `Dataset/manual_labeled_data_roboflow_species/` |
| Images | 435 |
| Classes | 8 Penicillium species |
| Split | 303 train / 45 test / 87 valid |
| Labels | Manual Roboflow bounding boxes |
| Format | YOLO segmentation/detection-compatible |

### 2.2 Model Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv26n-seg |
| Pretrained | COCO (`yolo26n-seg.pt`) |
| Epochs | 100 |
| Image size | 640×640 |
| Batch size | 8 |
| Workers | 2 |
| Early stopping | patience 20 |
| Device | RTX 2060 12GB |

### 2.3 Remote Execution

| Property | Value |
|----------|-------|
| Instance ID | 36259342 |
| GPU | NVIDIA RTX 2060 (12 GB) |
| SSH Port | 61872 |
| Training time | ~0.34 hours |
| GPU memory | ~2.07 GB |

### 2.4 Inference Protocol

Full inference run on all `prepared.jpg` leaves under `Dataset/prepared/`.
Per leaf:
1. YOLOv26 inference, top-3 boxes
2. KMeans segmentation, top-3 regions
3. Save `metadata.json`, `bbox_yolo26.jpg`, `pipeline_yolo26.jpg`, `segments/segment_yolo26_*.jpg`

For report comparison, 25 random prepared leaves sampled and rendered to `results/yolo26_comparison/`.

---

## 3. Results

### 3.1 Key Metrics

| Metric | Value |
|--------|-------|
| **Best mAP@50** | **0.81562** |
| **Best mAP@50-95** | **0.71501** |
| Best epoch | 84 |
| Final mAP@50 | 0.81074 |
| Epochs completed | 100 |
| Train images | 303 |
| Validation images | 45 |

### 3.2 Detection Comparison

- YOLOv26: **2.96 detections/image**
- KMeans: **3.0 detections/image**
- Comparison samples: **25**

### 3.3 Prepared Dataset Inference Outputs

| Artifact | Count |
|----------|-------|
| `metadata.json` | 100 |
| `bbox_yolo26.jpg` | 99 |
| `pipeline_yolo26.jpg` | 99 |
| `segments/segment_yolo26_*.jpg` | 289 |

One prepared leaf had zero YOLO detections, so no YOLO visualization was emitted there:
`Dataset/prepared/penicillium-polonicum/dto-148-c9/cyas/ob`

### 3.4 Artifact Paths

| Path | Description |
|------|-------------|
| `weights/yolo26/yolo26n-seg_species_best.pt` | best checkpoint |
| `weights/yolo26/yolo26n-seg_species_last.pt` | final checkpoint |
| `results/yolo26_finetune/train/results.csv` | per-epoch metrics |
| `results/yolo26_comparison/*.jpg` | 25 sampled comparison images |
| `Dataset/prepared/**/metadata.json` | per-leaf YOLO + KMeans boxes |
| `Dataset/prepared/**/segments/segment_yolo26_*.jpg` | YOLO crop outputs |

---

## 4. Discussion

Training outcome strong enough to continue downstream integration. Main improvement came from switching away from kmeans-generated training boxes to manual Roboflow labels with species classes. Inference bug also fixed so only `prepared.jpg` leaves are processed, not all images recursively.

KMeans still guarantees 3 heuristic regions, while YOLOv26 can abstain when confidence is low. That behavior caused exactly one no-detection leaf in this run.

---

## 5. Conclusion

Current best model: `weights/yolo26/yolo26n-seg_species_best.pt`.

Status:
- training complete
- full prepared inference complete
- metadata, segments, and visualizations written locally
- sampled comparison artifacts regenerated

---

## 6. References

- Ultralytics YOLOv26: https://docs.ultralytics.com/models/yolo26/
- MycoAI 006-yolo26-seg-finetune spec/plan/tasks
- Existing KMeans segmentation: `src/preprocessing/kmeans.py`

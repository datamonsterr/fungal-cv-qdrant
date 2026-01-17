# Fungal Species Classification: Comprehensive Report

## 1. Problem Statement


### 1.1 Overview
This project explores fungal species classification using computer vision techniques. Each sample corresponds to a fungal strain grown under multiple environments (media), each environment showing three colonies captured under a microscope from two angles: oblique (ob) and reverse (rev).

The goal is to predict the fungal species of a strain by comparing extracted visual features against a labeled reference database. The approach focuses on feature extraction and similarity-based inference rather than traditional supervised model fine-tuning.

 The design allows evaluation of robustness under environmental variations, illumination differences, and texture noise.

### 1.2 Dataset
The dataset follows a hierarchical structure: `Species` -> `Strain` -> `Environment` -> `Image`.

#### Dataset Statistics
| Metric | Value |
| :--- | :--- |
| **Total Files** | 435 |
| **Processed Files** | 435 |
| **Total Segments** | 1305 |
| **Failed Files** | 0 |
| **Unknown Species** | 0 |

#### Species and Strain Distribution (Train vs. Test Split)
| Species | Train Strains (Reference) | Test Strains (Query) |
| :--- | :--- | :--- |
| *Penicillium cyclopium* | DTO 148-C8 | - |
| *Penicillium polonicum* | DTO 148-C9, 157-A3, 217-D6, 248-G3, 402-I8, 469-G8 | DTO 148-D1 |
| *Penicillium melanoconidium* | DTO 148-D2, 216-I7, 470-H3 | DTO 158-D1 |
| *Penicillium viridicatum* | DTO 148-D3, 470-F1, 478-C6 | DTO 163-I2 |
| *Penicillium tricolor* | DTO 157-A4, 472-B6 | DTO 470-I9 |
| *Penicillium freii* | DTO 162-C6, 470-A1, 470-A2 | DTO 469-I4 |
| *Penicillium neoechinulatum* | DTO 206-F5, 251-A1, 470-F3 | DTO 217-D9 |
| *Penicillium aurantiogriseum* | DTO 457-A6, 470-H9, 473-D6 | DTO 469-I5 |

The data management strategy involves standardizing raw image filenames to extract strain, species, environment, and angle information. The dataset is segmented and organized to facilitate both automated processing and manual inspection.

### 1.3 Processed Data
Raw images of Petri dishes are processed to extract individual fungal colonies.

#### Example: Processing Pipeline (*Penicillium polonicum*, Strain DTO 148-D1)

| Stage | Image | Description |
| :--- | :--- | :--- |
| **1. Original** | <img src="../Dataset/original/DTO 148-D1 Penicillium polonicum/DTO 148-D1 MEAob_edited.jpg" width="200" /> | Raw microbiological image on MEA medium. |
| **2. Processed** | <img src="../Dataset/full_image/f9036734b8b95250a89d905f01fb67b7.jpg" width="200" /> | Preprocessed full-dish image: resized, circle-detected, and masked. |
| **3. Segmented** | <img src="../Dataset/hierarchical/Penicillium polonicum/DTO 148-D1/MEA/DTO_148-D1_MEA_ob_seg0.jpg" width="200" /> | Single isolated colony extracted via K-Means clustering. |

#### 1.3.1 Preprocessing
The preprocessing pipeline prepares raw images for segmentation:
1.  **Resizing**: Images are standardized to a consistent resolution.
2.  **Circle Hough Transform**: Detects the circular Petri dish boundary to isolate the dish from the background.
3.  **Cropping & Masking**: The background outside the dish is masked to black, and the image is cropped to the dish area, removing irrelevant visual noise.

#### 1.3.2 Segmentation (K-Means)
After cropping, individual colonies are segmented using K-Means clustering:
1.  **Color Space Conversion**: Image converted to HSV to better separate fungal colonies from the medium.
2.  **Blurring**: Gaussian blur is applied to reduce high-frequency noise.
3.  **Clustering**: K-Means clustering separates pixels into background (agar) and foreground (colonies).
4.  **Bounding Boxes**: Coordinates for each cluster are calculated and refined to extract 3 distinct segment images per dish, representing the three individual colonies.

#### 1.3.3 Visual Pipeline Diagram

![Visual Pipeline Diagram](mermaid/preprocessed_diagram_flow.png)

#### 1.3.4 Distinct Test Set
Testing is performed at the **strain level**. A predefined set of strains is reserved for testing to ensure no overlap with the reference database.
- **Query**: All segments from a specific test strain.
- **Database**: Contains segments from *other* strains.
- **Filtering**: When querying, "sibling" segments (segments originating from the same parent image as the query fragment) are strictly filtered out to prevent data leakage and ensure fair evaluation.

---

## 2. Methodology

### 2.1 Solution Overview

The system uses a retrieval-based classification approach.

![Solution Overview](mermaid/query_flow.png)

### 2.2 Feature Extractors
A variety of feature extractors are used to capture different visual characteristics. All extracted features are L2 normalized to facilitate cosine similarity comparison.

| Extractor | Type | Description | Weighting/Normalization |
| :--- | :--- | :--- | :--- |
| **HOG** | Hand-crafted | Histogram of Oriented Gradients. Captures shape and edge structure. | L2 Normalized. |
| **Gabor** | Hand-crafted | Gabor filters at multiple frequencies/orientations. Captures texture. | Mean/Std per kernel. L2 Normalized. |
| **ColorHist** | Hand-crafted | RGB Color Histogram (32 bins). | L2 Normalized. |
| **ColorHistHS** | Hand-crafted | HSV (Hue/Sat only) Histogram. focus on color profile. | L2 Normalized. |
| **ResNet50** | Deep Learning | Pretrained on ImageNet. Last FC layer removed. | Output dim: 2048. L2 Normalized. |
| **MobileNetV2** | Deep Learning | Pretrained on ImageNet. Lightweight. | Output dim: 1280. L2 Normalized. |
| **EfficientNetB1**| Deep Learning | Pretrained on ImageNet. | Output dim: 1280. L2 Normalized. |
| **HS+ResNet50** | Hybrid | Concatenation of HS Hist and ResNet50. | HS weighted **3.0x** before concat. |

### 2.3 Aggregation Strategy
Since a single strain yields multiple query segments, and each segment yields multiple retrieval results, predictions must be aggregated to form a consistent strain-level prediction.

- **Mechanism**:
    1. **Retrieval**: Retrieve `k=7` nearest neighbors for *each* segment of the query strain.
    2. **Scoring**: Sum the similarity scores (cosine distance) for each candidate species across all retrieved neighbors from all segments.
    3. **Weighted Strategy (Default)**: The final score for a species is its `Total Similarity Score / Total Neighbor Count`. This normalizes for species that might appear frequently but with low similarity.
    4. **Prediction**: The species with the highest final score is selected as the prediction for the strain.

### 2.4 Environment Selection Strategy
Evaluation is conducted using different strategies to test the model's robustness against limited environmental data:
- **E1 (Standard)**: Train on all available environments, Test on all available environments.
- **E2 (Balanced)**: Test sets are balanced to include equal representation from different growth media.
- **E3 (Single-Environment Test)**: The model is trained on all data but tested ONLY on images from a specific growth medium (e.g., MEA). This tests the model's ability to recognize species based on colony appearance in a specific condition.
- **E4 (Leave-One-Env-Out)**: The model is trained on all data but tested on all environments EXCEPT one specific medium. This evaluates generalization when a specific visual phenotype is missing from the test set.

### 2.5 Query Strategy
- **Database**: A vector database is used for high-speed similarity search.
- **Optimization**: To increase query velocity, all segments (both potential train and test data) are uploaded to oen collection. During the testing phase, the system dynamically filters the search space. When processing a test strain, the engine strictly **excludes** any vectors belonging to that specific strain from the search candidates, essentially creating a "leave-one-strain-out" scenario dynamically.

### 2.6 Combine Models (Ensemble)
To further improve accuracy, an ensemble approach is used.
- **Ranking**: Raw similarity results from multiple feature extractors (e.g., combining ResNet50 for texture/shape and ColorHistHS for color) are combined.
- **Strategy**: The probability scores or rankings from individual models are combined using a weighted sum or simple voting mechanism.
- **Complementary Analysis**: Analysis identifies "complementary" cases where one model type (e.g., Deep Learning) fails but another (e.g., Color Histogram) succeeds. This information is used to tune the ensemble weights.

---

## 3. System Components

### 3.1 Vector Database
The system utilizes a specialized vector search engine to manage embeddings. It handles the efficient storage and retrieval of high-dimensional vectors produced by the feature extractors, supporting Cosine similarity measures and metadata-based filtering.

### 3.2 Prediction and Evaluation Logic
The core prediction engine handles the logic of taking an input strain, retrieving its segments, querying the database, and aggregating the results.
The evaluation framework orchestrates batch testing runs, iterating over different feature extractors, environment strategies, and aggregation methods to generate comprehensive performance metrics.

### 3.3 Project Architecture
The project is structured as a modular pipeline:
- **Configuration**: Centralized management of paths, constants, and hyperparameters.
- **Orchestration**: A central controller handles CLI commands to trigger different stages of the pipeline (reformatting, feature extraction, training, prediction).

---

## 4. Evaluation Results

**Settings**: `k=7`, Distance Metric: Cosine.

### Visualizations
Results are stored in `results/`.

#### Confusion Matrix (ResNet50 Weighted Strategy)
![Confusion Matrix](../results/run_20260116_163442_k7/resnet50_weighted_all/confusion_matrix.png)

#### Prediction Examples

**1. Correct Prediction**: *Penicillium polonicum* (DTO 148-D1)
The system correctly retrieves similar segments from other *P. polonicum* strains.
![Correct Prediction](../results/run_20260116_163442_k7/resnet50_weighted_all/visualizations/correct/pred_DTO%20148-D1_set0.jpg)

**2. Incorrect Prediction**: *Penicillium melanoconidium* (DTO 158-D1)
The model confuses it with *P. aurantiogriseum* or *P. polonicum*, likely due to similar texture patterns in the retrieved neighbors.
![Incorrect Prediction](../results/run_20260116_163442_k7/resnet50_weighted_all/visualizations/incorrect/pred_DTO%20158-D1_set0.jpg)

### Observations
- **Deep Learning vs Hand-Crafted**: ResNet50 generally outperforms HOG/Gabor on texture but can overfit to background artifacts if not carefully segmented.
- **Color Importance**: The Hybrid `HS+ResNet50` extractor often yields better results by enforcing color constraints.
- **Environment Impact**: `E3` experiments show that some media (like MEA) are more discriminative than others (like DG18).

---

## 5. Conclusion and Future Work

### Conclusion
The current similarity-based approach demonstrates that fungal species can be classified by aggregated visual features from multiple colonies. Deep learning extractors provide strong baselines, but color histograms offer crucial complementary information.

### Future Work
1.  **Fine-tuning**: Instead of off-the-shelf ImageNet weights, fine-tune the backbone (ResNet/MobileNet) using Contrastive Learning (SimCLR) or Triplet Loss on the fungal dataset to learn a better embedding space.
2.  **Advanced Ensemble**: dynamic weighting based on query confidence.
3.  **End-to-End Pipeline**: Integrate segmentation and classification into a single model step.


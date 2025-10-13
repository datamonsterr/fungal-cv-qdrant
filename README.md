# Myco Fungi Feature Extraction and Classification

This project experiments with different feature extractor methods for classifying myco fungi species from images.

## Prerequisites

- Python 3.8+
- Docker (for Qdrant vector database)
- Nix (optional, for reproducible environment)

## 1. Environment Setup

### Using Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Using Nix (Optional)

```bash
nix-shell
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 2. Setup Qdrant Vector Database

Run Qdrant using Docker:

```bash
# Pull and run Qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

Qdrant will be available at `http://localhost:6333`

## 3. Dataset Processing

### Prepare Dataset Structure

Update the `ORIGINAL_DATASET_PATH` in `reformat_dataset.py`:

```python
ORIGINAL_DATASET_PATH = "../Dataset/original"  # Change this to your dataset path
```

### Run Dataset Reformatting

This script:
- Preprocesses raw images
- Segments fungi colonies using K-means clustering
- Generates metadata for full and segmented images

```bash
python reformat_dataset.py
```

**Output:**
- `../Dataset/full_image/` - Preprocessed full images
- `../Dataset/segmented_image/` - Segmented colony images
- `../Dataset/full_image_metadata.json` - Metadata for full images
- `../Dataset/segmented_image_metadata.json` - Metadata for segments

## 4. Feature Extraction

Extract features using multiple methods (HOG, Gabor, Color Histogram, ResNet50):

```bash
python feature_extractors.py
```

**Output:**
- `../Dataset/segmented_features.json` - Extracted features for all segments

**Feature extractors:**
- **HOG**: Histogram of Oriented Gradients
- **Gabor**: Gabor filter responses
- **ColorHistogram**: RGB color distribution
- **ResNet50**: Deep learning features (pre-trained on ImageNet)

## 5. Upload Features to Qdrant

Upload extracted features to the vector database:

```bash
python upload_qdrant.py
```

This creates a collection named `myco_fungi_features` with multiple named vectors for each feature type.

## 6. Query Similar Images

### Using the Query Script

Run the example query script:

```bash
python query.py
```

This generates visualizations showing query images and their nearest neighbors for each feature type.

**Output:** `./results/{image_id}_{feature_type}.jpg`

### Using Query Utils Programmatically

```python
from qdrant_client import QdrantClient
from query_utils import (
    find_nearest_neighbors_by_id,
    find_nearest_neighbors_by_image,
    get_image_metadata,
    visualize_neighbors
)
from feature_extractors import ResNet50Extractor

# Initialize client
client = QdrantClient(host="localhost", port=6333)
collection_name = "myco_fungi_features"

# Query by image ID (already in database)
neighbors = find_nearest_neighbors_by_id(
    client=client,
    collection_name=collection_name,
    query_image_id="your_image_id",
    feature_type="resnet50",  # hog, gabor, colorhistogram, resnet50
    num_neighbors=10,
    environment="CYA",  # Optional: filter by environment
    exclude_self=True
)

# Query by new image file (not in database)
extractor = ResNet50Extractor()
neighbors = find_nearest_neighbors_by_image(
    client=client,
    collection_name=collection_name,
    image_path="path/to/query/image.jpg",
    extractor=extractor,
    feature_type="resnet50",
    num_neighbors=10
)

# Get metadata for an image
metadata = get_image_metadata(
    client=client,
    collection_name=collection_name,
    image_id="your_image_id"
)

# Visualize results
visualize_neighbors(
    query_image_path="path/to/query/image.jpg",
    neighbors=neighbors,
    segmented_image_dir="../Dataset/segmented_image",
    output_path="output.jpg",
    query_metadata=metadata,
    max_neighbors=5
)
```

## Project Structure

```
.
├── config.py                          # Configuration settings
├── feature_extractors.py              # Feature extraction implementations
├── feature_utils.py                   # Feature utility functions
├── kmeans.py                          # K-means segmentation
├── preprocess.py                      # Image preprocessing
├── query_utils.py                     # Query and visualization utilities
├── query.py                           # Example query script
├── reformat_dataset.py                # Dataset processing pipeline
├── upload_qdrant.py                   # Upload features to Qdrant
├── requirements.txt                   # Python dependencies
└── examples/                          # Example query results
```

## Query Filters

You can filter queries by metadata:

- `environment`: Growth medium (e.g., "CYA", "MEA")
- `angle`: Viewing angle ("ob" or "rev")
- `strain`: Strain identifier (e.g., "DTO 123-A1")
- `specy`: Species name

```python
neighbors = find_nearest_neighbors_by_id(
    client=client,
    collection_name=collection_name,
    query_image_id="your_image_id",
    feature_type="resnet50",
    environment="CYA",
    angle="ob",
    num_neighbors=10
)
```

## Notes

- **Pydantic** is used for data validation and settings management
- Always use `uv` for package management when in Nix shell
- Add new dependencies to `requirements.txt`
- Ensure Qdrant is running before uploading or querying

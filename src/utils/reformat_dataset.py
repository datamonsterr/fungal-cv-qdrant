import json
import os
import re
import shutil
import uuid

import cv2
import pandas as pd

from src.config import (
    DATASET_ROOT,
    FULL_IMAGE_METADATA_PATH,
    FULL_IMAGE_PATH,
    ORIGINAL_DATASET_PATH,
    PROJECT_ROOT,
    SEGMENTED_IMAGE_DIR,
    SEGMENTED_METADATA_PATH,
    STRAIN_SPECIES_MAPPING_PATH,
)
from src.experiments.preprocessing.kmeans import segment_kmeans
from src.experiments.preprocessing.preprocess import process_image

POSSIBLE_ENVIRONMENTS = []
POSSIBLE_ANGLES = ["ob", "rev"]
FILE_EXTENSION = ".jpg"
HIERARCHICAL_DATASET_PATH = DATASET_ROOT / "hierarchical"


def get_specy_from_strain(strain: str, strain_to_specy: pd.DataFrame) -> str | None:
    result = strain_to_specy[strain_to_specy["Strain"] == strain]
    if not result.empty:
        return result["Species"].iloc[0]
    return None


class Metadata:
    def __init__(self, filename: str, id: str, strain_to_specy: pd.DataFrame):
        self.filename = filename
        self._clean_filename()
        self.id = id
        match = re.match(r"(DTO\s[0-9]+-[A-Z0-9]+)\s([A-Z0-9]+)(rev|ob)", self.filename)

        self.strain = "unknown"
        self.environment = "unknown"
        self.angle = "unknown"

        if match:
            self.strain = match.group(1)
            self.environment = match.group(2)
            self.angle = match.group(3)

        self.specy = get_specy_from_strain(self.strain, strain_to_specy) or "unknown"

    def get_metadata(self) -> dict[str, dict[str, str] | str]:
        return {
            "id": self.id,
            "data": {
                "strain": self.strain,
                "environment": self.environment,
                "angle": self.angle,
                "specy": self.specy,
            },
        }

    def _clean_filename(self):
        self.filename = self.filename.removesuffix(FILE_EXTENSION)
        self.filename = self.filename.removesuffix("_edited")


def _save_hierarchical_segment(
    metadata: Metadata,
    segment_img,
    segment_index: int,
) -> str:
    """Save one segmented image into Dataset/hierarchical and return relative path."""
    clean_strain = metadata.strain.replace(" ", "_").replace("/", "-")
    hierarchical_filename = (
        f"{clean_strain}_{metadata.environment}_{metadata.angle}"
        f"_seg{segment_index}{FILE_EXTENSION}"
    )
    hierarchical_dir = (
        HIERARCHICAL_DATASET_PATH
        / metadata.specy
        / metadata.strain
        / metadata.environment
    )
    hierarchical_dir.mkdir(parents=True, exist_ok=True)
    hierarchical_file_path = hierarchical_dir / hierarchical_filename
    cv2.imwrite(str(hierarchical_file_path), segment_img)
    return str(hierarchical_file_path.relative_to(PROJECT_ROOT))


def _save_hierarchical_original(
    metadata: Metadata,
    image_id: str,
    full_img_path,
) -> str:
    """Save one full/original image into Dataset/hierarchical and return relative path."""
    clean_strain = metadata.strain.replace(" ", "_").replace("/", "-")
    hierarchical_filename = (
        f"{clean_strain}_{metadata.environment}_{metadata.angle}_{image_id}_original"
        f"{FILE_EXTENSION}"
    )
    hierarchical_dir = (
        HIERARCHICAL_DATASET_PATH
        / metadata.specy
        / metadata.strain
        / metadata.environment
    )
    hierarchical_dir.mkdir(parents=True, exist_ok=True)
    hierarchical_file_path = hierarchical_dir / hierarchical_filename
    shutil.copyfile(full_img_path, hierarchical_file_path)
    return str(hierarchical_file_path.relative_to(PROJECT_ROOT))


def reformat_dataset(create_hierarchical: bool = True):
    # Load mapping
    if os.path.exists(STRAIN_SPECIES_MAPPING_PATH):
        strain_to_specy = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
    else:
        print(f"Warning: {STRAIN_SPECIES_MAPPING_PATH} not found.")
        strain_to_specy = pd.DataFrame(columns=["Strain", "Species"])

    # Create directories if they don't exist
    FULL_IMAGE_PATH.mkdir(parents=True, exist_ok=True)
    SEGMENTED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    if not FULL_IMAGE_METADATA_PATH.exists():
        with open(FULL_IMAGE_METADATA_PATH, "w") as f:
            json.dump([], f)

    if not SEGMENTED_METADATA_PATH.exists():
        with open(SEGMENTED_METADATA_PATH, "w") as f:
            json.dump([], f)

    metadata_list: list[dict[str, dict[str, str] | str]] = []
    segment_metadata_list: list[dict[str, dict[str, str] | str]] = []

    if not ORIGINAL_DATASET_PATH.exists():
        print(f"Error: {ORIGINAL_DATASET_PATH} does not exist.")
        return

    for dir_name in os.listdir(ORIGINAL_DATASET_PATH):
        dir_path = ORIGINAL_DATASET_PATH / dir_name
        if not dir_path.is_dir():
            continue

        for filename in os.listdir(dir_path):
            if filename.endswith(FILE_EXTENSION):
                # Generate ID and metadata for full image
                id = uuid.uuid5(uuid.NAMESPACE_DNS, filename).hex
                metadata = Metadata(filename, id, strain_to_specy)

                original_img_path = dir_path / filename
                full_img_path = FULL_IMAGE_PATH / f"{id}{FILE_EXTENSION}"

                # Copy full image
                shutil.copyfile(original_img_path, full_img_path)
                full_metadata = metadata.get_metadata()
                full_metadata["file_path"] = str(
                    full_img_path.relative_to(PROJECT_ROOT)
                )
                if create_hierarchical:
                    full_metadata["hierarchical_path"] = _save_hierarchical_original(
                        metadata=metadata,
                        image_id=id,
                        full_img_path=full_img_path,
                    )
                metadata_list.append(full_metadata)

                print(f"Processing {filename}...")

                # Step 1: Preprocess the image
                img = cv2.imread(str(full_img_path))
                if img is None:
                    print(f"Failed to read {full_img_path}")
                    continue

                preprocessed_img = process_image(img)

                # Save preprocessed image temporarily for segmentation
                temp_path = f"temp_{id}.jpg"
                cv2.imwrite(temp_path, preprocessed_img)

                # Step 2: Segment using KMeans
                try:
                    bboxes = segment_kmeans(temp_path)

                    # Step 3: Crop and save segmented images
                    for i, bbox in enumerate(bboxes):
                        segment_id = f"{id}_{i}"
                        segment_img = preprocessed_img[
                            bbox["ymin"] : bbox["ymax"], bbox["xmin"] : bbox["xmax"]
                        ]

                        # Resize to standard size if needed, or keep as is?
                        # Config says HEIGHT/WIDTH is 256.
                        # Let's resize to ensure consistency
                        segment_img = cv2.resize(segment_img, (256, 256))

                        segment_path = (
                            SEGMENTED_IMAGE_DIR / f"{segment_id}{FILE_EXTENSION}"
                        )
                        cv2.imwrite(str(segment_path), segment_img)

                        # Create metadata for segmented image
                        seg_metadata = metadata.get_metadata()
                        seg_metadata["id"] = segment_id
                        seg_metadata["parent_id"] = id

                        # Add paths
                        seg_metadata["file_path"] = str(
                            segment_path.relative_to(PROJECT_ROOT)
                        )

                        if create_hierarchical:
                            seg_metadata["hierarchical_path"] = (
                                _save_hierarchical_segment(
                                    metadata=metadata,
                                    segment_img=segment_img,
                                    segment_index=i,
                                )
                            )

                        segment_metadata_list.append(seg_metadata)

                except Exception as e:
                    print(f"Error segmenting {filename}: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    # Save metadata files
    with open(FULL_IMAGE_METADATA_PATH, "w") as f:
        json.dump(metadata_list, f, indent=4)

    with open(SEGMENTED_METADATA_PATH, "w") as f:
        json.dump(segment_metadata_list, f, indent=4)

    print("\nProcessing complete!")
    print(f"Full images: {len(metadata_list)}")
    print(f"Segmented images: {len(segment_metadata_list)}")


if __name__ == "__main__":
    reformat_dataset()

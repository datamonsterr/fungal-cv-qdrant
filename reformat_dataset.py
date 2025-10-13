import os
import uuid
import json
import re
import shutil
import pandas as pd
import cv2
from kmeans import segment_kmeans
from preprocess import process_image

ORIGINAL_DATASET_PATH = "../Dataset/original"
FULL_IMAGE_PATH = "../Dataset/full_image"
FULL_IMAGE_METADATA_PATH = "../Dataset/full_image_metadata.json"
SEGMENT_IMAGE_PATH = "../Dataset/segmented_image"
SEGMENT_IMAGE_PATH_METADATA_PATH = "../Dataset/segmented_image_metadata.json"
STRAIN_SPECIES_MAPPING_PATH = "../Dataset/strain_to_specy.csv"

POSSIBLE_ENVIRONMENTS = []
POSSIBLE_ANGLES = ["ob", "rev"]
FILE_EXTENSION = ".jpg"
METADATA_LIST: list[dict[str, dict[str,str] | str]] = []

strain_to_specy: pd.DataFrame = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)

def get_specy_from_strain(strain: str) -> str | None:
    result = strain_to_specy[strain_to_specy['Strain'] == strain]
    if not result.empty:
        return result['Species'].iloc[0]
    return None

class Metadata:
    def __init__(self, filename: str, id: str):
        self.filename = filename
        self._clean_filename()
        self.id = id
        match = re.match(r'(DTO\s[0-9]+-[A-Z0-9]+)\s([A-Z0-9]+)(rev|ob)', self.filename)

        self.strain = "unknown"
        self.environment = "unknown"
        self.angle = "unknown"

        if match:
            self.strain = match.group(1)
            self.environment = match.group(2)
            self.angle = match.group(3)

        self.specy = get_specy_from_strain(self.strain) or "unknown"

    def get_metadata(self) -> dict[str, dict[str, str] | str]:
        return {
            "id": self.id,
            "data": {
                "strain": self.strain,
                "environment": self.environment,
                "angle": self.angle,
                "specy": self.specy
            }
        }

    def _clean_filename(self):
        self.filename = self.filename.removesuffix(FILE_EXTENSION)
        self.filename = self.filename.removesuffix("_edited")
        
def main():
    # Create directories if they don't exist
    if not os.path.exists(FULL_IMAGE_PATH):
        os.makedirs(FULL_IMAGE_PATH)
    
    if not os.path.exists(SEGMENT_IMAGE_PATH):
        os.makedirs(SEGMENT_IMAGE_PATH)

    if not os.path.exists(FULL_IMAGE_METADATA_PATH):
        open(FULL_IMAGE_METADATA_PATH, 'w').close()
    
    if not os.path.exists(SEGMENT_IMAGE_PATH_METADATA_PATH):
        open(SEGMENT_IMAGE_PATH_METADATA_PATH, 'w').close()

    segment_metadata_list: list[dict[str, dict[str, str] | str]] = []

    for dir in os.listdir(ORIGINAL_DATASET_PATH):
        dir_path = os.path.join(ORIGINAL_DATASET_PATH, dir)
        if not os.path.isdir(dir_path):
            continue
            
        for filename in os.listdir(dir_path):
            if filename.endswith(FILE_EXTENSION):
                # Generate ID and metadata for full image
                id = uuid.uuid5(uuid.NAMESPACE_DNS, filename).hex
                metadata = Metadata(filename, id)
                
                original_img_path = os.path.join(ORIGINAL_DATASET_PATH, dir, filename)
                full_img_path = os.path.join(FULL_IMAGE_PATH, f"{id}{FILE_EXTENSION}")
                
                # Copy full image
                shutil.copyfile(original_img_path, full_img_path)
                METADATA_LIST.append(metadata.get_metadata())
                
                print(f"Processing {filename}...")
                
                # Step 1: Preprocess the image
                img = cv2.imread(full_img_path)
                if img is None:
                    print(f"Failed to read {full_img_path}, skipping...")
                    continue
                    
                preprocessed_img = process_image(img)
                
                # Save preprocessed image temporarily for segmentation
                temp_preprocessed_path = os.path.join(FULL_IMAGE_PATH, f"{id}_preprocessed{FILE_EXTENSION}")
                cv2.imwrite(temp_preprocessed_path, preprocessed_img)
                
                try:
                    # Step 2: Segment the preprocessed image into 3 images using kmeans
                    bboxes = segment_kmeans(temp_preprocessed_path)
                    
                    # Step 3: Save the 3 segmented images with IDs and metadata
                    for idx, bbox in enumerate(bboxes):
                        # Generate unique ID for each segment
                        segment_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_segment_{idx}").hex
                        
                        # Crop the segment from preprocessed image
                        xmin, ymin = bbox["xmin"], bbox["ymin"]
                        xmax, ymax = bbox["xmax"], bbox["ymax"]
                        segmented_img = preprocessed_img[ymin:ymax, xmin:xmax]
                        
                        # Save segmented image
                        segment_img_path = os.path.join(SEGMENT_IMAGE_PATH, f"{segment_id}{FILE_EXTENSION}")
                        cv2.imwrite(segment_img_path, segmented_img)
                        
                        # Create metadata for segmented image (same as full image metadata)
                        segment_metadata = {
                            "id": segment_id,
                            "parent_id": id,
                            "segment_index": idx,
                            "bbox": bbox,
                            "data": metadata.get_metadata()["data"]
                        }
                        segment_metadata_list.append(segment_metadata)
                        
                    print(f"Segmented {filename} into {len(bboxes)} images")
                    
                except Exception as e:
                    print(f"Failed to segment {filename}: {e}")
                
                # Clean up temporary preprocessed image
                if os.path.exists(temp_preprocessed_path):
                    os.remove(temp_preprocessed_path)
    
    # Save metadata files
    with open(FULL_IMAGE_METADATA_PATH, 'w') as f:
        json.dump(METADATA_LIST, f, indent=4)
    
    with open(SEGMENT_IMAGE_PATH_METADATA_PATH, 'w') as f:
        json.dump(segment_metadata_list, f, indent=4)
    
    print(f"\nProcessing complete!")
    print(f"Full images: {len(METADATA_LIST)}")
    print(f"Segmented images: {len(segment_metadata_list)}")

if __name__ == "__main__":
    main()
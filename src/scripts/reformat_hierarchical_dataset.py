"""
Reformat dataset into hierarchical structure: {species_name}/{strain_name}/{environment}/
Each leaf directory contains ~6 images (preprocessed and segmented into 3 parts each = 6 images total per original image pair)
"""

import os
import json
import shutil
import pandas as pd
import cv2
from pathlib import Path
from typing import Dict, List
from src.preprocessing.kmeans import segment_kmeans
from src.preprocessing.preprocess import process_image
from src.config import (
    ORIGINAL_DATASET_PATH,
    STRAIN_SPECIES_MAPPING_PATH,
    DATASET_ROOT
)

# Target path
HIERARCHICAL_DATASET_PATH = DATASET_ROOT / "hierarchical"

# Constants
FILE_EXTENSION = ".jpg"


def get_specy_from_strain(strain: str, strain_to_specy: pd.DataFrame) -> str | None:
    """Get species name from strain name using the mapping CSV."""
    result = strain_to_specy[strain_to_specy['Strain'] == strain]
    if not result.empty:
        return result['Species'].iloc[0]
    return None


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse filename to extract metadata.
    Expected format: 'DTO {number}-{code} {environment}{angle}.jpg'
    Example: 'DTO 123-A1 CREAob.jpg'
    """
    # Remove extension and _edited suffix if present
    clean_name = filename.removesuffix(FILE_EXTENSION).removesuffix("_edited")
    
    # Try to parse the filename
    import re
    match = re.match(r'(DTO\s[0-9]+-[A-Z0-9]+)\s([A-Z0-9]+)(rev|ob)', clean_name)
    
    if match:
        strain = match.group(1)
        environment = match.group(2)
        angle = match.group(3)
        return {
            "strain": strain,
            "environment": environment,
            "angle": angle
        }
    else:
        print(f"Warning: Could not parse filename: {filename}")
        return {
            "strain": "unknown",
            "environment": "unknown",
            "angle": "unknown"
        }


def create_hierarchical_structure(
    original_img_path: str,
    filename: str,
    strain: str,
    specy: str,
    environment: str,
    angle: str,
    output_base: str
) -> int:
    """
    Process a single image: preprocess and segment into 3 parts.
    Save to hierarchical structure.
    Returns number of images saved.
    """
    # Create target directory structure
    target_dir = Path(output_base) / specy / strain / environment
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Read and preprocess image
    img = cv2.imread(original_img_path)
    if img is None:
        print(f"Failed to read {original_img_path}, skipping...")
        return 0
    
    preprocessed_img = process_image(img)
    
    # Save preprocessed image temporarily for segmentation
    temp_preprocessed_path = str(target_dir / f"temp_{filename}")
    cv2.imwrite(temp_preprocessed_path, preprocessed_img)
    
    saved_count = 0
    try:
        # Segment the preprocessed image
        bboxes = segment_kmeans(temp_preprocessed_path)
        
        # Save each segmented image
        for idx, bbox in enumerate(bboxes):
            xmin, ymin = bbox["xmin"], bbox["ymin"]
            xmax, ymax = bbox["xmax"], bbox["ymax"]
            segmented_img = preprocessed_img[ymin:ymax, xmin:xmax]
            
            # Create filename: {strain}_{environment}_{angle}_segment{idx}.jpg
            clean_strain = strain.replace(" ", "_").replace("/", "-")
            segment_filename = f"{clean_strain}_{environment}_{angle}_seg{idx}{FILE_EXTENSION}"
            segment_path = target_dir / segment_filename
            
            cv2.imwrite(str(segment_path), segmented_img)
            saved_count += 1
            
        print(f"✓ Saved {saved_count} segments from {filename} to {target_dir}")
        
    except Exception as e:
        print(f"✗ Failed to segment {filename}: {e}")
    
    finally:
        # Clean up temporary preprocessed image
        if os.path.exists(temp_preprocessed_path):
            os.remove(temp_preprocessed_path)
    
    return saved_count


def main():
    """Main function to reformat the dataset into hierarchical structure."""
    print("=" * 80)
    print("Reformatting Dataset into Hierarchical Structure")
    print("=" * 80)
    
    # Load strain to species mapping
    if not os.path.exists(STRAIN_SPECIES_MAPPING_PATH):
        print(f"Error: Strain-to-species mapping not found at {STRAIN_SPECIES_MAPPING_PATH}")
        return
    
    strain_to_specy = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
    print(f"Loaded {len(strain_to_specy)} strain-to-species mappings")
    
    # Create output directory
    if os.path.exists(HIERARCHICAL_DATASET_PATH):
        response = input(f"{HIERARCHICAL_DATASET_PATH} already exists. Remove and recreate? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(HIERARCHICAL_DATASET_PATH)
        else:
            print("Aborted.")
            return
    
    os.makedirs(HIERARCHICAL_DATASET_PATH, exist_ok=True)
    
    # Statistics
    stats = {
        "total_files": 0,
        "processed_files": 0,
        "total_segments": 0,
        "failed_files": 0,
        "unknown_species": 0
    }
    
    # Track images per directory for reporting
    images_per_dir: Dict[str, int] = {}
    
    # Process each directory in the original dataset
    for dir_name in sorted(os.listdir(ORIGINAL_DATASET_PATH)):
        dir_path = os.path.join(ORIGINAL_DATASET_PATH, dir_name)
        if not os.path.isdir(dir_path):
            continue
        
        print(f"\nProcessing directory: {dir_name}")
        print("-" * 80)
        
        for filename in sorted(os.listdir(dir_path)):
            if not filename.endswith(FILE_EXTENSION):
                continue
            
            stats["total_files"] += 1
            
            # Parse filename to get metadata
            metadata = parse_filename(filename)
            strain = metadata["strain"]
            environment = metadata["environment"]
            angle = metadata["angle"]
            
            # Get species from strain
            specy = get_specy_from_strain(strain, strain_to_specy)
            if specy is None:
                specy = "unknown_species"
                stats["unknown_species"] += 1
            
            # Process and save
            original_img_path = os.path.join(dir_path, filename)
            segments_saved = create_hierarchical_structure(
                original_img_path=original_img_path,
                filename=filename,
                strain=strain,
                specy=specy,
                environment=environment,
                angle=angle,
                output_base=HIERARCHICAL_DATASET_PATH
            )
            
            if segments_saved > 0:
                stats["processed_files"] += 1
                stats["total_segments"] += segments_saved
                
                # Track images per directory
                dir_key = f"{specy}/{strain}/{environment}"
                images_per_dir[dir_key] = images_per_dir.get(dir_key, 0) + segments_saved
            else:
                stats["failed_files"] += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files found:        {stats['total_files']}")
    print(f"Successfully processed:   {stats['processed_files']}")
    print(f"Failed to process:        {stats['failed_files']}")
    print(f"Unknown species:          {stats['unknown_species']}")
    print(f"Total segments created:   {stats['total_segments']}")
    print(f"\nUnique directories:       {len(images_per_dir)}")
    
    # Show sample of directories with image counts
    print("\nSample of directories with image counts:")
    print("-" * 80)
    for dir_key, count in sorted(images_per_dir.items())[:20]:
        print(f"  {dir_key}: {count} images")
    
    if len(images_per_dir) > 20:
        print(f"  ... and {len(images_per_dir) - 20} more directories")
    
    # Save directory statistics to JSON
    stats_file = os.path.join(HIERARCHICAL_DATASET_PATH, "dataset_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump({
            "summary": stats,
            "images_per_directory": images_per_dir
        }, f, indent=2)
    
    print(f"\nStatistics saved to: {stats_file}")
    print(f"Dataset saved to: {HIERARCHICAL_DATASET_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()

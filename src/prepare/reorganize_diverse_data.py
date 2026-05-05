"""
Reorganize Dataset/new_data into Dataset/diverse_data with hierarchical structure.

Target structure:
    Dataset/diverse_data/
    └── images/
        └── {species}/
            └── {environment}/
                ├── {Strain}_{angle}_{id}_original.jpg    (original source)
                ├── {Strain}_{angle}_{id}_centered.jpg   (width cropped to height)
                ├── {Strain}_{angle}_{id}_resized.jpg    (resized to 512x512)
                ├── {Strain}_{angle}_{id}_cropped.jpg    (Hough circle cropped)
                ├── {Strain}_{angle}_{id}.jpg            (preprocessed 256x256)
                ├── {Strain}_{angle}_{id}_bboxes.jpg      (bbox visualization)
                └── {Strain}_{angle}_{id}_seg{n}.jpg     (segmented colonies)
    diverse_data_metadata.json (with step_images paths and bbox info)

Usage:
    uv run python src/prepare/reorganize_diverse_data.py --mode pipeline
"""

import argparse
import json
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_ROOT, relative_to_workspace  # noqa: E402
from src.preprocessing.kmeans import segment_kmeans  # noqa: E402

# Constants
IMG_SIZE = 256
PREPROCESS_SIZE = 512  # Resize to this first before Hough detection
CIRCLE_RADIUS = 112

NEW_DATA_PATH = DATASET_ROOT / "new_data"
DIVERSE_DATA_PATH = DATASET_ROOT / "diverse_data"
DIVERSE_IMAGES_PATH = DIVERSE_DATA_PATH / "images"
DIVERSE_METADATA_PATH = DIVERSE_DATA_PATH / "diverse_data_metadata.json"

KNOWN_ENVS = {
    "CREA",
    "CYA",
    "CYAS",
    "MEA",
    "DG18",
    "YES",
    "OA",
    "MALT",
    "STEFF",
    "Sabouraud",
}


def normalize_angle(angle_raw: str) -> str:
    angle = angle_raw.lower().strip()
    if angle in ("ob", "o", "a"):
        return "ob"
    elif angle in ("rev", "r", "b"):
        return "rev"
    return angle


def parse_filename(filename: str, species_folder: str = "") -> dict:
    name = Path(filename).stem
    name = " ".join(name.split())
    name = re.sub(
        r"\s*(edited|detail\s*colony|Auto)\s*$", "", name, flags=re.IGNORECASE
    )

    def extract_env_angle_from_suffix(s: str) -> tuple:
        s = s.upper()
        for env in KNOWN_ENVS:
            if s.startswith(env.upper()):
                rest = s[len(env) :]
                if rest in ("O", "OB", "A"):
                    return env, "ob"
                elif rest in ("R", "REV", "B"):
                    return env, "rev"
                elif rest == "":
                    return env, "ob"
        return None, None

    parts = name.split()
    if len(parts) >= 2:
        env, angle = extract_env_angle_from_suffix(parts[-1])
        if env:
            return {"strain": " ".join(parts[:-1]), "environment": env, "angle": angle}

    if len(parts) >= 3:
        second_last = parts[-2].upper()
        last = parts[-1].lower()
        if second_last in KNOWN_ENVS:
            return {
                "strain": " ".join(parts[:-2]),
                "environment": second_last,
                "angle": normalize_angle(last),
            }

    for env in KNOWN_ENVS:
        pattern = rf"^\S+\s+(CBS\s+\d+[_\.]\d+)\s+{env}(o|r|ob|rev|b)?$"
        match = re.match(pattern, name, re.IGNORECASE)
        if match:
            return {
                "strain": match.group(1),
                "environment": env,
                "angle": normalize_angle(match.group(2) or ""),
            }

        pattern = rf"^\S+\s+(IBT\s+\d+)\s+{env}(o|r|ob|rev|b)?$"
        match = re.match(pattern, name, re.IGNORECASE)
        if match:
            return {
                "strain": match.group(1),
                "environment": env,
                "angle": normalize_angle(match.group(2) or ""),
            }

    strain_pattern = r"^(T\(N\)|T\d+|IBT\s+\d+|CBS\s+\d+[._]\d+)"
    for env in KNOWN_ENVS:
        pattern = rf"^{strain_pattern}\s+{env}\s+(ob|rev|o|r|a|b)$"
        match = re.match(pattern, name, re.IGNORECASE)
        if match:
            return {
                "strain": match.group(1),
                "environment": env,
                "angle": normalize_angle(match.group(2)),
            }

    parts = name.split()
    if len(parts) >= 3:
        for i, p in enumerate(parts[:-2]):
            if p.lower() in {"ob", "rev", "o", "r"}:
                potential_env = parts[i + 1].upper()
                if potential_env in KNOWN_ENVS:
                    return {
                        "strain": " ".join(parts[:i]),
                        "environment": potential_env,
                        "angle": normalize_angle(p),
                    }

    for env in KNOWN_ENVS:
        if env in name.upper():
            if " ob " in name.lower() or name.lower().endswith(" ob"):
                angle = "ob"
            elif " rev " in name.lower() or name.lower().endswith(" rev"):
                angle = "rev"
            else:
                angle = "ob"
            idx = name.upper().find(env)
            strain = name[:idx].strip()
            if strain:
                return {"strain": strain, "environment": env, "angle": angle}

    return {"strain": name, "environment": "UNKNOWN", "angle": "UNKNOWN"}


def find_all_images(new_data_path: Path) -> list:
    extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
    images = []

    for letter_dir in sorted(new_data_path.iterdir()):
        if letter_dir.is_file():
            continue
        for species_dir in sorted(letter_dir.iterdir()):
            if species_dir.is_file():
                continue
            species_name = species_dir.name

            for item in sorted(species_dir.iterdir()):
                if item.is_file():
                    ext = item.suffix.lower()
                    if ext not in [".jpg", ".jpeg", ".png"]:
                        continue
                    parsed = parse_filename(item.name, species_name)
                    images.append(
                        {
                            "source_path": str(item),
                            "species": species_name,
                            "strain": parsed["strain"],
                            "environment": parsed["environment"],
                            "angle": parsed["angle"],
                            "filename": item.name,
                        }
                    )
                else:
                    strain_name = item.name
                    for img_file in sorted(item.iterdir()):
                        if not img_file.is_file():
                            continue
                        ext = img_file.suffix.lower()
                        if ext not in extensions:
                            continue
                        parsed = parse_filename(img_file.name, species_name)
                        strain = (
                            parsed["strain"]
                            if parsed["strain"] != "UNKNOWN"
                            else strain_name
                        )
                        images.append(
                            {
                                "source_path": str(img_file),
                                "species": species_name,
                                "strain": strain,
                                "environment": parsed["environment"],
                                "angle": parsed["angle"],
                                "filename": img_file.name,
                            }
                        )

    return images


def analyze_data(images: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    species_strain = defaultdict(set)
    for img in images:
        species_strain[img["species"]].add(img["strain"])

    species_data = []
    for species, strains in sorted(species_strain.items()):
        species_data.append(
            {
                "species": species,
                "num_strains": len(strains),
                "strains": sorted(strains),
            }
        )
    species_df = pd.DataFrame(species_data)

    env_counts = defaultdict(int)
    for img in images:
        env_counts[img["environment"]] += 1

    env_df = pd.DataFrame(
        [
            {"environment": env, "num_samples": count}
            for env, count in sorted(env_counts.items())
        ]
    )

    return species_df, env_df


def detect_plate_circle(image):
    """Detect plate circle using Hough circle transform. Returns (cx, cy, r) or None."""
    # Resize to larger size first for better Hough detection
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(w, h),
        param1=90,
        param2=90,
        minRadius=min(w, h) // 4,
        maxRadius=min(w, h) // 2,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Return largest circle
        largest = max(circles, key=lambda c: c[2])
        return (largest[0], largest[1], largest[2])
    return None


def draw_bboxes(image, bboxes, label="colony"):
    """Draw bounding boxes on image with labels."""
    result = image.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Different color per colony
    for i, bbox in enumerate(bboxes):
        x, y, w, h = (
            bbox["xmin"],
            bbox["ymin"],
            bbox["xmax"] - bbox["xmin"],
            bbox["ymax"] - bbox["ymin"],
        )
        color = colors[i % len(colors)]
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            result, f"{label}_{i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    return result


def process_pipeline(diverse_images_path: Path, metadata_path: Path, limit: int = None):
    """Full pipeline with hierarchical structure and bbox visualization."""
    diverse_images_path = Path(diverse_images_path)
    metadata_path = Path(metadata_path)

    print("Scanning source images...")
    images = find_all_images(NEW_DATA_PATH)
    print(f"Found {len(images)} images\n")

    if limit:
        images = images[:limit]

    all_metadata = []
    count = 0

    for img_info in images:
        source_path = Path(img_info["source_path"])
        if not source_path.exists():
            print(f"Warning: {source_path} not found, skipping")
            continue

        src = cv2.imread(str(source_path))
        if src is None:
            print(f"Warning: cannot read {source_path}")
            continue

        h, w = src.shape[:2]

        # Step 1: Crop width to center so width = height (make square)
        if w > h:
            crop_left = (w - h) // 2
            centered = src[:, crop_left : crop_left + h, :]
        else:
            centered = src

        # Step 2: Resize to larger size for better Hough detection
        resized = cv2.resize(centered, (PREPROCESS_SIZE, PREPROCESS_SIZE))

        # Step 3: Detect plate circle using Hough transform
        circle = detect_plate_circle(resized)

        cropped = None
        if circle is not None:
            cx, cy, r = circle
            # Mask outside circle
            mask = np.zeros_like(resized)
            cv2.circle(mask, (cx, cy), r, (255, 255, 255), -1)
            masked = np.where(mask == 0, 0, resized)

            # Ensure boundaries are valid
            cy = max(r, min(cy, PREPROCESS_SIZE - r))
            cx = max(r, min(cx, PREPROCESS_SIZE - r))
            cropped = masked[cy - r : cy + r, cx - r : cx + r, :]
        else:
            # Fallback: just resize to square
            cropped = resized

        # Step 4: Resize cropped circle to IMG_SIZE x IMG_SIZE
        preprocessed = cv2.resize(
            cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
        )

        # Generate unique ID for this image (needed for temp file and filenames)
        img_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(source_path)).hex

        # Step 5: KMeans segmentation
        # Save preprocessed to temp file for kmeans (it expects an image path)
        temp_path = PROJECT_ROOT / f"temp_{img_id[:8]}.jpg"
        cv2.imwrite(str(temp_path), preprocessed)
        bboxes = segment_kmeans(str(temp_path))
        if temp_path.exists():
            temp_path.unlink()

        # Create hierarchical path: {species}/{environment}/
        species_dir = diverse_images_path / img_info["species"]
        env_dir = species_dir / img_info["environment"]
        env_dir.mkdir(parents=True, exist_ok=True)

        # Target filename: {Strain}_{angle}.jpg
        clean_strain = re.sub(r"[^\w\-]", "_", img_info["strain"])
        target_filename = f"{clean_strain}_{img_info['angle']}_{img_id[:8]}.jpg"
        target_path = env_dir / target_filename

        # Handle duplicate filenames
        if target_path.exists():
            target_filename = (
                f"{clean_strain}_{img_info['angle']}_{img_id[:8]}_{count}.jpg"
            )
            target_path = env_dir / target_filename

        # Save all processing steps
        step_paths = {}

        # Step 0: Original image
        original_path = (
            env_dir / f"{clean_strain}_{img_info['angle']}_{img_id[:8]}_original.jpg"
        )
        cv2.imwrite(str(original_path), src)
        step_paths["original"] = relative_to_workspace(original_path)

        # Step 1: Center-cropped to square (width = height)
        centered_path = (
            env_dir / f"{clean_strain}_{img_info['angle']}_{img_id[:8]}_centered.jpg"
        )
        cv2.imwrite(str(centered_path), centered)
        step_paths["centered"] = relative_to_workspace(centered_path)

        # Step 2: Resized to 512x512 for Hough detection
        resized_path = (
            env_dir / f"{clean_strain}_{img_info['angle']}_{img_id[:8]}_resized.jpg"
        )
        cv2.imwrite(str(resized_path), resized)
        step_paths["resized"] = relative_to_workspace(resized_path)

        # Step 3: Hough-circle-cropped image
        cropped_path = (
            env_dir / f"{clean_strain}_{img_info['angle']}_{img_id[:8]}_cropped.jpg"
        )
        cv2.imwrite(str(cropped_path), cropped)
        step_paths["cropped"] = relative_to_workspace(cropped_path)

        # Step 4: Preprocessed image (final, resized to 256x256)
        cv2.imwrite(str(target_path), preprocessed)
        step_paths["preprocessed"] = relative_to_workspace(target_path)

        # Step 4: Bboxes visualization
        bboxes_img = draw_bboxes(preprocessed, bboxes)
        bbox_path = (
            env_dir / f"{clean_strain}_{img_info['angle']}_{img_id[:8]}_bboxes.jpg"
        )
        cv2.imwrite(str(bbox_path), bboxes_img)
        step_paths["bboxes"] = relative_to_workspace(bbox_path)

        # Step 5: Segmented colonies
        segment_paths = []
        for idx, bbox in enumerate(bboxes):
            x, y, w, h = (
                bbox["xmin"],
                bbox["ymin"],
                bbox["xmax"] - bbox["xmin"],
                bbox["ymax"] - bbox["ymin"],
            )
            segment_img = preprocessed[y : y + h, x : x + w]
            segment_img = cv2.resize(segment_img, (IMG_SIZE, IMG_SIZE))
            seg_path = (
                env_dir
                / f"{clean_strain}_{img_info['angle']}_{img_id[:8]}_seg{idx}.jpg"
            )
            cv2.imwrite(str(seg_path), segment_img)
            segment_paths.append(relative_to_workspace(seg_path))
        step_paths["segments"] = segment_paths

        # Add metadata
        metadata_entry = {
            "id": img_id,
            "file_path": relative_to_workspace(target_path),
            "step_images": step_paths,
            "data": {
                "species": img_info["species"],
                "strain": img_info["strain"],
                "environment": img_info["environment"],
                "angle": img_info["angle"],
                "original_filename": img_info["filename"],
                "bboxes": bboxes,
                "num_colonies": len(bboxes),
                "segment_paths": segment_paths,
            },
        }
        all_metadata.append(metadata_entry)

        print(
            f"Processed: {img_info['species']}/{img_info['environment']}/{target_filename} -> {len(bboxes)} colonies"
        )
        count += 1

    # Save metadata
    with open(metadata_path, "w") as f:
        json.dump({"images": all_metadata}, f, indent=2)

    print("\nPipeline complete!")
    print(f"Total images processed: {len(all_metadata)}")
    print(f"Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Reorganize Dataset/new_data")
    parser.add_argument("--mode", choices=["analyze", "pipeline"], default="analyze")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--species-csv", type=str, default=None)
    parser.add_argument("--env-csv", type=str, default=None)

    args = parser.parse_args()

    print("Scanning Dataset/new_data structure...")
    images = find_all_images(NEW_DATA_PATH)
    print(f"Found {len(images)} images\n")

    if args.mode == "analyze":
        species_df, env_df = analyze_data(images)

        print("=" * 60)
        print("SPECIES AND STRAIN COUNTS")
        print("=" * 60)
        for _, row in species_df.iterrows():
            print(f"\n{row['species']}: {row['num_strains']} strains")
            print(f"  Strains: {', '.join(row['strains'])}")

        print("\n" + "=" * 60)
        print("ENVIRONMENT SAMPLE COUNTS")
        print("=" * 60)
        for _, row in env_df.iterrows():
            print(f"  {row['environment']}: {row['num_samples']} samples")

        if args.species_csv:
            species_df.to_csv(args.species_csv, index=False)
            print(f"\nSpecies CSV saved to: {args.species_csv}")
        if args.env_csv:
            env_df.to_csv(args.env_csv, index=False)
            print(f"Environment CSV saved to: {args.env_csv}")

    elif args.mode == "pipeline":
        process_pipeline(DIVERSE_IMAGES_PATH, DIVERSE_METADATA_PATH, limit=args.limit)


if __name__ == "__main__":
    main()

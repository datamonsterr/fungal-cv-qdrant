"""
Prepare test strains for threshold experiment:
1. Copy test strain segments from segmented_image/ to diverse_data/{species}/{env}/
2. Add them to diverse_data_metadata.json
3. Create diverse_retrieval_results.csv with proper known/unk classification
4. Run threshold analysis

Test strains (hold-out, train=False):
  DTO 217-D9  (neoechinulatum) — 42 images
  DTO 470-I9  (tricolor)       — 42 images
  DTO 158-D1  (melanoconidium)  — 42 images
  DTO 148-D1  (polonicum)      — 42 images
  DTO 469-I5  (aurantiogriseum) — 42 images
  DTO 469-I4  (freii)          — 42 images
  DTO 163-I2  (viridicatum)    — 42 images
  cyclopium DTO 148-C8 kept in Qdrant (Train=True, no hold-out available)

All other strains in diverse_data → unknown (train=False in Qdrant).

Usage:
    uv run python -m src.experiments.threshold.prepare_test_strains
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_ROOT, RESULTS_DIR, WORKSPACE_ROOT, relative_to_workspace  # noqa: E402

DIVERSE_DATA_ROOT = DATASET_ROOT / "diverse_data"
DIVERSE_IMAGES = DIVERSE_DATA_ROOT / "images"
DIVERSE_METADATA_PATH = DIVERSE_DATA_ROOT / "diverse_data_metadata.json"
SEGMENTED_META = DATASET_ROOT / "segmented_image_metadata.json"
STRAIN_SPLIT_CSV = DATASET_ROOT / "strain_split.csv"
OUTPUT_CSV = RESULTS_DIR / "threshold" / "diverse_retrieval_results.csv"
OUTPUT_DIR = OUTPUT_CSV.parent

# Species name mapping: DTO species name → diverse_data folder name
# (Penicillium freii → freii, etc.)
SPECIES_TO_FOLDER = {
    "Penicillium aurantiogriseum": "aurantiogriseum",
    "Penicillium cyclopium": "cyclopium",
    "Penicillium freii": "freii",
    "Penicillium melanoconidium": "melanoconidium",
    "Penicillium neoechinulatum": "neoechinulatum",
    "Penicillium polonicum": "polonicum",
    "Penicillium tricolor": "tricolor",
    "Penicillium viridicatum": "viridicatum",
}

# Environment mapping for E1 filter compatibility
ENV_ALIASES = {
    "CYA30": "CYA",  # CYA30 → CYA for E1 filtering
    "CYAS": "CYAS",
    "CREA": "CREA",
    "DG18": "DG18",
    "MEA": "MEA",
    "YES": "YES",
}

# Test strains (hold-out, NOT in Qdrant for retrieval)
TEST_STRAINS = {
    "DTO 217-D9",
    "DTO 470-I9",
    "DTO 158-D1",
    "DTO 148-D1",
    "DTO 469-I5",
    "DTO 469-I4",
    "DTO 163-I2",
}

# Environments to copy (matching diverse_data existing envs)
ENVS_TO_COPY = {"CREA", "CYA", "DG18", "MEA", "YES"}  # Skip CYA30, CYAS for now


def load_strain_split() -> dict[str, bool]:
    """Load strain → train=True/False mapping."""
    split = {}
    with open(STRAIN_SPLIT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            split[row["Strain"].strip()] = row["Train"].strip() == "True"
    return split


def load_segmented_meta() -> list:
    with open(SEGMENTED_META) as f:
        return json.load(f)


def load_diverse_metadata() -> dict:
    with open(DIVERSE_METADATA_PATH) as f:
        return json.load(f)


def prepare_test_strain_images():
    """
    Copy test strain segment images to diverse_data/images/{species}/{env}/
    with proper filenames matching the diverse_data naming convention.
    Returns list of new metadata entries.
    """
    load_strain_split()
    seg_meta = load_segmented_meta()

    # Index segmented images by (strain, env, angle, seg_idx)
    seg_index = defaultdict(list)
    for img in seg_meta:
        data = img["data"]
        strain = data["strain"]
        if strain not in TEST_STRAINS:
            continue
        env = data["environment"]
        if env not in ENVS_TO_COPY:
            continue
        # Normalize env for diverse_data
        norm_env = ENV_ALIASES.get(env, env)
        angle = data["angle"]
        seg_idx = img["id"].split("_")[-1]
        seg_index[(strain, norm_env, angle, seg_idx)].append(img)

    new_entries = []

    for (strain, env, angle, seg_idx), images in sorted(seg_index.items()):
        species_db = images[0]["data"]["specy"]  # e.g. "Penicillium neoechinulatum"
        folder = SPECIES_TO_FOLDER.get(species_db)
        if folder is None:
            print(f"WARNING: no folder mapping for {species_db}")
            continue

        # Destination path: Dataset/diverse_data/images/{folder}/{env}/{strain}_{angle}_{uuid}.jpg
        dest_dir = DIVERSE_IMAGES / folder / env
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Create unique ID from original segment path
        orig_path = images[0]["file_path"]
        unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, orig_path).hex[:8]
        dest_name = f"{strain.replace(' ', '_')}_{angle}_{unique_id}.jpg"
        dest_path = dest_dir / dest_name

        # Copy the preprocessed segment image (256x256)
        src_path = WORKSPACE_ROOT / orig_path
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
        else:
            print(f"WARNING: source not found: {src_path}")
            continue

        # Build metadata entry
        # The "preprocessed" path is what retrieval reads
        entry = {
            "id": unique_id,
            "file_path": relative_to_workspace(dest_path),
            "step_images": {
                "original": relative_to_workspace(dest_path),
                "preprocessed": relative_to_workspace(dest_path),
            },
            "data": {
                "species": folder,  # diverse_data species name
                "strain": strain,
                "environment": env,
                "angle": angle,
                "original_filename": Path(orig_path).name,
                "bboxes": [],  # no bboxes for test strain segments
                "num_colonies": 1,
                "segment_paths": [relative_to_workspace(dest_path)],
            },
        }
        new_entries.append(entry)

    print(f"Copied {len(new_entries)} test strain segment images to diverse_data/")
    return new_entries


def build_combined_metadata(existing_meta: dict, new_entries: list) -> dict:
    """Merge new test strain entries with existing diverse_data metadata."""
    existing_images = existing_meta.get("images", [])
    # Remove any existing test strain images (in case of re-run)
    existing_ids = {img["id"] for img in existing_images}
    filtered_new = [e for e in new_entries if e["id"] not in existing_ids]
    combined = existing_images + filtered_new
    return {"images": combined}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: Copying test strain images to diverse_data/")
    new_entries = prepare_test_strain_images()

    print("\nStep 2: Building combined metadata")
    existing = load_diverse_metadata()
    combined = build_combined_metadata(existing, new_entries)
    print(f"  Total images in metadata: {len(combined['images'])}")

    # Save combined metadata (for reference only — retrieval reads new_entries only)
    combined_meta_path = (
        DIVERSE_DATA_ROOT / "diverse_data_with_test_strains_metadata.json"
    )
    with open(combined_meta_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  Saved combined metadata to: {combined_meta_path}")

    print("\nStep 3: Building retrieval list (test strains = known, others = unknown)")
    retrieval_list = []
    for entry in new_entries:
        is_known = 1  # test strains = known species
        retrieval_list.append(
            {
                **entry,
                "is_known": is_known,
                "train_in_qdrant": False,  # excluded from Qdrant during retrieval
            }
        )

    # Add existing diverse_data images as unknown
    for img in existing["images"]:
        retrieval_list.append(
            {
                **img,
                "is_known": 0,  # not a test strain = unknown
                "train_in_qdrant": True,  # in Qdrant (train strains)
            }
        )

    print(f"  Total for retrieval: {len(retrieval_list)} images")
    known_count = sum(1 for r in retrieval_list if r["is_known"] == 1)
    unk_count = sum(1 for r in retrieval_list if r["is_known"] == 0)
    print(f"  Known (test strains): {known_count}, Unknown: {unk_count}")

    # Save retrieval list
    retrieval_list_path = OUTPUT_DIR / "test_strain_retrieval_list.json"
    with open(retrieval_list_path, "w") as f:
        json.dump(retrieval_list, f, indent=2)
    print(f"  Saved retrieval list to: {retrieval_list_path}")

    print("\nStep 4: Summary of test strains to retrieve:")
    from collections import Counter

    by_species = Counter(
        r["data"]["species"] for r in retrieval_list if r["is_known"] == 1
    )
    for sp, count in sorted(by_species.items()):
        print(f"  {sp}: {count} images")

    print("\nNext: Run retrieval with train=True filter in Qdrant:")
    print("  uv run python -m src.experiments.threshold.retrieve_with_train_filter")
    print("  Then: uv run python -m src.experiments.threshold.threshold_analysis")


if __name__ == "__main__":
    main()

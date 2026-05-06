import json
import os
import re
from pathlib import Path
from typing import Dict, Set

import pandas as pd

from src.config import (
    CURATED_SOURCE_DATASET_PATH,
    PREPARED_ITEMS_METADATA_PATH,
    PROJECT_ROOT,
    STRAIN_SPECIES_MAPPING_PATH,
)


def get_strain_species_from_folders(dataset_path: Path) -> Dict[str, str]:
    """
    Scan dataset folders to extract Strain and Species information.
    Folder format expected: "DTO 123-A1 Species Name"
    """
    if not dataset_path.exists():
        return {}

    strain_to_specy = {}

    for dir_name in os.listdir(dataset_path):
        dir_path = dataset_path / dir_name
        if not dir_path.is_dir():
            continue

        # Regex to extract Strain (DTO ...) and Species (Rest)
        # Pattern: Start with DTO, space, number, dash, code, space, Species Name
        match = re.match(r"(DTO\s[0-9]+-[A-Z0-9]+)\s(.+)", dir_name)
        if match:
            strain = match.group(1)
            species = match.group(2)
            strain_to_specy[strain] = species

    return strain_to_specy


def get_available_strains_from_metadata(metadata_path: Path) -> Set[str]:
    if not metadata_path.exists():
        return set()

    with open(metadata_path, "r") as f:
        data = json.load(f)

    strains = set()
    for item in data:
        strain = item.get("strain") or item.get("data", {}).get("strain")
        if strain and strain != "unknown":
            strains.add(strain)
    return strains


def generate_strain_mapping(
    source_csv_path: Path = PROJECT_ROOT / "strain_to_specy.csv",
    output_csv_path: Path = STRAIN_SPECIES_MAPPING_PATH,
    prepared_items_path: Path = PREPARED_ITEMS_METADATA_PATH,
):
    print("Generating strain mapping...")

    # 1. Scan Original Dataset for Mapping
    print(f"Scanning {CURATED_SOURCE_DATASET_PATH} for strain-species mapping...")
    folder_mapping = get_strain_species_from_folders(CURATED_SOURCE_DATASET_PATH)

    if not folder_mapping:
        print("Warning: No mapping found from dataset folders.")
        # Fallback to existing CSV if available
        if source_csv_path.exists():
            print(f"Loading from existing CSV: {source_csv_path}")
            df_master = pd.read_csv(source_csv_path)
            folder_mapping = dict(zip(df_master["Strain"], df_master["Species"]))
        else:
            print("Error: No source for mapping found.")
            return
    else:
        print(f"Found {len(folder_mapping)} strains from folder names.")

    # Convert to DataFrame
    df_filtered = pd.DataFrame(
        list(folder_mapping.items()), columns=["Strain", "Species"]
    )

    # 2. Identify Available Strains (Validation)
    # We already know they are available if we scanned folders, but let's check metadata too if exists
    available_strains = set(folder_mapping.keys())

    if prepared_items_path.exists():
        print(f"Verifying with metadata: {prepared_items_path}")
        metadata_strains = get_available_strains_from_metadata(prepared_items_path)
        if metadata_strains:
            # Intersection
            available_strains = available_strains.intersection(metadata_strains)
            df_filtered = df_filtered[df_filtered["Strain"].isin(available_strains)]
            print(f"Strains confirmed in metadata: {len(df_filtered)}")

    # 3. Assign Test Set (One strain per species)
    # Logic: Group by Species, pick the 2nd strain if available, else 1st.

    species_groups = df_filtered.groupby("Species")["Strain"].apply(list).to_dict()
    test_strains = set()

    for species, strains in species_groups.items():
        # Sort strains to ensure deterministic selection
        strains.sort()

        if len(strains) > 1:
            # If multiple strains, pick the second one as test (arbitrary but consistent)
            test_strains.add(strains[1])
        else:
            # If only one strain, we can't really test on unseen strain for this species.
            pass

    df_filtered["Test"] = df_filtered["Strain"].apply(lambda x: x in test_strains)

    # 4. Save
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_csv_path, index=False)
    print(f"Saved generated mapping to {output_csv_path}")
    print(f"Total Strains: {len(df_filtered)}")
    print(f"Test Strains: {df_filtered['Test'].sum()}")


if __name__ == "__main__":
    generate_strain_mapping()

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_strain_to_species_mapping(csv_path: Path) -> Dict[str, str]:
    """Load strain to species mapping from CSV."""
    df = pd.read_csv(csv_path)
    return dict(zip(df["Strain"], df["Species"]))


def select_test_strains(
    available_strains: List[str], strain_to_specy: Dict[str, str]
) -> Dict[str, str]:
    """
    Select one strain per species for testing.
    Logic: Take the second strain if available, else the first.
    """
    species_to_strains = defaultdict(list)

    for strain in available_strains:
        if strain in strain_to_specy:
            species = strain_to_specy[strain]
            species_to_strains[species].append(strain)

    test_strains = {}
    for species, strains in species_to_strains.items():
        if len(strains) > 1:
            test_strains[species] = strains[1]
        else:
            test_strains[species] = strains[0]

    return test_strains


def get_available_strains(metadata_path: Path) -> List[str]:
    with open(metadata_path, "r") as f:
        metadata_list = json.load(f)

    strains = set()
    for item in metadata_list:
        if "strain" in item:  # Handle flat structure
            strains.add(item["strain"])
        elif "data" in item and "strain" in item["data"]:  # Handle nested structure
            strains.add(item["data"]["strain"])

    return sorted(list(strains))

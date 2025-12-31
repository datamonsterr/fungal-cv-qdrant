import os
import json
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set

from src.config import (
    STRAIN_SPECIES_MAPPING_PATH,
    SEGMENTED_METADATA_PATH,
    ORIGINAL_DATASET_PATH,
    PROJECT_ROOT
)

def get_available_strains_from_metadata(metadata_path: Path) -> Set[str]:
    if not metadata_path.exists():
        return set()
    
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    strains = set()
    for item in data:
        # Handle both flat and nested structure
        strain = item.get('data', item).get('strain')
        if strain and strain != "unknown":
            strains.add(strain)
    return strains

def get_available_strains_from_folders(dataset_path: Path) -> Set[str]:
    if not dataset_path.exists():
        return set()
    
    strains = set()
    # Assuming folders might contain strain names or files contain them
    # Based on reformat_dataset.py, it iterates folders and then files
    # Filename regex: (DTO\s[0-9]+-[A-Z0-9]+)
    
    for dir_name in os.listdir(dataset_path):
        dir_path = dataset_path / dir_name
        if not dir_path.is_dir():
            continue
            
        for filename in os.listdir(dir_path):
            match = re.match(r'(DTO\s[0-9]+-[A-Z0-9]+)', filename)
            if match:
                strains.add(match.group(1))
                
    return strains

def generate_strain_mapping(
    source_csv_path: Path = PROJECT_ROOT / "strain_to_specy.csv",
    output_csv_path: Path = STRAIN_SPECIES_MAPPING_PATH
):
    print(f"Generating strain mapping...")
    print(f"Source CSV: {source_csv_path}")
    print(f"Output CSV: {output_csv_path}")

    # 1. Load Master Mapping
    if not source_csv_path.exists():
        print(f"Error: Source CSV {source_csv_path} not found.")
        return

    df_master = pd.read_csv(source_csv_path)
    # Ensure columns exist
    if 'Strain' not in df_master.columns or 'Species' not in df_master.columns:
        print("Error: Source CSV must have 'Strain' and 'Species' columns.")
        return

    # 2. Identify Available Strains
    available_strains = set()
    
    # Try metadata first
    if SEGMENTED_METADATA_PATH.exists():
        print(f"Reading strains from metadata: {SEGMENTED_METADATA_PATH}")
        available_strains = get_available_strains_from_metadata(SEGMENTED_METADATA_PATH)
    
    # If no metadata or empty, try original folders
    if not available_strains and ORIGINAL_DATASET_PATH.exists():
        print(f"Reading strains from original dataset: {ORIGINAL_DATASET_PATH}")
        available_strains = get_available_strains_from_folders(ORIGINAL_DATASET_PATH)
        
    if not available_strains:
        print("Warning: No available strains found in metadata or dataset folders.")
        print("Using all strains from master CSV.")
        available_strains = set(df_master['Strain'].unique())
    else:
        print(f"Found {len(available_strains)} unique strains in dataset.")

    # 3. Filter Master DataFrame
    df_filtered = df_master[df_master['Strain'].isin(available_strains)].copy()
    
    # --- Reporting Logic (Merged from list_strains.py) ---
    csv_strains = set(df_master['Strain'].unique())
    in_both = available_strains & csv_strains
    only_in_dataset = available_strains - csv_strains
    only_in_csv = csv_strains - available_strains
    
    print("="*80)
    print("STRAIN AVAILABILITY REPORT")
    print("="*80)
    
    print(f"\nTotal strains in dataset: {len(available_strains)}")
    print(f"Total strains in Master CSV: {len(csv_strains)}")
    print(f"Strains in both: {len(in_both)}")
    print(f"Only in dataset (Missing from CSV): {len(only_in_dataset)}")
    print(f"Only in CSV (Missing from Dataset): {len(only_in_csv)}")
    
    if only_in_csv:
        print(f"\n{'='*80}")
        print(f"⚠ STRAINS IN CSV BUT NOT IN DATASET ({len(only_in_csv)} strains)")
        print(f"{'='*80}")
        print("These strains cannot be used for training/prediction:")
        # Get species for these if possible
        strain_to_specy_master = dict(zip(df_master['Strain'], df_master['Species']))
        for strain in sorted(only_in_csv):
            species = strain_to_specy_master.get(strain, "Unknown")
            print(f"  {strain:<20} {species}")
    
    if only_in_dataset:
        print(f"\n{'='*80}")
        print(f"ℹ STRAINS IN DATASET BUT NOT IN CSV ({len(only_in_dataset)} strains)")
        print(f"{'='*80}")
        print("These strains have no ground truth species in Master CSV:")
        for strain in sorted(only_in_dataset):
            print(f"  {strain}")
            
    if in_both:
        print(f"\n{'='*80}")
        print("SPECIES DISTRIBUTION (Available Strains)")
        print(f"{'='*80}")
        
        species_count = defaultdict(int)
        strain_to_specy_master = dict(zip(df_master['Strain'], df_master['Species']))
        
        for strain in in_both:
            species = strain_to_specy_master.get(strain, "Unknown")
            species_count[species] += 1
        
        print(f"\n{'Species':<40} {'Count'}")
        print("-"*80)
        for species in sorted(species_count.keys()):
            count = species_count[species]
            print(f"{species:<40} {count}")
        
        print(f"\nTotal species: {len(species_count)}")
    # -----------------------------------------------------

    if df_filtered.empty:
        print("Warning: No intersection between master CSV and available strains.")
        return

    # 4. Assign Test Set (One strain per species)
    # Logic: Group by Species, pick the 2nd strain if available, else 1st.
    
    species_groups = df_filtered.groupby('Species')['Strain'].apply(list).to_dict()
    test_strains = set()
    
    for species, strains in species_groups.items():
        # Sort to ensure deterministic selection
        strains.sort()
        
        if len(strains) > 1:
            test_strains.add(strains[1]) # Pick 2nd
        else:
            test_strains.add(strains[0]) # Pick 1st if only 1
            
    df_filtered['Test'] = df_filtered['Strain'].apply(lambda x: x in test_strains)
    
    # 5. Save
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_csv_path, index=False)
    print(f"Saved generated mapping to {output_csv_path}")
    print(f"Total Strains: {len(df_filtered)}")
    print(f"Test Strains: {df_filtered['Test'].sum()}")

if __name__ == "__main__":
    generate_strain_mapping()

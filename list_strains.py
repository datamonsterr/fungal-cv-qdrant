"""
Utility to list available strains in the Qdrant database.
"""
from qdrant_client import QdrantClient
from prediction import get_available_strains, load_strain_to_species_mapping


def main():
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features"
    
    # Get strains from database
    db_strains = set(get_available_strains(client, collection_name))
    
    # Get strains from CSV
    csv_mapping = load_strain_to_species_mapping()
    csv_strains = set(csv_mapping.keys())
    
    # Find overlaps and differences
    in_both = db_strains & csv_strains
    only_in_db = db_strains - csv_strains
    only_in_csv = csv_strains - db_strains
    
    print("="*80)
    print("STRAIN AVAILABILITY REPORT")
    print("="*80)
    
    print(f"\nTotal strains in database: {len(db_strains)}")
    print(f"Total strains in CSV: {len(csv_strains)}")
    print(f"Strains in both: {len(in_both)}")
    print(f"Only in database: {len(only_in_db)}")
    print(f"Only in CSV: {len(only_in_csv)}")
    
    if in_both:
        print(f"\n{'='*80}")
        print(f"STRAINS AVAILABLE FOR PREDICTION ({len(in_both)} strains)")
        print(f"{'='*80}")
        print(f"\n{'Strain':<20} {'Species'}")
        print("-"*80)
        for strain in sorted(in_both):
            species = csv_mapping[strain]
            print(f"{strain:<20} {species}")
    
    if only_in_csv:
        print(f"\n{'='*80}")
        print(f"⚠ STRAINS IN CSV BUT NOT IN DATABASE ({len(only_in_csv)} strains)")
        print(f"{'='*80}")
        print("These strains cannot be used for prediction:")
        for strain in sorted(only_in_csv):
            species = csv_mapping[strain]
            print(f"  {strain:<20} {species}")
    
    if only_in_db:
        print(f"\n{'='*80}")
        print(f"ℹ STRAINS IN DATABASE BUT NOT IN CSV ({len(only_in_db)} strains)")
        print(f"{'='*80}")
        print("These strains have no ground truth:")
        for strain in sorted(only_in_db):
            print(f"  {strain}")
    
    # Group by species
    if in_both:
        print(f"\n{'='*80}")
        print("SPECIES DISTRIBUTION")
        print(f"{'='*80}")
        
        species_count = {}
        for strain in in_both:
            species = csv_mapping[strain]
            species_count[species] = species_count.get(species, 0) + 1
        
        print(f"\n{'Species':<40} {'Count'}")
        print("-"*80)
        for species in sorted(species_count.keys()):
            count = species_count[species]
            print(f"{species:<40} {count}")
        
        print(f"\nTotal species: {len(species_count)}")


if __name__ == "__main__":
    main()

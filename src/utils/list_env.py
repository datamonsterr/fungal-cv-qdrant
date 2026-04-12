"""
Utility to list available environments in the Qdrant database.
"""

from collections import defaultdict
from typing import Dict, List

from qdrant_client import QdrantClient


def get_environment_list(client: QdrantClient, collection_name: str) -> List[str]:
    """
    Get a simple list of unique environment names from the collection.

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection

    Returns:
        Sorted list of unique environment names
    """
    environments = set()
    offset = None

    while True:
        result = client.scroll(
            collection_name=collection_name, limit=100, offset=offset, with_payload=True
        )

        points, next_offset = result

        for point in points:
            environment = point.payload.get("environment", "unknown")  # type: ignore
            if environment and environment != "unknown":
                environments.add(environment)

        if next_offset is None:
            break
        offset = next_offset

    return sorted(list(environments))


def get_available_environments(
    client: QdrantClient, collection_name: str
) -> Dict[str, int]:
    """
    Get all unique environments and their sample counts from the collection.

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection

    Returns:
        Dictionary mapping environment names to sample counts
    """
    env_counts = defaultdict(int)
    offset = None

    while True:
        result = client.scroll(
            collection_name=collection_name, limit=100, offset=offset, with_payload=True
        )

        points, next_offset = result

        for point in points:
            environment = point.payload.get("environment", "unknown")  # type: ignore
            env_counts[environment] += 1

        if next_offset is None:
            break
        offset = next_offset

    return dict(env_counts)


def get_environments_by_strain(
    client: QdrantClient, collection_name: str
) -> Dict[str, Dict[str, int]]:
    """
    Get environments grouped by strain.

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection

    Returns:
        Dictionary mapping strain to environment counts
    """
    strain_env_counts = defaultdict(lambda: defaultdict(int))
    offset = None

    while True:
        result = client.scroll(
            collection_name=collection_name, limit=100, offset=offset, with_payload=True
        )

        points, next_offset = result

        for point in points:
            strain = point.payload.get("strain", "unknown")  # type: ignore
            environment = point.payload.get("environment", "unknown")  # type: ignore
            strain_env_counts[strain][environment] += 1

        if next_offset is None:
            break
        offset = next_offset

    return {k: dict(v) for k, v in strain_env_counts.items()}


def get_environments_by_species(
    client: QdrantClient, collection_name: str
) -> Dict[str, Dict[str, int]]:
    """
    Get environments grouped by species.

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection

    Returns:
        Dictionary mapping species to environment counts
    """
    species_env_counts = defaultdict(lambda: defaultdict(int))
    offset = None

    while True:
        result = client.scroll(
            collection_name=collection_name, limit=100, offset=offset, with_payload=True
        )

        points, next_offset = result

        for point in points:
            species = point.payload.get("specy", "unknown")  # type: ignore
            environment = point.payload.get("environment", "unknown")  # type: ignore
            species_env_counts[species][environment] += 1

        if next_offset is None:
            break
        offset = next_offset

    return {k: dict(v) for k, v in species_env_counts.items()}


def main():
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features_all"

    print("=" * 80)
    print("ENVIRONMENT AVAILABILITY REPORT")
    print("=" * 80)
    print(f"\nCollection: {collection_name}")

    # Get total collection info
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"Total samples in collection: {collection_info.points_count}")
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return

    # Get environment counts
    print("\nFetching environment data...")
    env_counts = get_available_environments(client, collection_name)

    # Overall environment statistics
    print(f"\n{'='*80}")
    print("OVERALL ENVIRONMENT DISTRIBUTION")
    print(f"{'='*80}")
    print(f"\nTotal unique environments: {len(env_counts)}")
    print(f"\n{'Environment':<30} {'Sample Count':<15} {'Percentage'}")
    print("-" * 80)

    total_samples = sum(env_counts.values())
    for env in sorted(env_counts.keys()):
        count = env_counts[env]
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"{env:<30} {count:<15} {percentage:>6.2f}%")

    print("-" * 80)
    print(f"{'Total':<30} {total_samples:<15} 100.00%")

    # Environments by strain
    print(f"\n{'='*80}")
    print("ENVIRONMENTS BY STRAIN")
    print(f"{'='*80}")
    print("\nFetching strain-environment data...")
    strain_env_counts = get_environments_by_strain(client, collection_name)

    # Show summary: how many strains have samples in each environment
    env_strain_counts = defaultdict(int)
    for strain, env_dict in strain_env_counts.items():
        for env in env_dict.keys():
            env_strain_counts[env] += 1

    print(f"\n{'Environment':<30} {'Number of Strains'}")
    print("-" * 80)
    for env in sorted(env_strain_counts.keys()):
        print(f"{env:<30} {env_strain_counts[env]}")

    # Show detailed breakdown for each strain
    print(f"\n{'='*80}")
    print("DETAILED STRAIN-ENVIRONMENT BREAKDOWN")
    print(f"{'='*80}")

    for strain in sorted(strain_env_counts.keys()):
        env_dict = strain_env_counts[strain]
        total_for_strain = sum(env_dict.values())
        print(f"\nStrain: {strain} (Total: {total_for_strain} samples)")
        print("-" * 60)
        for env in sorted(env_dict.keys()):
            count = env_dict[env]
            print(f"  {env:<28} {count:>5} samples")

    # Environments by species
    print(f"\n{'='*80}")
    print("ENVIRONMENTS BY SPECIES")
    print(f"{'='*80}")
    print("\nFetching species-environment data...")
    species_env_counts = get_environments_by_species(client, collection_name)

    # Show summary: how many species have samples in each environment
    env_species_counts = defaultdict(int)
    for species, env_dict in species_env_counts.items():
        for env in env_dict.keys():
            env_species_counts[env] += 1

    print(f"\n{'Environment':<30} {'Number of Species'}")
    print("-" * 80)
    for env in sorted(env_species_counts.keys()):
        print(f"{env:<30} {env_species_counts[env]}")

    # Show detailed breakdown for each species
    print(f"\n{'='*80}")
    print("DETAILED SPECIES-ENVIRONMENT BREAKDOWN")
    print(f"{'='*80}")

    for species in sorted(species_env_counts.keys()):
        env_dict = species_env_counts[species]
        total_for_species = sum(env_dict.values())
        print(f"\nSpecies: {species} (Total: {total_for_species} samples)")
        print("-" * 60)
        for env in sorted(env_dict.keys()):
            count = env_dict[env]
            print(f"  {env:<28} {count:>5} samples")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"\nTotal environments: {len(env_counts)}")
    print(f"Total strains: {len(strain_env_counts)}")
    print(f"Total species: {len(species_env_counts)}")
    print(f"Total samples: {total_samples}")

    # Environment coverage
    print("\nEnvironment Coverage:")
    strains_with_all_envs = sum(
        1 for env_dict in strain_env_counts.values() if len(env_dict) == len(env_counts)
    )
    print(f"  Strains with samples in all environments: {strains_with_all_envs}")

    species_with_all_envs = sum(
        1
        for env_dict in species_env_counts.values()
        if len(env_dict) == len(env_counts)
    )
    print(f"  Species with samples in all environments: {species_with_all_envs}")


if __name__ == "__main__":
    main()

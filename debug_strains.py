"""
Debug script to check strain names in Qdrant.
"""
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
collection_name = "myco_fungi_features"

# Get a few sample points to see strain format
sample = client.scroll(
    collection_name=collection_name,
    limit=10,
    with_payload=True
)

print("Sample strain names in Qdrant:")
print("="*60)

seen_strains = set()
for point in sample[0]:
    strain = point.payload.get('strain', 'unknown')
    if strain not in seen_strains:
        print(f"  '{strain}'")
        seen_strains.add(strain)

print("\n" + "="*60)
print("Checking for 'DTO 003-E9':")

# Try exact match
result = client.scroll(
    collection_name=collection_name,
    scroll_filter={
        "must": [
            {"key": "strain", "match": {"value": "DTO 003-E9"}}
        ]
    },
    limit=1,
    with_payload=True
)

if result[0]:
    print(f"  ✓ Found with exact match")
else:
    print(f"  ✗ Not found with exact match")
    
# Try without space
result2 = client.scroll(
    collection_name=collection_name,
    scroll_filter={
        "must": [
            {"key": "strain", "match": {"value": "DTO003-E9"}}
        ]
    },
    limit=1,
    with_payload=True
)

if result2[0]:
    print(f"  ✓ Found as 'DTO003-E9' (no space)")
else:
    print(f"  ✗ Not found as 'DTO003-E9'")

print("\nAll unique strains in collection:")
print("="*60)

# Get all unique strains
all_strains = set()
offset = None
while True:
    result = client.scroll(
        collection_name=collection_name,
        limit=100,
        offset=offset,
        with_payload=True
    )
    
    points, next_offset = result
    
    for point in points:
        strain = point.payload.get('strain')
        if strain:
            all_strains.add(strain)
    
    if next_offset is None:
        break
    offset = next_offset

for strain in sorted(all_strains):
    print(f"  '{strain}'")

print(f"\nTotal unique strains: {len(all_strains)}")

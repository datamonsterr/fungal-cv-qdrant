"""
Test script to verify collect_testset creates 6 unique test sets.
"""
from collections import defaultdict

# Simulate strain images with proper metadata
def create_test_images():
    """Create mock images matching the hierarchical dataset structure."""
    images = []
    environments = ['CREA', 'CYA', 'CYA30', 'CYAS', 'DG18', 'MEA', 'YES']
    segments = [0, 1, 2]
    angles = ['ob', 'rev']
    
    img_id = 0
    for env in environments:
        for seg in segments:
            for angle in angles:
                images.append({
                    'id': f'img_{img_id}',
                    'strain': 'DTO 158-D1',
                    'environment': env,
                    'segment_index': seg,
                    'angle': angle,
                    'filename': f'DTO_158-D1_{env}_{angle}_seg{seg}.jpg'
                })
                img_id += 1
    
    return images


def collect_testset_fixed(strain_images, exclude_env=None):
    """Fixed version of collect_testset for E1/E2/E4 strategies."""
    # Group images by environment, segment_index, and angle
    env_segment_angle_images = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for img in strain_images:
        env = img.get('environment', 'unknown')
        # Skip excluded environment for E4 strategy
        if exclude_env and env == exclude_env:
            continue
        segment_idx = img.get('segment_index', 0)
        angle = img.get('angle', 'unknown')
        env_segment_angle_images[env][segment_idx][angle].append(img)
    
    # Create test sets based on segment_index and angle combinations
    test_sets = []
    
    # Define test set configurations: (segment_index, preferred_angle)
    test_configs = [
        (0, 'ob'), (0, 'rev'),
        (1, 'ob'), (1, 'rev'),
        (2, 'ob'), (2, 'rev'),
    ]
    
    for segment_idx, preferred_angle in test_configs:
        test_set = []
        
        # For each environment, pick the image with this segment_index and angle
        for env in sorted(env_segment_angle_images.keys()):
            segment_images = env_segment_angle_images[env]
            
            # Try to get image with current segment_index and preferred angle
            img_selected = None
            if segment_idx in segment_images:
                # Try preferred angle first
                angle_variations = {
                    'ob': ['ob', 'obverse'],
                    'rev': ['rev', 'reverse']
                }
                for angle_var in angle_variations.get(preferred_angle, [preferred_angle]):
                    if angle_var in segment_images[segment_idx] and segment_images[segment_idx][angle_var]:
                        img_selected = segment_images[segment_idx][angle_var][0]
                        break
                
                # If preferred angle not found, use any angle from this segment
                if img_selected is None:
                    for angle in segment_images[segment_idx]:
                        if segment_images[segment_idx][angle]:
                            img_selected = segment_images[segment_idx][angle][0]
                            break
            
            # Fallback: use any available image from this environment
            if img_selected is None:
                for seg_idx in sorted(segment_images.keys()):
                    for angle in segment_images[seg_idx]:
                        if segment_images[seg_idx][angle]:
                            img_selected = segment_images[seg_idx][angle][0]
                            break
                    if img_selected is not None:
                        break
            
            if img_selected is not None:
                test_set.append(img_selected)
        
        # Only add test set if it has images from all environments
        if test_set and len(test_set) == len(env_segment_angle_images):
            test_sets.append(test_set)
    
    return test_sets


def main():
    print("="*80)
    print("Testing collect_testset Fix")
    print("="*80)
    
    # Create test images
    images = create_test_images()
    print(f"\nCreated {len(images)} test images")
    print(f"  Environments: 7 (CREA, CYA, CYA30, CYAS, DG18, MEA, YES)")
    print(f"  Segments per env: 3 (0, 1, 2)")
    print(f"  Angles per segment: 2 (ob, rev)")
    print(f"  Total per env: 6 images")
    
    # Test E2 strategy (all environments)
    print("\n" + "-"*80)
    print("Testing E2 Strategy (all environments)")
    print("-"*80)
    test_sets = collect_testset_fixed(images)
    
    print(f"\nGenerated {len(test_sets)} test sets")
    
    for idx, test_set in enumerate(test_sets, 1):
        print(f"\nTest Set {idx}:")
        print(f"  Size: {len(test_set)} images")
        
        # Check uniqueness
        filenames = [img['filename'] for img in test_set]
        unique_filenames = set(filenames)
        print(f"  Unique images: {len(unique_filenames)}")
        
        # Show segment and angle distribution
        segments = [img['segment_index'] for img in test_set]
        angles = [img['angle'] for img in test_set]
        print(f"  Segments used: {set(segments)}")
        print(f"  Angles used: {set(angles)}")
        
        # Show first few filenames
        print(f"  Sample filenames: {filenames[:3]}")
    
    # Check for duplicates across test sets
    print("\n" + "-"*80)
    print("Checking for Duplicates Across Test Sets")
    print("-"*80)
    
    all_used_images = []
    for test_set in test_sets:
        all_used_images.extend([img['id'] for img in test_set])
    
    print(f"\nTotal image uses: {len(all_used_images)}")
    print(f"Unique images used: {len(set(all_used_images))}")
    
    # Count how many times each image is used
    from collections import Counter
    usage_counts = Counter(all_used_images)
    max_usage = max(usage_counts.values())
    
    if max_usage > 1:
        print(f"\n⚠️  WARNING: Some images used {max_usage} times!")
        duplicated = [img_id for img_id, count in usage_counts.items() if count > 1]
        print(f"  Duplicated images: {len(duplicated)}")
        print(f"  Examples: {duplicated[:5]}")
    else:
        print("\n✓ SUCCESS: Each image used exactly once across all test sets!")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

"""
Test script to verify the prediction system is working correctly.
Run this after installation to ensure everything is set up properly.
"""
import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    
    required_modules = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('qdrant_client', 'qdrant-client'),
        ('sklearn', 'scikit-learn'),
        ('PIL', 'pillow'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
    ]
    
    missing = []
    for module, package in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: uv pip install " + ' '.join(missing))
        return False
    
    print("  All dependencies installed!\n")
    return True


def check_qdrant_connection():
    """Check if Qdrant is running and accessible."""
    print("Checking Qdrant connection...")
    
    try:
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print(f"  ✓ Connected to Qdrant")
        print(f"  ✓ Found {len(collections.collections)} collections")
        
        # Check for our collection
        collection_names = [c.name for c in collections.collections]
        if "myco_fungi_features" in collection_names:
            print(f"  ✓ Found 'myco_fungi_features' collection")
            
            # Get collection info
            info = client.get_collection("myco_fungi_features")
            print(f"  ✓ Collection has {info.points_count} points")
            return True, client
        else:
            print(f"  ✗ 'myco_fungi_features' collection not found")
            print(f"  Available collections: {', '.join(collection_names)}")
            return False, None
            
    except Exception as e:
        print(f"  ✗ Cannot connect to Qdrant: {e}")
        print("  Make sure Qdrant is running (docker ps)")
        return False, None


def check_files():
    """Check if required files exist."""
    print("\nChecking required files...")
    
    required_files = [
        ("../Dataset/strain_to_specy.csv", "Strain-to-species mapping"),
        ("../Dataset/segmented_image", "Segmented images directory"),
        ("query_utils.py", "Query utilities"),
        ("feature_extractors.py", "Feature extractors"),
        ("prediction.py", "Prediction module"),
    ]
    
    all_exist = True
    for path, description in required_files:
        if os.path.exists(path):
            if os.path.isdir(path):
                num_files = len([f for f in os.listdir(path) if f.endswith('.jpg')])
                print(f"  ✓ {description}: {path} ({num_files} images)")
            else:
                print(f"  ✓ {description}: {path}")
        else:
            print(f"  ✗ {description}: {path} (not found)")
            all_exist = False
    
    if not all_exist:
        print("\n  Some files are missing. Check your directory structure.")
        return False
    
    print("  All required files present!\n")
    return True


def test_prediction(client):
    """Run a simple prediction test."""
    print("Running test prediction...")
    
    try:
        from feature_extractors import ResNet50Extractor
        from prediction import predict, load_strain_to_species_mapping, get_all_images_for_strain
        
        # Load strains from CSV
        strain_to_specy = load_strain_to_species_mapping()
        
        # Find a strain that exists in the database
        test_strain = None
        for strain in strain_to_specy.keys():
            images = get_all_images_for_strain(
                client=client,
                collection_name="myco_fungi_features",
                strain=strain
            )
            if images:
                test_strain = strain
                break
        
        if not test_strain:
            print("  ✗ No strains from CSV found in database")
            return False
        
        print(f"  Testing with strain: {test_strain}")
        print(f"  Ground truth: {strain_to_specy[test_strain]}")
        
        # Run prediction with minimal output
        result = predict(
            client=client,
            collection_name="myco_fungi_features",
            strain=test_strain,
            feature_extractor=ResNet50Extractor(),
            k=3,  # Small k for quick test
            without_siblings=True,
            environment=None
        )
        
        print(f"\n  ✓ Prediction successful!")
        print(f"  Predicted: {result['predicted_specy']}")
        print(f"  Confidence: {result['predicted_confidence']:.4f}")
        print(f"  Correct: {result['correct']}")
        print(f"  Query images: {result['num_query_images']}")
        print(f"  Total neighbors: {result['num_neighbors_total']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("PREDICTION SYSTEM TEST")
    print("="*80 + "\n")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Test failed: Missing dependencies")
        print("Install with: uv pip install -r requirements.txt")
        return False
    
    # Check Qdrant
    qdrant_ok, client = check_qdrant_connection()
    if not qdrant_ok:
        print("\n❌ Test failed: Qdrant not accessible")
        return False
    
    # Check files
    if not check_files():
        print("\n❌ Test failed: Missing files")
        return False
    
    # Run prediction test
    print("\n" + "="*80)
    if test_prediction(client):
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nThe prediction system is ready to use.")
        print("Try: python prediction.py")
        print("Or:  python example_prediction.py")
        return True
    else:
        print("\n❌ Test failed: Prediction error")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

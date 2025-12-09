"""
Quick test script for visualize_prediction.py
Tests the visualization without running full evaluation.
"""
import json
import os


def create_mock_prediction_result():
    """
    Create a mock prediction result for testing visualization.
    """
    return {
        'strain': 'TEST-STRAIN-001',
        'ground_truth': 'Penicillium chrysogenum',
        'predicted_specy': 'Penicillium roqueforti',
        'predicted_confidence': 0.756,
        'correct': False,
        'num_query_images': 3,
        'num_neighbors_total': 21,
        'raw_results': [
            {
                'query_image_id': '5ad9f46612de5e1cb6e7cb9643f060f2',
                'query_environment': 'CYA',
                'neighbors': [
                    {
                        'image_id': '5d54ad461ecd52969c1f7927c0bc617b',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 456-B2',
                        'environment': 'CYA',
                        'score': 0.924,
                        'distance': 0.076,
                    },
                    {
                        'image_id': '67c41e8e50df59b186d1bac0a05169b9',
                        'specy': 'Penicillium chrysogenum',
                        'strain': 'DTO 789-C3',
                        'environment': 'CYA',
                        'score': 0.887,
                        'distance': 0.113,
                    },
                    {
                        'image_id': '70e9c2927d405dddac31e2e5f7fb790a',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 012-D4',
                        'environment': 'CYA',
                        'score': 0.865,
                        'distance': 0.135,
                    },
                    {
                        'image_id': '944c9a0995b557b7b068511197a9413f',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 345-E5',
                        'environment': 'CYA',
                        'score': 0.843,
                        'distance': 0.157,
                    },
                    {
                        'image_id': 'a665166bdba9545193824b6fbce5ad64',
                        'specy': 'Penicillium chrysogenum',
                        'strain': 'DTO 678-F6',
                        'environment': 'CYA',
                        'score': 0.821,
                        'distance': 0.179,
                    },
                    {
                        'image_id': 'ce077b6fe88e5147b5725a4f9130a21b',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 901-G7',
                        'environment': 'CYA',
                        'score': 0.798,
                        'distance': 0.202,
                    },
                    {
                        'image_id': 'e855238e954758daa0fd5db2225c93e3',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 234-H8',
                        'environment': 'CYA',
                        'score': 0.776,
                        'distance': 0.224,
                    },
                ]
            },
            {
                'query_image_id': '5ad9f46612de5e1cb6e7cb9643f060f3',
                'query_environment': 'MEA',
                'neighbors': [
                    {
                        'image_id': '5d54ad461ecd52969c1f7927c0bc617c',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 456-B2',
                        'environment': 'MEA',
                        'score': 0.912,
                        'distance': 0.088,
                    },
                    {
                        'image_id': '67c41e8e50df59b186d1bac0a05169c0',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 789-C3',
                        'environment': 'MEA',
                        'score': 0.876,
                        'distance': 0.124,
                    },
                    {
                        'image_id': '70e9c2927d405dddac31e2e5f7fb790b',
                        'specy': 'Penicillium chrysogenum',
                        'strain': 'DTO 012-D4',
                        'environment': 'MEA',
                        'score': 0.854,
                        'distance': 0.146,
                    },
                    {
                        'image_id': '944c9a0995b557b7b068511197a9414a',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 345-E5',
                        'environment': 'MEA',
                        'score': 0.832,
                        'distance': 0.168,
                    },
                    {
                        'image_id': 'a665166bdba9545193824b6fbce5ad65',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 678-F6',
                        'environment': 'MEA',
                        'score': 0.809,
                        'distance': 0.191,
                    },
                    {
                        'image_id': 'ce077b6fe88e5147b5725a4f9130a21c',
                        'specy': 'Penicillium chrysogenum',
                        'strain': 'DTO 901-G7',
                        'environment': 'MEA',
                        'score': 0.787,
                        'distance': 0.213,
                    },
                    {
                        'image_id': 'e855238e954758daa0fd5db2225c93e4',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 234-H8',
                        'environment': 'MEA',
                        'score': 0.765,
                        'distance': 0.235,
                    },
                ]
            },
            {
                'query_image_id': '5ad9f46612de5e1cb6e7cb9643f060f4',
                'query_environment': 'YES',
                'neighbors': [
                    {
                        'image_id': '5d54ad461ecd52969c1f7927c0bc617d',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 456-B2',
                        'environment': 'YES',
                        'score': 0.901,
                        'distance': 0.099,
                    },
                    {
                        'image_id': '67c41e8e50df59b186d1bac0a05169c1',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 789-C3',
                        'environment': 'YES',
                        'score': 0.865,
                        'distance': 0.135,
                    },
                    {
                        'image_id': '70e9c2927d405dddac31e2e5f7fb790c',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 012-D4',
                        'environment': 'YES',
                        'score': 0.843,
                        'distance': 0.157,
                    },
                    {
                        'image_id': '944c9a0995b557b7b068511197a9414b',
                        'specy': 'Penicillium chrysogenum',
                        'strain': 'DTO 345-E5',
                        'environment': 'YES',
                        'score': 0.821,
                        'distance': 0.179,
                    },
                    {
                        'image_id': 'a665166bdba9545193824b6fbce5ad66',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 678-F6',
                        'environment': 'YES',
                        'score': 0.798,
                        'distance': 0.202,
                    },
                    {
                        'image_id': 'ce077b6fe88e5147b5725a4f9130a21d',
                        'specy': 'Penicillium roqueforti',
                        'strain': 'DTO 901-G7',
                        'environment': 'YES',
                        'score': 0.776,
                        'distance': 0.224,
                    },
                    {
                        'image_id': 'e855238e954758daa0fd5db2225c93e5',
                        'specy': 'Penicillium chrysogenum',
                        'strain': 'DTO 234-H8',
                        'environment': 'YES',
                        'score': 0.754,
                        'distance': 0.246,
                    },
                ]
            },
        ],
        'aggregated_results': [
            {'specy': 'Penicillium roqueforti', 'score': 17.543},
            {'specy': 'Penicillium chrysogenum', 'score': 5.321},
        ],
        'feature_extractor': 'resnet50',
        'k': 7,
        'min_samples': None,
        'without_siblings': True,
        'environment': None,  # E1 strategy
        'strategy': 'avg',
        'timestamp': '2025-11-27T10:30:00'
    }


def test_visualization():
    """
    Test the visualization function with mock data.
    """
    from visualize_prediction import visualize_prediction_by_environment
    
    print("Creating mock prediction result...")
    mock_result = create_mock_prediction_result()
    
    # Use examples directory which has sample images
    segmented_image_dir = "./examples"
    output_path = "./test_visualization.jpg"
    
    print(f"Testing visualization...")
    print(f"  - Ground truth: {mock_result['ground_truth']}")
    print(f"  - Predicted: {mock_result['predicted_specy']}")
    print(f"  - Correct: {mock_result['correct']}")
    print(f"  - Strategy: E1 (Same Environment)")
    print(f"  - Environments: {len(mock_result['raw_results'])}")
    
    try:
        visualize_prediction_by_environment(
            prediction_result=mock_result,
            segmented_image_dir=segmented_image_dir,
            output_path=output_path,
            k=7
        )
        print(f"\n✓ Test passed! Visualization saved to: {output_path}")
        print("\nVisualization features:")
        print("  - Query image in first column (red border = false prediction)")
        print("  - 7 neighbors per environment")
        print("  - Green borders = species matches ground truth")
        print("  - Red borders = species doesn't match ground truth")
        print("  - Metadata displayed for each image")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_visualization()

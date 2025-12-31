import os
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import numpy as np

from src.config import SEGMENTED_IMAGE_DIR, RESULTS_DIR

def visualize_prediction_result(
    query_image_path: str,
    neighbors: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    ground_truth_species: Optional[str] = None,
    predicted_species: Optional[str] = None,
    show_plot: bool = False
) -> None:
    """
    Visualize a query image and its nearest neighbors.
    """
    query_img = cv2.imread(query_image_path)
    if query_img is None:
        print(f"Error: Could not read query image {query_image_path}")
        return
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    num_neighbors = len(neighbors)
    cols = num_neighbors + 1
    
    plt.figure(figsize=(4 * cols, 5))
    
    # Plot query image
    plt.subplot(1, cols, 1)
    plt.imshow(query_img)
    title = "Query"
    if ground_truth_species:
        title += f"\n(True: {ground_truth_species})"
    if predicted_species:
        title += f"\n(Pred: {predicted_species})"
    plt.title(title)
    plt.axis('off')
    
    # Plot neighbors
    for i, neighbor in enumerate(neighbors):
        neighbor_id = neighbor.get('image_id') or neighbor.get('id')
        score = neighbor.get('score')
        specy = neighbor.get('specy')
        strain = neighbor.get('strain')
        
        # Construct path to neighbor image
        # Assuming neighbor images are in the same segmented directory
        # If not, we might need to fetch them or know their path
        neighbor_path = os.path.join(str(SEGMENTED_IMAGE_DIR), f"{neighbor_id}.jpg")
        
        if not os.path.exists(neighbor_path):
            # Try searching recursively or assume it's missing
            # For now, just placeholder
            neighbor_img = np.zeros((100, 100, 3), dtype=np.uint8)
            plt.text(0.5, 0.5, "Image Not Found", ha='center', va='center')
        else:
            neighbor_img = cv2.imread(neighbor_path)
            if neighbor_img is not None:
                neighbor_img = cv2.cvtColor(neighbor_img, cv2.COLOR_BGR2RGB)
            else:
                neighbor_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        plt.subplot(1, cols, i + 2)
        plt.imshow(neighbor_img)
        plt.title(f"Neighbor {i+1}\n{specy}\n{strain}\nScore: {score:.4f}")
        plt.axis('off')
        
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        
    if show_plot:
        plt.show()
    
    plt.close()

def visualize_batch_predictions(
    results: List[Dict[str, Any]],
    output_dir: str = str(RESULTS_DIR / "visualizations"),
    max_samples: int = 10
) -> None:
    """
    Visualize a batch of prediction results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, result in enumerate(results):
        if i >= max_samples:
            break
            
        strain = result['strain']
        raw_results = result.get('raw_results', [])
        
        # Visualize the first query of the strain
        if raw_results:
            first_query = raw_results[0]
            query_id = first_query['query_image_id']
            neighbors = first_query['neighbors']
            
            img_path = os.path.join(str(SEGMENTED_IMAGE_DIR), f"{query_id}.jpg")
            out_path = os.path.join(output_dir, f"pred_{strain}_{query_id}.jpg")
            
            visualize_prediction_result(
                query_image_path=img_path,
                neighbors=neighbors,
                output_path=out_path,
                ground_truth_species=result['ground_truth'],
                predicted_species=result['predicted_specy']
            )

if __name__ == "__main__":
    pass

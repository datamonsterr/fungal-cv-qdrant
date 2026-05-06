import argparse

from src.prepare.dataset import prepare_dataset


def run_yolo_segmentation(
    source_dir: str,
    model_id: str = "my-first-project-3ddqp/2",
    local_model_path: str | None = None,
    confidence_threshold: float = 0.25,
    create_hierarchical: bool = True,
):
    del source_dir, model_id, local_model_path, confidence_threshold, create_hierarchical
    return prepare_dataset(source_collections=["incoming"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare canonical dataset for incoming images")
    parser.add_argument("--source", type=str, default="Dataset/new_data")
    parser.add_argument("--model-id", type=str, default="my-first-project-3ddqp/2")
    parser.add_argument("--local-path", type=str, default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--no-hierarchical", action="store_true")
    args = parser.parse_args()
    run_yolo_segmentation(
        source_dir=args.source,
        model_id=args.model_id,
        local_model_path=args.local_path,
        confidence_threshold=args.confidence,
        create_hierarchical=not args.no_hierarchical,
    )

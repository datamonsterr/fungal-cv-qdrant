"""
Training script for Google Colab - Feature Extractor Fine-tuning
Trains deep learning backbones for feature extraction (without classification head)

Usage in Colab:
1. Mount Google Drive
2. Ensure dataset is in /content/drive/MyDrive/mycoai/Dataset/hierarchical
3. Run this script to fine-tune ResNet50, MobileNetV2, and EfficientNetB1
"""

import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B1_Weights,
    MobileNet_V2_Weights,
    ResNet50_Weights,
    efficientnet_b1,
    mobilenet_v2,
    resnet50,
)
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

# Handle environments where __file__ is not defined (e.g., Google Colab)
try:
    _default_root = Path(__file__).parent.parent
except NameError:
    _default_root = Path.cwd()

os.environ["DATASET_ROOT"] = "/content/drive/MyDrive/mycoai/dataset"
os.environ["WEIGHTS_DIR"] = "/content/drive/MyDrive/mycoai/weights"
os.environ["RESULTS_DIR"] = "/content/drive/MyDrive/mycoai/results"

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", _default_root))
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", PROJECT_ROOT / "Dataset"))
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", PROJECT_ROOT / "weights"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "results"))

# Dataset Paths
ORIGINAL_DATASET_PATH = Path(
    os.getenv("ORIGINAL_DATASET_PATH", DATASET_ROOT / "original")
)
FULL_IMAGE_PATH = Path(os.getenv("FULL_IMAGE_PATH", DATASET_ROOT / "full_image"))
SEGMENTED_IMAGE_DIR = Path(
    os.getenv("SEGMENTED_IMAGE_DIR", DATASET_ROOT / "segmented_image")
)
HIERARCHICAL_DATASET_PATH = Path(
    os.getenv("HIERARCHICAL_DATASET_PATH", DATASET_ROOT / "hierarchical")
)

# Metadata Paths
FULL_IMAGE_METADATA_PATH = Path(
    os.getenv("FULL_IMAGE_METADATA_PATH", DATASET_ROOT / "full_image_metadata.json")
)
SEGMENTED_METADATA_PATH = Path(
    os.getenv("SEGMENTED_METADATA_PATH", DATASET_ROOT / "segmented_image_metadata.json")
)
STRAIN_SPECIES_MAPPING_PATH = Path(
    os.getenv("STRAIN_SPECIES_MAPPING_PATH", DATASET_ROOT / "strain_to_specy.csv")
)

# Feature Paths
FEATURES_JSON_PATH = Path(
    os.getenv("FEATURES_JSON_PATH", DATASET_ROOT / "segmented_features.json")
)

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "myco_fungi_features_full"

# Image Processing
HEIGHT = 256
WIDTH = 256
TARGET_SIZE = (HEIGHT, WIDTH)

# Ensure directories exist
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# Dataset Class
# ============================================================================


class FungiDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels: Union[List[int], np.ndarray],
        transform=None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================================
# Model Creation
# ============================================================================


def get_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    """Build model with ImageNet weights, unfreeze all layers for training."""
    if model_name == "ResNet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
        # Replace classification head
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "MobileNetV2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
        # Replace classification head
        num_ftrs = model.classifier[1].in_features  # type: ignore[union-attr]
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)  # type: ignore[assignment]
    elif model_name == "EfficientNetB1":
        model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
        # Replace classification head
        num_ftrs = model.classifier[1].in_features  # type: ignore[union-attr]
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model.to(device)


def save_backbone_weights(model: nn.Module, model_name: str, save_path: Path):
    """Save model backbone weights without the classification head."""
    state_dict = model.state_dict()

    # Remove classification head from state dict
    if model_name == "ResNet50":
        backbone_state = {
            k: v for k, v in state_dict.items() if not k.startswith("fc.")
        }
    elif model_name == "MobileNetV2":
        backbone_state = {
            k: v for k, v in state_dict.items() if not k.startswith("classifier.")
        }
    elif model_name == "EfficientNetB1":  # noqa: W293
        backbone_state = {
            k: v for k, v in state_dict.items() if not k.startswith("classifier.")
        }
    else:
        backbone_state = state_dict

    torch.save(backbone_state, save_path)
    print(f"Saved backbone weights (without classification head) to {save_path}")


def visualize_training_history(
    history: Dict[str, List[float]], model_name: str, save_path: Path
):
    """Visualize and save training history plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    epochs = range(1, len(history["accuracy"]) + 1)
    ax1.plot(epochs, history["accuracy"], "b-", label="Training Accuracy", linewidth=2)
    ax1.plot(
        epochs, history["val_accuracy"], "r-", label="Validation Accuracy", linewidth=2
    )
    ax1.set_title(f"{model_name} - Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot loss
    ax2.plot(epochs, history["loss"], "b-", label="Training Loss", linewidth=2)
    ax2.plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    ax2.set_title(f"{model_name} - Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training visualization to {save_path}")


# ============================================================================
# Training Function
# ============================================================================


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 25,
    patience: int = 5,
    save_path: Optional[Path] = None,
) -> Dict[str, List[float]]:

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)  # type: ignore[arg-type]
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)  # type: ignore[union-attr,arg-type]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "train":
                history["loss"].append(epoch_loss)
                history["accuracy"].append(epoch_acc.item())
            else:
                history["val_loss"].append(epoch_loss)
                history["val_accuracy"].append(epoch_acc.item())

                # Deep copy the model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {epochs_no_improve} epochs without improvement"
            )
            break

    print(f"Best val Acc: {best_acc:4f}")
    model.load_state_dict(best_model_wts)
    return history


# ============================================================================
# Main Training Script
# ============================================================================


def main():  # noqa: C901
    # Configuration
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001  # Lower LR for fine-tuning
    PATIENCE = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print paths for verification
    print("\n" + "=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    print(f"HIERARCHICAL_DATASET_PATH: {HIERARCHICAL_DATASET_PATH}")
    print(f"WEIGHTS_DIR: {WEIGHTS_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print(f"SEGMENTED_METADATA_PATH: {SEGMENTED_METADATA_PATH}")
    print(f"STRAIN_SPECIES_MAPPING_PATH: {STRAIN_SPECIES_MAPPING_PATH}")
    print("=" * 60 + "\n")

    # Load metadata and mappings
    if not STRAIN_SPECIES_MAPPING_PATH.exists():
        print(f"Error: {STRAIN_SPECIES_MAPPING_PATH} not found.")
        print(
            "Please ensure the strain_to_specy.csv file exists in your dataset directory."
        )
        return

    df_mapping = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
    if "Test" not in df_mapping.columns:
        print("Error: 'Test' column not found in mapping CSV.")
        print("Please ensure the CSV has the proper format.")
        return

    # Filter test strains (where Test column is True)
    test_strain_set = set(df_mapping[df_mapping["Test"]]["Strain"].tolist())

    print(f"Selected {len(test_strain_set)} strains for testing (one per species)")

    # Create strain to species mapping dictionary
    strain_to_species = dict(zip(df_mapping["Strain"], df_mapping["Species"]))
    print(f"Strain to species mapping: {len(strain_to_species)} entries")

    # Prepare dataset
    if not SEGMENTED_METADATA_PATH.exists():
        print(f"Error: {SEGMENTED_METADATA_PATH} not found.")
        return

    with open(SEGMENTED_METADATA_PATH, "r") as f:
        metadata_list = json.load(f)

    train_paths = []
    train_labels_raw = []
    val_paths = []
    val_labels_raw = []

    skipped_unknown = 0
    skipped_missing = 0
    skipped_not_found = 0

    for item in metadata_list:
        # Handle both flat and nested metadata structure
        data = item.get("data", item)
        strain = data.get("strain")
        image_id = item.get("id")
        environment = data.get("environment")
        angle = data.get("angle")
        parent_id = item.get("parent_id")

        if not strain or not image_id:
            skipped_missing += 1
            continue

        # Get species from CSV mapping instead of metadata
        species = strain_to_species.get(strain)

        if not species or species == "unknown":
            skipped_unknown += 1
            continue

        # Reconstruct hierarchical path using correct species from CSV
        # Extract segment index from image_id (format: parent_id_segmentIndex)
        if parent_id and "_" in image_id:
            segment_suffix = image_id.split("_")[-1]
        else:
            segment_suffix = "0"

        # Reconstruct the hierarchical filename
        clean_strain = strain.replace(" ", "_").replace("/", "-")
        hierarchical_filename = (
            f"{clean_strain}_{environment}_{angle}_seg{segment_suffix}.jpg"
        )

        # Construct the full hierarchical path with correct species
        hierarchical_path = (
            HIERARCHICAL_DATASET_PATH
            / species
            / strain
            / environment
            / hierarchical_filename
        )

        # Fall back to segmented_image if hierarchical doesn't exist
        if hierarchical_path.exists():
            image_path = hierarchical_path
        else:
            image_path = SEGMENTED_IMAGE_DIR / f"{image_id}.jpg"

        if not image_path.exists():
            skipped_not_found += 1
            # Debug: print first few missing paths
            if skipped_not_found <= 5:
                print(f"  Warning: File not found: {image_path}")
                print(f"    Tried hierarchical: {hierarchical_path}")
                print(f"    Tried segmented: {SEGMENTED_IMAGE_DIR / f'{image_id}.jpg'}")
            continue

        if strain in test_strain_set:
            val_paths.append(str(image_path))
            val_labels_raw.append(species)
        else:
            train_paths.append(str(image_path))
            train_labels_raw.append(species)

    print(f"Skipped {skipped_unknown} images with unknown species")
    print(f"Skipped {skipped_missing} images with missing strain/id")
    print(f"Skipped {skipped_not_found} images not found on disk")

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    if len(train_paths) == 0 or len(val_paths) == 0:
        print("Error: No training or validation samples found.")
        return

    # Encode labels
    le = LabelEncoder()
    all_labels = train_labels_raw + val_labels_raw
    le.fit(all_labels)
    train_labels = le.transform(train_labels_raw)
    val_labels = le.transform(val_labels_raw)

    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {le.classes_}")

    # Save label encoder classes
    np.save(WEIGHTS_DIR / "classes.npy", le.classes_)
    print(f"Saved classes to {WEIGHTS_DIR / 'classes.npy'}")

    # Transforms
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((HEIGHT, WIDTH)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((HEIGHT, WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Datasets and Dataloaders
    image_datasets = {
        "train": FungiDataset(train_paths, train_labels, data_transforms["train"]),  # type: ignore[arg-type]
        "val": FungiDataset(val_paths, val_labels, data_transforms["val"]),  # type: ignore[arg-type]
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        for x in ["train", "val"]
    }

    # Train models
    models_to_train = ["ResNet50", "MobileNetV2", "EfficientNetB1"]

    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print("=" * 60)

        model = get_model(model_name, num_classes, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        history = train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            device,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            save_path=None,
        )

        # Save backbone weights (without classification head)
        backbone_save_path = WEIGHTS_DIR / f"{model_name}_finetuned.pth"
        save_backbone_weights(model, model_name, backbone_save_path)

        # Save history
        history_path = WEIGHTS_DIR / f"{model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f)
        print(f"Saved training history to {history_path}")

        # Visualize training history
        viz_path = WEIGHTS_DIR / f"{model_name}_training_history.png"
        visualize_training_history(history, model_name, viz_path)

    print("\n" + "=" * 60)
    print("Training completed for all models!")
    print("=" * 60)


if __name__ == "__main__":
    main()

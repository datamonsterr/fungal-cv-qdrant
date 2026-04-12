import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1
from tqdm import tqdm

# ============================================================================
# Configuration & Paths
# ============================================================================
# (Keeping your original path setup)
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
SEGMENTED_IMAGE_DIR = Path(
    os.getenv("SEGMENTED_IMAGE_DIR", DATASET_ROOT / "segmented_image")
)
HIERARCHICAL_DATASET_PATH = Path(
    os.getenv("HIERARCHICAL_DATASET_PATH", DATASET_ROOT / "hierarchical")
)
SEGMENTED_METADATA_PATH = Path(
    os.getenv("SEGMENTED_METADATA_PATH", DATASET_ROOT / "segmented_image_metadata.json")
)
STRAIN_SPECIES_MAPPING_PATH = Path(
    os.getenv("STRAIN_SPECIES_MAPPING_PATH", DATASET_ROOT / "strain_to_specy.csv")
)

# Image Params
HEIGHT = 256
WIDTH = 256

# Ensure directories exist
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================================
# 1. Triplet Dataset Class
# ============================================================================
class TripletFungiDataset(Dataset):
    """
    Returns triplets: (anchor, positive, negative) + label
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: Union[List[int], np.ndarray],
        transform=None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        # Group data by label for fast sampling
        self.data_dict = defaultdict(list)
        self.all_labels = set(labels)
        for path, label in zip(image_paths, labels):
            self.data_dict[label].append(path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Anchor
        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]

        # 2. Positive (Same label)
        # Try to find a different image of the same class
        possible_positives = self.data_dict[anchor_label]
        if len(possible_positives) > 1:
            # removing anchor from options is ideal, but for speed random choice is usually fine
            # if the dataset is large enough. Here we pick randomly.
            pos_path = random.choice(possible_positives)
        else:
            # If only 1 image exists for this class, use the anchor itself
            pos_path = anchor_path

        # 3. Negative (Different label)
        neg_label = random.choice(list(self.all_labels - {anchor_label}))
        neg_path = random.choice(self.data_dict[neg_label])

        anchor_img = self.load_image(anchor_path)
        pos_img = self.load_image(pos_path)
        neg_img = self.load_image(neg_path)

        return anchor_img, pos_img, neg_img, anchor_label

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# ============================================================================
# 2. EfficientNet Embedding Model
# ============================================================================
def get_efficientnet_embedding_model(
    embedding_dim: int, device: torch.device
) -> nn.Module:
    """Builds EfficientNetB1 replacing the head with an embedding layer."""
    model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    # Get input features of the final layer
    # EfficientNet B1 classifier structure: Sequential(Dropout, Linear)
    num_ftrs = model.classifier[1].in_features

    # Replace with embedding head (Linear layer)
    # No activation function at the end, normalization happens in training loop
    model.classifier[1] = nn.Linear(num_ftrs, embedding_dim)

    return model.to(device)


# ============================================================================
# 3. Triplet Training Loop
# ============================================================================
def train_triplet_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 25,
    patience: int = 10,
    save_path: Optional[Path] = None,
) -> Dict[str, List[float]]:

    history = {"loss": [], "val_loss": []}
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            # Iterate over triplets
            pbar = tqdm(dataloaders[phase], desc=f"{phase}")
            for anchor, positive, negative, _ in pbar:
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # Forward pass
                    a_out = model(anchor)
                    p_out = model(positive)
                    n_out = model(negative)

                    # Normalize embeddings (Important for Triplet Loss)
                    a_out = F.normalize(a_out, p=2, dim=1)
                    p_out = F.normalize(p_out, p=2, dim=1)
                    n_out = F.normalize(n_out, p=2, dim=1)

                    loss = criterion(a_out, p_out, n_out)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * anchor.size(0)

                # Update progress bar
                pbar.set_postfix({"loss": loss.item()})

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f}")

            if phase == "train":
                history["loss"].append(epoch_loss)
            else:
                history["val_loss"].append(epoch_loss)

                # Save best model
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    if save_path:
                        torch.save(model.state_dict(), save_path)
                        print(f"  New best model saved! Loss: {best_loss:.4f}")
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epochs_no_improve} epochs.")
            break

    return history


# ============================================================================
# 4. Main Function
# ============================================================================
def main():
    # --- Configuration ---
    # Reduced batch size because Triplet training loads 3 images per sample
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    EMBEDDING_DIM = 128
    PATIENCE = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: EfficientNetB1 | Loss: TripletMarginLoss | Dim: {EMBEDDING_DIM}")

    # --- Load Metadata & Mappings ---
    if not STRAIN_SPECIES_MAPPING_PATH.exists():
        print(f"Error: {STRAIN_SPECIES_MAPPING_PATH} not found.")
        return

    df_mapping = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
    test_strain_set = set(df_mapping[df_mapping["Test"]]["Strain"].tolist())
    strain_to_species = dict(zip(df_mapping["Strain"], df_mapping["Species"]))

    if not SEGMENTED_METADATA_PATH.exists():
        print(f"Error: {SEGMENTED_METADATA_PATH} not found.")
        return

    with open(SEGMENTED_METADATA_PATH, "r") as f:
        metadata_list = json.load(f)

    train_paths = []
    train_labels_raw = []
    val_paths = []
    val_labels_raw = []

    # --- Parse Metadata ---
    for item in metadata_list:
        data = item.get("data", item)
        strain = data.get("strain")
        image_id = item.get("id")
        environment = data.get("environment")
        angle = data.get("angle")
        parent_id = item.get("parent_id")

        if not strain or not image_id:
            continue

        species = strain_to_species.get(strain)
        if not species or species == "unknown":
            continue

        # Logic to find image path (hierarchical -> flat fallback)
        segment_suffix = (
            image_id.split("_")[-1] if parent_id and "_" in image_id else "0"
        )
        clean_strain = strain.replace(" ", "_").replace("/", "-")
        hierarchical_filename = (
            f"{clean_strain}_{environment}_{angle}_seg{segment_suffix}.jpg"
        )

        hierarchical_path = (
            HIERARCHICAL_DATASET_PATH
            / species
            / strain
            / environment
            / hierarchical_filename
        )

        if hierarchical_path.exists():
            image_path = hierarchical_path
        else:
            image_path = SEGMENTED_IMAGE_DIR / f"{image_id}.jpg"

        if not image_path.exists():
            continue

        if strain in test_strain_set:
            val_paths.append(str(image_path))
            val_labels_raw.append(species)
        else:
            train_paths.append(str(image_path))
            train_labels_raw.append(species)

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    # --- Encode Labels ---
    le = LabelEncoder()
    all_labels = train_labels_raw + val_labels_raw
    le.fit(all_labels)
    train_labels = le.transform(train_labels_raw)
    val_labels = le.transform(val_labels_raw)

    np.save(WEIGHTS_DIR / "classes.npy", le.classes_)

    # --- Transforms ---
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

    # --- Datasets (Triplet) ---
    # Using the new TripletFungiDataset class
    image_datasets = {
        "train": TripletFungiDataset(
            train_paths, train_labels, data_transforms["train"]
        ),
        "val": TripletFungiDataset(val_paths, val_labels, data_transforms["val"]),
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        for x in ["train", "val"]
    }

    # --- Model Setup (EfficientNet Only) ---
    model_name = "EfficientNetB1"
    model = get_efficientnet_embedding_model(EMBEDDING_DIM, device)

    # --- Triplet Loss Setup ---
    # Margin: How far apart positive and negative should be
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training ---
    print(f"\nStarting training for {model_name}...")
    save_path = WEIGHTS_DIR / f"{model_name}_triplet.pth"

    history = train_triplet_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        device,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        save_path=save_path,
    )

    # --- Save History ---
    history_path = WEIGHTS_DIR / f"{model_name}_triplet_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f)

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"{model_name} Triplet Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(WEIGHTS_DIR / f"{model_name}_triplet_plot.png")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Weights saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

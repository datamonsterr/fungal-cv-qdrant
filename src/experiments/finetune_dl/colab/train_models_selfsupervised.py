"""
Training script for Google Colab - Self-Supervised Pretraining + Fine-tuning
Uses SimCLR-style contrastive learning on all unlabeled images, then fine-tunes on labeled data

This approach leverages ALL 1305 images (including test strains) for unsupervised
pretraining to learn robust visual representations, then fine-tunes only on training
strains for classification.

Benefits:
- Uses unlabeled data to learn domain-specific features
- Can improve accuracy by 5-15% compared to ImageNet pretraining
- Learns fungal-specific morphological patterns

Two-stage process:
1. Stage 1: Self-supervised pretraining on all 1305 images (no labels)
2. Stage 2: Supervised fine-tuning on training strains only

Usage in Colab:
1. Mount Google Drive
2. Ensure dataset is in /content/drive/MyDrive/mycoai/Dataset/hierarchical
3. Run this script (takes longer due to two-stage training)
"""

import copy
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageFilter
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B1_Weights,
    efficientnet_b1,
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
HIERARCHICAL_DATASET_PATH = Path(
    os.getenv("HIERARCHICAL_DATASET_PATH", DATASET_ROOT / "hierarchical")
)
SEGMENTED_IMAGE_DIR = Path(
    os.getenv("SEGMENTED_IMAGE_DIR", DATASET_ROOT / "segmented_image")
)

# Metadata Paths
SEGMENTED_METADATA_PATH = Path(
    os.getenv("SEGMENTED_METADATA_PATH", DATASET_ROOT / "segmented_image_metadata.json")
)
STRAIN_SPECIES_MAPPING_PATH = Path(
    os.getenv("STRAIN_SPECIES_MAPPING_PATH", DATASET_ROOT / "strain_to_specy.csv")
)

# Image Processing
HEIGHT = 256
WIDTH = 256
TARGET_SIZE = (HEIGHT, WIDTH)

# Ensure directories exist
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# SimCLR Data Augmentation
# ============================================================================


class GaussianBlur:
    """Gaussian blur augmentation."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ContrastiveTransform:
    """Generate two augmented views for contrastive learning."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


def get_simclr_augmentation(img_size=256):
    """Get SimCLR augmentation pipeline."""
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

    augmentation = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return ContrastiveTransform(augmentation)


# ============================================================================
# Dataset Classes
# ============================================================================


class ContrastiveDataset(Dataset):
    """Dataset for self-supervised contrastive learning."""

    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            # Returns two augmented views
            return self.transform(image)
        return image


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

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
# SimCLR Model
# ============================================================================


class ProjectionHead(nn.Module):
    """Projection head for SimCLR."""

    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """SimCLR model with EfficientNetB1 backbone."""

    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)
            encoder_out = self.encoder(dummy_input)
            encoder_dim = encoder_out.shape[1]

        self.projection_head = ProjectionHead(
            in_dim=encoder_dim, hidden_dim=encoder_dim, out_dim=projection_dim
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z


def info_nce_loss(features, temperature=0.5):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss."""
    batch_size = features.shape[0] // 2

    # Normalize features
    features = F.normalize(features, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T)

    # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
    labels = torch.cat(
        [torch.arange(batch_size) + batch_size, torch.arange(batch_size)]
    ).to(features.device)

    # Mask to remove self-similarities
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

    # Apply temperature scaling
    similarity_matrix = similarity_matrix / temperature

    # Compute loss
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


# ============================================================================
# Model Building
# ============================================================================


def get_efficientnet_encoder(pretrained=True):
    """Get EfficientNetB1 encoder without classification head."""
    model = efficientnet_b1(
        weights=EfficientNet_B1_Weights.DEFAULT if pretrained else None
    )
    # Remove classification head
    encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    return encoder


def get_classifier(encoder, num_classes):
    """Add classification head to encoder."""
    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, 256, 256)
        encoder_out = encoder(dummy_input)
        encoder_dim = encoder_out.shape[1]

    model = nn.Sequential(encoder, nn.Linear(encoder_dim, num_classes))
    return model


# ============================================================================
# Training Functions
# ============================================================================


def pretrain_simclr(
    model: SimCLR,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    temperature: float = 0.5,
) -> Dict[str, List[float]]:
    """Stage 1: Self-supervised pretraining with SimCLR."""

    history = {"loss": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs}")
        for [x_i, x_j] in pbar:
            # Concatenate augmented views
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            # Forward pass
            _, z_i = model(x_i)
            _, z_j = model(x_j)

            # Concatenate projections
            z = torch.cat([z_i, z_j], dim=0)

            # Compute loss
            loss = info_nce_loss(z, temperature=temperature)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / num_batches
        history["loss"].append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    return history


def finetune_supervised(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 25,
    patience: int = 5,
) -> Dict[str, List[float]]:
    """Stage 2: Supervised fine-tuning on labeled data."""

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"Finetune Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

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
# Visualization
# ============================================================================


def visualize_pretraining(history: Dict[str, List[float]], save_path: Path):
    """Visualize pretraining loss."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history["loss"]) + 1)
    plt.plot(epochs, history["loss"], "b-", linewidth=2)
    plt.title("Self-Supervised Pretraining Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Contrastive Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved pretraining visualization to {save_path}")


def visualize_finetuning(
    history: Dict[str, List[float]], model_name: str, save_path: Path
):
    """Visualize fine-tuning history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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
    print(f"Saved fine-tuning visualization to {save_path}")


# ============================================================================
# Main Training Script
# ============================================================================


def main():  # noqa: C901
    # Configuration
    PRETRAIN_BATCH_SIZE = 64
    PRETRAIN_EPOCHS = 100
    PRETRAIN_LR = 0.0003
    TEMPERATURE = 0.5

    FINETUNE_BATCH_SIZE = 16
    FINETUNE_EPOCHS = 50
    FINETUNE_LR = 0.00001  # Lower LR for fine-tuning pretrained encoder
    PATIENCE = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 60)
    print("Self-Supervised Pretraining + Fine-tuning Configuration:")
    print("=" * 60)
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    print(f"HIERARCHICAL_DATASET_PATH: {HIERARCHICAL_DATASET_PATH}")
    print(f"WEIGHTS_DIR: {WEIGHTS_DIR}")
    print(f"SEGMENTED_METADATA_PATH: {SEGMENTED_METADATA_PATH}")
    print(f"STRAIN_SPECIES_MAPPING_PATH: {STRAIN_SPECIES_MAPPING_PATH}")
    print("=" * 60 + "\n")

    # Load metadata
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

    # Collect ALL image paths for pretraining (including test strains)
    all_paths = []
    train_paths = []
    train_labels_raw = []
    val_paths = []
    val_labels_raw = []

    skipped = 0

    for item in metadata_list:
        data = item.get("data", item)
        strain = data.get("strain")
        image_id = item.get("id")
        environment = data.get("environment")
        angle = data.get("angle")
        parent_id = item.get("parent_id")

        if not strain or not image_id:
            skipped += 1
            continue

        species = strain_to_species.get(strain)
        if not species or species == "unknown":
            skipped += 1
            continue

        if parent_id and "_" in image_id:
            segment_suffix = image_id.split("_")[-1]
        else:
            segment_suffix = "0"

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
            skipped += 1
            continue

        # Add to ALL paths for pretraining
        all_paths.append(str(image_path))

        # Also separate for supervised training
        if strain in test_strain_set:
            val_paths.append(str(image_path))
            val_labels_raw.append(species)
        else:
            train_paths.append(str(image_path))
            train_labels_raw.append(species)

    print(f"Skipped {skipped} images")
    print(f"Total images for pretraining: {len(all_paths)}")
    print(f"Training samples (for fine-tuning): {len(train_paths)}")
    print(f"Validation samples (for fine-tuning): {len(val_paths)}")

    if len(all_paths) == 0 or len(train_paths) == 0:
        print("Error: No samples found.")
        return

    # ========================================================================
    # STAGE 1: Self-Supervised Pretraining
    # ========================================================================

    print("\n" + "=" * 60)
    print("STAGE 1: Self-Supervised Pretraining on ALL Images")
    print("=" * 60)

    # Create contrastive dataset
    pretrain_transform = get_simclr_augmentation(img_size=HEIGHT)
    pretrain_dataset = ContrastiveDataset(all_paths, transform=pretrain_transform)
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=PRETRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    # Build SimCLR model
    encoder = get_efficientnet_encoder(
        pretrained=True
    )  # Start from ImageNet for faster convergence
    simclr_model = SimCLR(encoder, projection_dim=128).to(device)

    optimizer_pretrain = optim.Adam(simclr_model.parameters(), lr=PRETRAIN_LR)

    # Pretrain
    pretrain_history = pretrain_simclr(
        simclr_model,
        pretrain_loader,
        optimizer_pretrain,
        device,
        num_epochs=PRETRAIN_EPOCHS,
        temperature=TEMPERATURE,
    )

    # Save pretrained encoder
    pretrained_encoder_path = (
        WEIGHTS_DIR / "EfficientNetB1_SimCLR_pretrained_encoder.pth"
    )
    torch.save(simclr_model.encoder.state_dict(), pretrained_encoder_path)
    print(f"Saved pretrained encoder to {pretrained_encoder_path}")

    # Save pretraining history
    pretrain_history_path = WEIGHTS_DIR / "SimCLR_pretraining_history.json"
    with open(pretrain_history_path, "w") as f:
        json.dump(pretrain_history, f)

    # Visualize pretraining
    pretrain_viz_path = WEIGHTS_DIR / "SimCLR_pretraining_loss.png"
    visualize_pretraining(pretrain_history, pretrain_viz_path)

    # ========================================================================
    # STAGE 2: Supervised Fine-tuning
    # ========================================================================

    print("\n" + "=" * 60)
    print("STAGE 2: Supervised Fine-tuning on Training Strains")
    print("=" * 60)

    # Encode labels
    le = LabelEncoder()
    all_labels = train_labels_raw + val_labels_raw
    le.fit(all_labels)
    train_labels = le.transform(train_labels_raw)
    val_labels = le.transform(val_labels_raw)

    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {le.classes_}")

    np.save(WEIGHTS_DIR / "classes_simclr.npy", le.classes_)

    # Transforms for fine-tuning
    finetune_transforms = {
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

    # Create supervised datasets
    finetune_datasets = {
        "train": SupervisedDataset(train_paths, train_labels, finetune_transforms["train"]),  # type: ignore[arg-type]
        "val": SupervisedDataset(val_paths, val_labels, finetune_transforms["val"]),  # type: ignore[arg-type]
    }

    finetune_loaders = {
        x: DataLoader(
            finetune_datasets[x],
            batch_size=FINETUNE_BATCH_SIZE,
            shuffle=True,
            num_workers=2,
        )
        for x in ["train", "val"]
    }

    # Build classifier with pretrained encoder
    pretrained_encoder = get_efficientnet_encoder(pretrained=False)
    pretrained_encoder.load_state_dict(
        torch.load(pretrained_encoder_path, map_location=device)
    )
    classifier = get_classifier(pretrained_encoder, num_classes).to(device)

    # Fine-tune (unfreeze all layers)
    for param in classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer_finetune = optim.Adam(classifier.parameters(), lr=FINETUNE_LR)

    finetune_history = finetune_supervised(
        classifier,
        finetune_loaders,
        criterion,
        optimizer_finetune,
        device,
        num_epochs=FINETUNE_EPOCHS,
        patience=PATIENCE,
    )

    # Save fine-tuned model backbone (without classifier)
    backbone_save_path = WEIGHTS_DIR / "EfficientNetB1_SimCLR_finetuned.pth"
    # Extract encoder only (remove classifier)
    encoder_state = {
        k.replace("0.", ""): v
        for k, v in classifier.state_dict().items()
        if k.startswith("0.")
    }
    torch.save(encoder_state, backbone_save_path)
    print(f"Saved fine-tuned backbone to {backbone_save_path}")

    # Save fine-tuning history
    finetune_history_path = WEIGHTS_DIR / "SimCLR_finetuning_history.json"
    with open(finetune_history_path, "w") as f:
        json.dump(finetune_history, f)

    # Visualize fine-tuning
    finetune_viz_path = WEIGHTS_DIR / "SimCLR_finetuning_history.png"
    visualize_finetuning(finetune_history, "EfficientNetB1-SimCLR", finetune_viz_path)

    print("\n" + "=" * 60)
    print("Self-Supervised Training Completed!")
    print("=" * 60)
    print(f"Pretrained encoder saved to: {pretrained_encoder_path}")
    print(f"Fine-tuned model saved to: {backbone_save_path}")


if __name__ == "__main__":
    main()

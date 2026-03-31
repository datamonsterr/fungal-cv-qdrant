"""Fold-specific EfficientNetB1 fine-tuning script for notebook environments.

This script is for Colab, AI factories, and other notebook environments.
It automatically unzips hierarchical data and fold CSV files, then trains
one fold at a time using a fold mapping CSV with columns: Strain, Species, Test.

Saved artifacts (all in WEIGHTS_DIR):
- fold{idx}_EfficientNetB1_finetuned.pth (backbone-only)
- fold{idx}_EfficientNetB1_history.json
- fold{idx}_EfficientNetB1_training_history.png
- fold{idx}_EfficientNetB1_confusion_matrix.png
- fold{idx}_EfficientNetB1_per_class_accuracy.png
- fold{idx}_EfficientNetB1_val_predictions.csv
"""

import copy
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1
from tqdm import tqdm

# ============================================================================
# Environment / Paths / Unzip Setup
# ============================================================================

try:
    _default_root = Path(__file__).parent.parent
except NameError:
    _default_root = Path.cwd()

# Paths for notebook environment
os.environ.setdefault("NOTEBOOK_ROOT", str(Path.cwd()))
os.environ.setdefault("DATASET_ROOT", "Dataset")
os.environ.setdefault("WEIGHTS_DIR", "weights")
os.environ.setdefault("RESULTS_DIR", "results")
os.environ.setdefault("FOLD_ZIP_PATH", "mycoai_data.zip")
os.environ.setdefault("FOLD_INDEX", "0")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", _default_root))
NOTEBOOK_ROOT = Path(os.getenv("NOTEBOOK_ROOT", PROJECT_ROOT))
DATASET_ROOT = NOTEBOOK_ROOT / os.getenv("DATASET_ROOT", "Dataset")
WEIGHTS_DIR = NOTEBOOK_ROOT / os.getenv("WEIGHTS_DIR", "weights")
RESULTS_DIR = NOTEBOOK_ROOT / os.getenv("RESULTS_DIR", "results")
FOLD_ZIP_PATH = Path(os.getenv("FOLD_ZIP_PATH", "mycoai_data.zip"))
FOLD_INDEX = int(os.getenv("FOLD_INDEX", "0"))

HIERARCHICAL_DIR = DATASET_ROOT / "hierarchical"
FOLD_MAPPING_PATH = DATASET_ROOT / f"strain_to_specy_fold{FOLD_INDEX}.csv"

HEIGHT = 256
WIDTH = 256

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Unzip utilities
# ============================================================================


def extract_zip_if_needed(zip_path: Path, extract_to: Path) -> None:
    """Extract zip if not already extracted, then validate required files."""
    if not zip_path.exists():
        print(f"Warning: zip file not found at {zip_path}")
        print("Assuming data is already extracted.")
        return

    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=extract_to)
    print("✓ Extraction complete")

    if not HIERARCHICAL_DIR.exists():
        raise FileNotFoundError(
            f"Hierarchical images not found at {HIERARCHICAL_DIR} after extraction"
        )
    if not FOLD_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Fold mapping not found at {FOLD_MAPPING_PATH} after extraction"
        )

    print(f"✓ Hierarchical data found: {HIERARCHICAL_DIR}")
    print(f"✓ Fold mapping found: {FOLD_MAPPING_PATH}")


# ============================================================================
# Dataset
# ============================================================================


class FungiDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: np.ndarray, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================================
# Model / Training
# ============================================================================


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = True
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model.to(device)


def save_backbone_weights(model: nn.Module, save_path: Path) -> None:
    state_dict = model.state_dict()
    backbone_state = {
        k: v for k, v in state_dict.items() if not k.startswith("classifier.")
    }
    torch.save(backbone_state, save_path)
    print(f"Saved backbone weights to: {save_path}")


def train_one_fold(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "train":
                history["loss"].append(float(epoch_loss))
                history["accuracy"].append(float(epoch_acc.item()))
            else:
                history["val_loss"].append(float(epoch_loss))
                history["val_accuracy"].append(float(epoch_acc.item()))

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping after {epochs_no_improve} epochs without improvement"
            )
            break

    print(f"Best validation accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, history


# ============================================================================
# Evaluation / Visualization
# ============================================================================


def evaluate_on_validation(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    class_names: np.ndarray,
) -> pd.DataFrame:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation inference"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    rows = []
    for t, p in zip(y_true, y_pred):
        rows.append(
            {
                "ground_truth_idx": int(t),
                "predicted_idx": int(p),
                "ground_truth": str(class_names[t]),
                "predicted": str(class_names[p]),
                "correct": int(t == p),
            }
        )

    return pd.DataFrame(rows)


def save_training_history_plot(history: Dict[str, List[float]], out_path: Path) -> None:
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["accuracy"], label="Train Accuracy", linewidth=2)
    ax1.plot(epochs, history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax1.set_title("Training vs Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["loss"], label="Train Loss", linewidth=2)
    ax2.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    ax2.set_title("Training vs Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved history plot: {out_path}")


def save_confusion_matrix_plot(
    val_predictions: pd.DataFrame,
    class_names: np.ndarray,
    out_path: Path,
) -> None:
    cm = confusion_matrix(
        val_predictions["ground_truth_idx"],
        val_predictions["predicted_idx"],
        labels=list(range(len(class_names))),
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_title("Confusion Matrix (Validation Fold Test Set)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix: {out_path}")


def save_per_class_accuracy_plot(
    val_predictions: pd.DataFrame,
    class_names: np.ndarray,
    out_path: Path,
) -> None:
    per_class = (
        val_predictions.groupby("ground_truth")["correct"]
        .mean()
        .reindex(class_names, fill_value=0.0)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(per_class.index, per_class.values)
    ax.set_ylim(0, 1)
    ax.set_title("Per-Class Accuracy on Validation Fold Test Set")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Species")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved per-class accuracy plot: {out_path}")


# ============================================================================
# Data split from fold CSV + hierarchical structure
# ============================================================================


def load_fold_split(
    fold_mapping_path: Path,
    hierarchical_dir: Path,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load fold split directly from hierarchical folder structure.

    Hierarchical structure:
    hierarchical/
      Species1/
        Strain1/
          environment/
            image.jpg
    """

    if not fold_mapping_path.exists():
        raise FileNotFoundError(f"Fold mapping not found: {fold_mapping_path}")
    if not hierarchical_dir.exists():
        raise FileNotFoundError(f"Hierarchical directory not found: {hierarchical_dir}")

    fold_df = pd.read_csv(fold_mapping_path)
    required_cols = {"Strain", "Species", "Test"}
    if not required_cols.issubset(set(fold_df.columns)):
        raise ValueError(
            f"Fold CSV must include columns: {required_cols}. "
            f"Found: {list(fold_df.columns)}"
        )

    test_strains = set(fold_df[fold_df["Test"]]["Strain"].tolist())
    strain_to_species = dict(zip(fold_df["Strain"], fold_df["Species"]))

    train_paths: List[str] = []
    train_labels: List[str] = []
    val_paths: List[str] = []
    val_labels: List[str] = []

    # Scan hierarchical folder structure
    for species_dir in hierarchical_dir.iterdir():
        if not species_dir.is_dir():
            continue
        species = species_dir.name
        if species not in strain_to_species.values():
            continue

        for strain_dir in species_dir.iterdir():
            if not strain_dir.is_dir():
                continue
            strain = strain_dir.name

            for env_dir in strain_dir.iterdir():
                if not env_dir.is_dir():
                    continue

                for image_file in env_dir.glob("*.jpg"):
                    if strain in test_strains:
                        val_paths.append(str(image_file))
                        val_labels.append(species)
                    else:
                        train_paths.append(str(image_file))
                        train_labels.append(species)

    print(f"Fold mapping file: {fold_mapping_path}")
    print(f"Test strains: {len(test_strains)}")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    if not train_paths or not val_paths:
        raise RuntimeError(
            "Empty train/validation split. Check fold mapping and hierarchical data."
        )

    return train_paths, train_labels, val_paths, val_labels


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4
    patience = 10

    print("=" * 70)
    print("Fold-Specific EfficientNetB1 Fine-Tuning for Notebook Environments")
    print("=" * 70)

    # Extract data if needed
    extract_zip_if_needed(FOLD_ZIP_PATH, NOTEBOOK_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"NOTEBOOK_ROOT: {NOTEBOOK_ROOT}")
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    print(f"WEIGHTS_DIR: {WEIGHTS_DIR}")
    print(f"FOLD_INDEX: {FOLD_INDEX}")
    print(f"FOLD_MAPPING_PATH: {FOLD_MAPPING_PATH}")
    print(f"HIERARCHICAL_DIR: {HIERARCHICAL_DIR}\n")

    train_paths, train_labels_raw, val_paths, val_labels_raw = load_fold_split(
        fold_mapping_path=FOLD_MAPPING_PATH,
        hierarchical_dir=HIERARCHICAL_DIR,
    )

    le = LabelEncoder()
    le.fit(train_labels_raw + val_labels_raw)
    train_labels = le.transform(train_labels_raw)
    val_labels = le.transform(val_labels_raw)
    class_names = le.classes_

    np.save(WEIGHTS_DIR / f"fold{FOLD_INDEX}_classes.npy", class_names)

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

    datasets = {
        "train": FungiDataset(train_paths, train_labels, data_transforms["train"]),
        "val": FungiDataset(val_paths, val_labels, data_transforms["val"]),
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True, num_workers=2
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False, num_workers=2
        ),
    }

    model = build_model(num_classes=len(class_names), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model, history = train_one_fold(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
    )

    # Save fold artifacts in WEIGHTS_DIR
    prefix = f"fold{FOLD_INDEX}_EfficientNetB1"
    weight_path = WEIGHTS_DIR / f"{prefix}_finetuned.pth"
    history_json_path = WEIGHTS_DIR / f"{prefix}_history.json"
    history_plot_path = WEIGHTS_DIR / f"{prefix}_training_history.png"
    confmat_path = WEIGHTS_DIR / f"{prefix}_confusion_matrix.png"
    per_class_path = WEIGHTS_DIR / f"{prefix}_per_class_accuracy.png"
    val_pred_path = WEIGHTS_DIR / f"{prefix}_val_predictions.csv"

    save_backbone_weights(model, weight_path)

    with open(history_json_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history JSON: {history_json_path}")

    save_training_history_plot(history, history_plot_path)

    val_predictions = evaluate_on_validation(
        model=model,
        val_loader=dataloaders["val"],
        device=device,
        class_names=class_names,
    )
    val_predictions.to_csv(val_pred_path, index=False)
    print(f"Saved validation predictions: {val_pred_path}")

    save_confusion_matrix_plot(val_predictions, class_names, confmat_path)
    save_per_class_accuracy_plot(val_predictions, class_names, per_class_path)

    val_acc = float(val_predictions["correct"].mean())
    print("\n" + "=" * 70)
    print(f"Fold {FOLD_INDEX} training complete")
    print(f"Validation accuracy (selected test strains): {val_acc:.4f}")
    print(f"Artifacts saved in: {WEIGHTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

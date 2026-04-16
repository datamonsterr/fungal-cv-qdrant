"""
Training script for Google Colab - CellViT ViT Encoder Fine-tuning
Uses Vision Transformer pretrained on microscopy data (CellViT) for feature extraction

This script uses ViT encoder pretrained on biomedical microscopy images, which may
provide better domain-specific features compared to ImageNet pretraining.

Prerequisites:
1. Download CellViT pretrained weights from:
   https://drive.google.com/drive/folders/1zFO4bgo7yvjT9rCJi_6Mt6_07wfr0CKU?usp=sharing

   Available weight folders:
   - cellvit/: CellViT pretrained weights (nuclei segmentation on PanNuke)
   - sam_vit/: SAM-ViT pretrained weights (Segment Anything Model)
   - vit256/: ViT-256 pretrained weights (general microscopy)

2. Save the desired weights to /content/drive/MyDrive/mycoai/pretrained/
   Example structure:
   /content/drive/MyDrive/mycoai/pretrained/
   ├── cellvit/
   │   └── cellvit_sam.pth
   ├── sam_vit/
   │   └── sam_vit_b.pth
   └── vit256/
       └── vit_256_teacher.pth

Usage in Colab:
1. Mount Google Drive
2. Download and organize pretrained weights as shown above
3. Ensure dataset is in /content/drive/MyDrive/mycoai/Dataset/hierarchical
4. Configure PRETRAINED_WEIGHTS_CHOICE below (cellvit, sam_vit, or vit256)
5. Run this script to train ViT-based classifier
"""

import copy
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
from tqdm import tqdm

# Suppress TPU transparent hugepages warning (performance optimization, not critical)
warnings.filterwarnings("ignore", message=".*transparent_hugepage.*")

# TPU support
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    print("PyTorch XLA not available. TPU training disabled.")

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
os.environ["PRETRAINED_DIR"] = "/content/drive/MyDrive/mycoai/pretrained"

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", _default_root))
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", PROJECT_ROOT / "Dataset"))
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", PROJECT_ROOT / "weights"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "results"))
PRETRAINED_DIR = Path(os.getenv("PRETRAINED_DIR", PROJECT_ROOT / "pretrained"))

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

# Pretrained Weights Configuration
# Choose one of: "cellvit_x20", "cellvit_x40", "sam_vit_b", "sam_vit_l", "sam_vit_h",
#                "vit256_dino", or None for random initialization
PRETRAINED_WEIGHTS_CHOICE = (
    "vit256_dino"  # Change this to switch between weight options
)

# TPU Configuration
USE_TPU = True  # Set to True to use TPU if available
TPU_CORES = 2  # Number of TPU cores to use (1, 2, 4, or 8 for v5e)

# Pretrained weights paths for each option
PRETRAINED_WEIGHTS = {
    # CellViT models (nuclei segmentation pretrained)
    "cellvit_x20": PRETRAINED_DIR / "CellViT" / "CellViT-256-x20.pth",
    "cellvit_x40": PRETRAINED_DIR / "CellViT" / "CellViT-256-x40.pth",
    # SAM ViT encoder-only models (Segment Anything pretrained)
    "sam_vit_b": PRETRAINED_DIR / "SAM" / "sam_vit_b.pth",
    "sam_vit_l": PRETRAINED_DIR / "SAM" / "sam_vit_l.pth",
    "sam_vit_h": PRETRAINED_DIR / "SAM" / "sam_vit_h.pth",
    # ViT-256 DINO self-supervised
    "vit256_dino": PRETRAINED_DIR / "ViT-256" / "vit256_small_dino.pth",
}

# Ensure directories exist
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
PRETRAINED_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# Vision Transformer Implementation (Compatible with CellViT)
# ============================================================================


class PatchEmbed(nn.Module):
    """Image to Patch Embedding for ViT."""

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        return x


class Attention(nn.Module):
    """Multi-head Self Attention."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP Block."""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for classification."""

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Classification head on cls token
        return self.head(x[:, 0])


# ============================================================================
# Dataset Class
# ============================================================================


class FungiDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels: Union[List[int], np.ndarray],
        transform=None,
        augmentation_multiplier: int = 1,
    ):
        """
        Args:
            image_paths: List of paths to images
            labels: Corresponding labels
            transform: Transform to apply
            augmentation_multiplier: How many augmented versions per image (1 = no augmentation)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augmentation_multiplier = augmentation_multiplier

        # Expand dataset by creating virtual augmented copies
        if augmentation_multiplier > 1:
            self.expanded_paths = []
            self.expanded_labels = []
            for path, label in zip(image_paths, labels):
                for _ in range(augmentation_multiplier):
                    self.expanded_paths.append(path)
                    self.expanded_labels.append(label)
        else:
            self.expanded_paths = image_paths
            self.expanded_labels = labels

    def __len__(self):
        return len(self.expanded_paths)

    def __getitem__(self, idx):
        img_path = self.expanded_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.expanded_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================================
# Model Creation and Loading
# ============================================================================


def get_vit_model(
    num_classes: int, device: torch.device, pretrained_path: Optional[Path] = None
) -> nn.Module:
    """Build ViT model with optional CellViT pretrained weights."""
    # ViT-256 configuration (typical for CellViT)
    model = VisionTransformer(
        img_size=256,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
    )

    # Load pretrained weights if available
    if pretrained_path and pretrained_path.exists():
        print(f"Loading pretrained weights from {pretrained_path}")
        try:
            # Always load checkpoint on CPU first, then move to target device
            # weights_only=False is required for checkpoints containing numpy objects
            # Only use this for trusted checkpoints (CellViT, SAM, DINO from official sources)
            checkpoint = torch.load(
                pretrained_path, map_location="cpu", weights_only=False
            )

            # Handle different checkpoint formats
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Remove classification head from pretrained weights
            state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith("head")
            }

            # Load weights (strict=False to allow missing head weights)
            model.load_state_dict(state_dict, strict=False)
            print(
                "Pretrained weights loaded successfully (classification head initialized randomly)"
            )
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Training from random initialization...")
    else:
        print("No pretrained weights found. Training from random initialization...")

    # Unfreeze all layers for fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    return model.to(device)


def save_backbone_weights(model: nn.Module, save_path: Path):
    """Save ViT backbone weights without the classification head."""
    state_dict = model.state_dict()
    backbone_state = {k: v for k, v in state_dict.items() if not k.startswith("head.")}

    torch.save(backbone_state, save_path)
    print(f"Saved ViT backbone weights (without classification head) to {save_path}")


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
    use_tpu: bool = False,
    scheduler: Optional[Any] = None,  # Changed to Any to accept any scheduler
    grad_clip: float = 1.0,  # Gradient clipping threshold
) -> Dict[str, List[float]]:

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    # Epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", position=0)

    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Wrap dataloader for TPU if needed
            if use_tpu and TPU_AVAILABLE:
                loader = pl.MpDeviceLoader(dataloaders[phase], device)
            else:
                loader = dataloaders[phase]

            # Batch progress bar
            batch_pbar = tqdm(
                loader,
                desc=f"{phase:5s}",
                leave=False,
                position=1,
                disable=use_tpu,  # Disable for TPU to avoid clutter
            )

            for inputs, labels in batch_pbar:
                if not use_tpu:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()

                        # Gradient clipping to prevent exploding gradients
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip
                            )

                        if use_tpu and TPU_AVAILABLE:
                            xm.optimizer_step(optimizer)  # TPU-specific optimizer step
                        else:
                            optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update batch progress bar with current metrics
                if not use_tpu:
                    current_loss = running_loss / ((batch_pbar.n + 1) * inputs.size(0))
                    current_acc = running_corrects.double() / (
                        (batch_pbar.n + 1) * inputs.size(0)
                    )
                    batch_pbar.set_postfix(
                        {"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"}
                    )

            epoch_loss = running_loss / len(dataloaders[phase].dataset)  # type: ignore[arg-type]
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)  # type: ignore[union-attr,arg-type]

            # For TPU, sync metrics across cores
            if use_tpu and TPU_AVAILABLE:
                epoch_acc = epoch_acc.item()

            if phase == "train":
                history["loss"].append(epoch_loss)
                history["accuracy"].append(
                    epoch_acc.item() if torch.is_tensor(epoch_acc) else epoch_acc
                )

                # Get current learning rate
                current_lr = optimizer.param_groups[0]["lr"]

                epoch_pbar.set_postfix(
                    {
                        "train_loss": f"{epoch_loss:.4f}",
                        "train_acc": f"{epoch_acc:.4f}",
                        "lr": f"{current_lr:.6f}",
                    }
                )
            else:
                history["val_loss"].append(epoch_loss)
                history["val_accuracy"].append(
                    epoch_acc.item() if torch.is_tensor(epoch_acc) else epoch_acc
                )

                # Get current learning rate
                current_lr = optimizer.param_groups[0]["lr"]

                epoch_pbar.set_postfix(
                    {
                        "train_loss": f"{history['loss'][-1]:.4f}",
                        "train_acc": f"{history['accuracy'][-1]:.4f}",
                        "val_loss": f"{epoch_loss:.4f}",
                        "val_acc": f"{epoch_acc:.4f}",
                        "best": f"{best_acc:.4f}",
                        "lr": f"{current_lr:.6f}",
                    }
                )

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    tqdm.write(f"✓ New best accuracy: {best_acc:.4f}")
                else:
                    epochs_no_improve += 1

        # Step scheduler at end of epoch (after both train and val)
        if scheduler is not None:
            scheduler.step()

        if epochs_no_improve >= patience:
            tqdm.write(
                f"\n⚠ Early stopping triggered after {epochs_no_improve} epochs without improvement"
            )
            break

    epoch_pbar.close()
    tqdm.write(f"\n✓ Training complete! Best validation accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return history


# ============================================================================
# Main Training Script
# ============================================================================


def main():  # noqa: C901
    # Configuration
    BATCH_SIZE = 32  # Increased from 16 - ViT benefits from larger batches
    NUM_EPOCHS = 100  # Increased epochs for ViT
    LEARNING_RATE = 0.0005  # Reduced from 0.001 - More conservative for stability
    MIN_LR = 1e-6  # Minimum LR for cosine annealing
    WEIGHT_DECAY = 0.05  # Keep weight decay
    WARMUP_EPOCHS = 5  # Reduced from 10 - Shorter warmup
    PATIENCE = 20  # Increased patience for longer training
    GRAD_CLIP = 1.0  # Gradient clipping to prevent exploding gradients

    # Data Augmentation Configuration
    # Generate 10,000+ samples from 1,011 training images
    # Strategy: Focus on INTERNAL colony features (texture, color, patterns)
    # RandomResizedCrop extracts different regions of circular colonies
    AUGMENTATION_MULTIPLIER = 10  # Each image creates 10 augmented versions
    # Total training samples will be: 1,011 * 10 = 10,110

    # Device selection: TPU > CUDA > CPU
    use_tpu = USE_TPU and TPU_AVAILABLE
    if use_tpu:
        device = torch_xla.device()
        print(f"Using TPU device: {device}")
        print(f"TPU cores configured: {TPU_CORES}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if USE_TPU and not TPU_AVAILABLE:
            print("Warning: TPU requested but PyTorch XLA not available")
            print(
                "Install with: uv pip install torch-xla -f "
                "https://storage.googleapis.com/libtpu-wheels/index.html"
            )

    # Print paths for verification
    print("\n" + "=" * 60)
    print("CellViT ViT Training Configuration:")
    print("=" * 60)
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    print(f"HIERARCHICAL_DATASET_PATH: {HIERARCHICAL_DATASET_PATH}")
    print(f"WEIGHTS_DIR: {WEIGHTS_DIR}")
    print(f"PRETRAINED_DIR: {PRETRAINED_DIR}")
    print(f"PRETRAINED_WEIGHTS_CHOICE: {PRETRAINED_WEIGHTS_CHOICE}")
    print(f"SEGMENTED_METADATA_PATH: {SEGMENTED_METADATA_PATH}")
    print(f"STRAIN_SPECIES_MAPPING_PATH: {STRAIN_SPECIES_MAPPING_PATH}")
    print("=" * 60 + "\n")

    # Load metadata and mappings
    if not STRAIN_SPECIES_MAPPING_PATH.exists():
        print(f"Error: {STRAIN_SPECIES_MAPPING_PATH} not found.")
        return

    df_mapping = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
    if "Test" not in df_mapping.columns:
        print("Error: 'Test' column not found in mapping CSV.")
        return

    test_strain_set = set(df_mapping[df_mapping["Test"]]["Strain"].tolist())
    strain_to_species = dict(zip(df_mapping["Strain"], df_mapping["Species"]))

    print(f"Test strains: {len(test_strain_set)}")
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
        data = item.get("data", item)
        strain = data.get("strain")
        image_id = item.get("id")
        environment = data.get("environment")
        angle = data.get("angle")
        parent_id = item.get("parent_id")

        if not strain or not image_id:
            skipped_missing += 1
            continue

        species = strain_to_species.get(strain)

        if not species or species == "unknown":
            skipped_unknown += 1
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
            skipped_not_found += 1
            if skipped_not_found <= 5:
                print(f"  Warning: File not found: {image_path}")
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

    np.save(WEIGHTS_DIR / "classes_vit.npy", le.classes_)

    # Transforms with VERY STRONG augmentation optimized for circular colonies
    # Focus on INTERNAL features (texture, color, patterns) rather than shape
    data_transforms = {
        "train": transforms.Compose(
            [
                # RandomResizedCrop: Crops random regions focusing on colony interior
                # Scale 0.5-1.0 means crop 50-100% of image, then resize to 256x256
                # This captures different parts of the colony each time
                transforms.RandomResizedCrop(
                    size=(HEIGHT, WIDTH),
                    scale=(0.5, 1.0),  # Crop 50-100% of colony
                    ratio=(0.9, 1.1),  # Keep roughly square (colonies are circular)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(180),  # Full rotation - shape doesn't matter
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),  # Reduced translation (crop handles this)
                    scale=(0.9, 1.1),  # Reduced scale (crop handles this)
                    shear=5,  # Reduced shear for circular colonies
                ),
                # Strong color/texture augmentation for internal features
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.RandomGrayscale(p=0.15),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(
                    p=0.15, scale=(0.02, 0.15)
                ),  # Simulates partial occlusion
            ]
        ),
        "val": transforms.Compose(
            [
                # Validation: Center crop to focus on colony center
                transforms.Resize(int(HEIGHT * 1.1)),  # Slight resize
                transforms.CenterCrop((HEIGHT, WIDTH)),  # Center crop
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Datasets and Dataloaders with augmentation multiplier
    image_datasets = {
        "train": FungiDataset(
            train_paths,
            train_labels,  # type: ignore[arg-type]
            data_transforms["train"],
            augmentation_multiplier=AUGMENTATION_MULTIPLIER,
        ),
        "val": FungiDataset(
            val_paths,
            val_labels,  # type: ignore[arg-type]
            data_transforms["val"],
            augmentation_multiplier=1,  # No augmentation for validation
        ),
    }

    print("\nData Augmentation Applied:")
    print(f"  Original training samples: {len(train_paths)}")
    print(f"  Augmentation multiplier: {AUGMENTATION_MULTIPLIER}x")
    print(f"  Augmented training samples: {len(image_datasets['train'])}")
    print(f"  Validation samples (no augmentation): {len(image_datasets['val'])}")
    print(
        f"  Total dataset size: {len(image_datasets['train']) + len(image_datasets['val'])}"
    )
    print("\n  Augmentation Strategy:")
    print("    - RandomResizedCrop: Focus on colony interior (50-100% crop)")
    print("    - Strong color/texture augmentation for internal features")
    print("    - Full rotation (shape invariant for circular colonies)")
    print("    - Random erasing (simulates partial occlusion)\n")

    # Adjust num_workers for TPU (TPU doesn't benefit from multiple workers)
    num_workers = 0 if use_tpu else 2

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,  # Recommended for TPU
        )
        for x in ["train", "val"]
    }

    # Train ViT model
    print(f"\n{'=' * 60}")
    print("Training Vision Transformer with pretrained weights...")
    print("=" * 60)

    # Look for pretrained weights based on configuration
    pretrained_path = None
    if PRETRAINED_WEIGHTS_CHOICE:
        if PRETRAINED_WEIGHTS_CHOICE in PRETRAINED_WEIGHTS:
            candidate_path = PRETRAINED_WEIGHTS[PRETRAINED_WEIGHTS_CHOICE]
            if candidate_path.exists():
                pretrained_path = candidate_path
                print(f"\nUsing pretrained weights: {PRETRAINED_WEIGHTS_CHOICE}")
                print(f"Path: {pretrained_path}")
            else:
                print(
                    f"\nWarning: {PRETRAINED_WEIGHTS_CHOICE} weights not found at {candidate_path}"
                )
                print(
                    "Download from: https://drive.google.com/drive/folders/1zFO4bgo7yvjT9rCJi_6Mt6_07wfr0CKU?usp=sharing"
                )
                print("Training will proceed from random initialization.\n")
        else:
            print(f"\nWarning: Unknown weight choice '{PRETRAINED_WEIGHTS_CHOICE}'")
            print(f"Available options: {list(PRETRAINED_WEIGHTS.keys())}")
            print("Training will proceed from random initialization.\n")
    else:
        print(
            "\nNo pretrained weights selected. Training from random initialization.\n"
        )

    model = get_vit_model(num_classes, device, pretrained_path)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Cosine Annealing with Warmup
    # Simple and effective for ViT
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,  # Start at 10% of base LR
        end_factor=1.0,  # Reach 100% of base LR
        total_iters=WARMUP_EPOCHS,
    )

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=MIN_LR
    )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    print("Learning Rate Schedule:")
    print(f"  Base LR: {LEARNING_RATE}")
    print(
        f"  Warmup epochs: {WARMUP_EPOCHS} (linear warmup from {LEARNING_RATE * 0.1:.6f} to {LEARNING_RATE})"
    )
    print(
        f"  Cosine annealing: {NUM_EPOCHS - WARMUP_EPOCHS} epochs (decay to {MIN_LR:.6f})"
    )
    print(f"  Gradient clipping: {GRAD_CLIP}")
    print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}\n")

    history = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        device,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        use_tpu=use_tpu,
        scheduler=scheduler,
        grad_clip=GRAD_CLIP,
    )

    # Save backbone weights
    backbone_save_path = WEIGHTS_DIR / "ViT_CellViT_finetuned.pth"
    save_backbone_weights(model, backbone_save_path)

    # Save history
    history_path = WEIGHTS_DIR / "ViT_CellViT_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"Saved training history to {history_path}")

    # Visualize training history
    viz_path = WEIGHTS_DIR / "ViT_CellViT_training_history.png"
    visualize_training_history(history, "ViT-CellViT", viz_path)

    print("\n" + "=" * 60)
    print("ViT training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

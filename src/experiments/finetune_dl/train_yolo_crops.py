from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
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

from src.config import RESULTS_DIR, WEIGHTS_DIR
from src.experiments.finetune_dl.crop_dataset import (
    build_crop_dataset_summary,
    create_crop_dataset,
    default_crop_output_root,
)
from src.utils.yolo_dataset_pipeline import default_output_root


class CropClassificationDataset(Dataset):
    def __init__(self, split_root: Path, transform: transforms.Compose):
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform
        if not split_root.exists():
            return
        class_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
        for class_dir in class_dirs:
            class_id = int(class_dir.name)
            for image_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((image_path, class_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, class_id = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), class_id


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "ResNet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model

    if model_name == "MobileNetV2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.features[-1].parameters():
            param.requires_grad = True
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        return model

    if model_name == "EfficientNetB1":
        model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.features[-1].parameters():
            param.requires_grad = True
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def export_backbone_weights(
    model: nn.Module, model_name: str, output_path: Path
) -> Path:
    state_dict = model.state_dict()
    if model_name == "ResNet50":
        filtered = {
            key: value for key, value in state_dict.items() if not key.startswith("fc.")
        }
    else:
        filtered = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("classifier.")
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(filtered, output_path)
    return output_path


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> dict[str, list[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,
    )
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for _ in range(epochs):
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0
            sample_count = 0

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    predictions = outputs.argmax(dim=1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_correct += int((predictions == labels).sum().item())
                sample_count += inputs.size(0)

            epoch_loss = running_loss / max(sample_count, 1)
            epoch_accuracy = running_correct / max(sample_count, 1)
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_accuracy"].append(epoch_accuracy)

            if phase == "val" and epoch_accuracy >= best_val_acc:
                best_val_acc = epoch_accuracy
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return history


def run_crop_finetuning(
    source_dataset_root: Path | None = None,
    crop_dataset_root: Path | None = None,
    model_name: str = "ResNet50",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    image_size: int = 224,
) -> dict[str, object]:
    dataset_root = source_dataset_root or default_output_root()
    output_root = crop_dataset_root or default_crop_output_root(dataset_root)
    crop_summary = create_crop_dataset(dataset_root, output_root, crop_size=image_size)
    build_crop_dataset_summary(crop_summary)

    train_root = output_root / "train"
    val_root = output_root / "test"
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = CropClassificationDataset(train_root, transform)
    val_dataset = CropClassificationDataset(val_root, transform)

    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Crop dataset must contain both train and test samples")

    class_ids = sorted(
        {label for _, label in train_dataset.samples + val_dataset.samples}
    )
    num_classes = len(class_ids)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes).to(device)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    weights_root = WEIGHTS_DIR / "yolo_finetuned"
    results_root = RESULTS_DIR / "cross_validation_yolo" / "finetune"
    weights_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    backbone_path = export_backbone_weights(
        model,
        model_name=model_name,
        output_path=weights_root / f"{model_name}_finetuned.pth",
    )
    classifier_path = weights_root / f"{model_name}_classifier_checkpoint.pth"
    torch.save(model.state_dict(), classifier_path)

    history_path = results_root / f"{model_name}_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    summary = {
        "source_dataset_root": str(dataset_root),
        "crop_dataset_root": str(output_root),
        "model_name": model_name,
        "device": str(device),
        "num_classes": num_classes,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "backbone_weights_path": str(backbone_path),
        "classifier_checkpoint_path": str(classifier_path),
        "history_path": str(history_path),
        "train_count": len(train_dataset),
        "test_count": len(val_dataset),
    }
    summary_path = results_root / f"{model_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune extractor backbones on YOLO crop datasets"
    )
    parser.add_argument("--dataset-root", type=Path, default=default_output_root())
    parser.add_argument("--crop-dataset-root", type=Path, default=None)
    parser.add_argument(
        "--model-name",
        type=str,
        default="ResNet50",
        choices=["ResNet50", "MobileNetV2", "EfficientNetB1"],
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    result = run_crop_finetuning(
        source_dataset_root=args.dataset_root,
        crop_dataset_root=args.crop_dataset_root,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

import json
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from scipy import ndimage as ndi
from skimage.feature import hog
from skimage.filters import gabor_kernel
from torchvision.models import (
    EfficientNet_B1_Weights,
    MobileNet_V2_Weights,
    ResNet50_Weights,
    efficientnet_b1,
    mobilenet_v2,
    resnet50,
)

from src.config import WEIGHTS_DIR


def l2_normalize(features: np.ndarray) -> np.ndarray:
    """
    Apply L2 normalization to feature vector.

    Args:
        features: Feature vector as numpy array

    Returns:
        L2 normalized feature vector
    """
    norm = np.linalg.norm(features, ord=2)
    if norm == 0:
        return features
    return features / norm


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from an image.

        Args:
            image: Input image as numpy array (BGR format from cv2)

        Returns:
            Feature vector as 1D numpy array
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features extracted by this extractor.

        Returns:
            List of feature names
        """
        pass


class HOGExtractor(FeatureExtractor):
    """Histogram of Oriented Gradients (HOG) feature extractor."""

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
        target_size: Tuple[int, int] = (128, 128),
    ):
        super().__init__("hog")
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.target_size = target_size
        self._feature_dim: Optional[int] = None

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features from image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to target size
        gray = cv2.resize(gray, self.target_size)

        # Extract HOG features
        features = hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            visualize=False,
            feature_vector=True,
        )

        if self._feature_dim is None:
            self._feature_dim = len(features)

        # Apply L2 normalization
        features = l2_normalize(features)
        return features

    def get_feature_names(self) -> List[str]:
        """Get feature names for HOG."""
        if self._feature_dim is None:
            # Calculate expected dimension
            dummy = np.zeros(
                (self.target_size[0], self.target_size[1], 3), dtype=np.uint8
            )
            self.extract(dummy)

        feature_dim = self._feature_dim if self._feature_dim is not None else 0
        return [f"hog_{i}" for i in range(feature_dim)]


class GaborExtractor(FeatureExtractor):
    """Gabor filter feature extractor."""

    def __init__(
        self,
        frequencies: Optional[List[float]] = None,
        thetas: Optional[List[float]] = None,
        target_size: Tuple[int, int] = (128, 128),
    ):
        super().__init__("gabor")
        self.frequencies = frequencies or [0.1, 0.2, 0.3, 0.4]
        self.thetas = thetas or [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        self.target_size = target_size
        self._kernels: List[Any] = self._prepare_kernels()

    def _prepare_kernels(self) -> List[Any]:
        """Prepare Gabor kernels with frequencies and orientations."""
        kernels: List[Any] = []
        for freq in self.frequencies:
            for theta in self.thetas:
                kernel = np.real(gabor_kernel(freq, theta=int(theta)))  # type: ignore[arg-type]
                kernels.append(kernel)
        return kernels

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract Gabor filter features from image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to target size
        gray = cv2.resize(gray, self.target_size)

        # Apply Gabor filters and compute statistics
        features: List[float] = []
        for kernel in self._kernels:
            filtered = ndi.convolve(gray, kernel, mode="wrap")
            # Extract mean and std as features
            features.extend([filtered.mean(), filtered.std()])

        # Apply L2 normalization
        return l2_normalize(np.array(features))

    def get_feature_names(self) -> List[str]:
        """Get feature names for Gabor filters."""
        names: List[str] = []
        for i, _ in enumerate(self.frequencies):
            for j, _ in enumerate(self.thetas):
                names.append(f"gabor_f{i}_t{j}_mean")
                names.append(f"gabor_f{i}_t{j}_std")
        return names


class ColorHistogramExtractor(FeatureExtractor):
    """Color histogram feature extractor."""

    def __init__(self, bins: int = 32, target_size: Tuple[int, int] = (128, 128)):
        super().__init__("colorhistogram")
        self.bins = bins
        self.target_size = target_size

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features from image."""
        # Resize image
        resized = cv2.resize(image, self.target_size)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Compute histogram for each channel
        features: List[float] = []
        for i in range(3):  # R, G, B channels
            hist = cv2.calcHist([rgb], [i], None, [self.bins], [0, 256])
            hist = hist.flatten()
            features.extend(hist)

        # Apply L2 normalization
        return l2_normalize(np.array(features))

    def get_feature_names(self) -> List[str]:
        """Get feature names for color histogram."""
        names: List[str] = []
        for channel in ["r", "g", "b"]:
            for i in range(self.bins):
                names.append(f"hist_{channel}_{i}")
        return names


class ColorHistogramHSExtractor(FeatureExtractor):
    """Color histogram extractor using HSV (H and S channels only)."""

    def __init__(self, bins: int = 32, target_size: Tuple[int, int] = (128, 128)):
        super().__init__("colorhistogramhs")
        self.bins = bins
        self.target_size = target_size

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram from H and S channels of HSV."""
        # Resize image
        resized = cv2.resize(image, self.target_size)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Compute histogram for H and S channels only (skip V channel)
        features: List[float] = []
        for i in range(2):  # H and S channels (indices 0 and 1)
            if i == 0:  # Hue channel: range [0, 180] in OpenCV
                hist = cv2.calcHist([hsv], [i], None, [self.bins], [0, 180])
            else:  # Saturation channel: range [0, 256]
                hist = cv2.calcHist([hsv], [i], None, [self.bins], [0, 256])
            hist = hist.flatten()
            features.extend(hist)

        # Apply L2 normalization
        return l2_normalize(np.array(features))

    def get_feature_names(self) -> List[str]:
        """Get feature names for HSV histogram (H and S channels)."""
        names: List[str] = []
        for channel in ["h", "s"]:
            for i in range(self.bins):
                names.append(f"hist_{channel}_{i}")
        return names


class BaseDeepLearningExtractor(FeatureExtractor):
    """Base class for Deep Learning feature extractors using PyTorch."""

    def __init__(
        self,
        name: str,
        target_size: Tuple[int, int],
        weights_path: Optional[str] = None,
    ):
        super().__init__(name)
        self.target_size = target_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(weights_path)
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self._feature_dim = self._get_feature_dim()

    @abstractmethod
    def _build_model(self, weights_path: Optional[str]) -> nn.Module:
        pass

    def _get_feature_dim(self) -> int:
        # Run a dummy input to get output dimension
        dummy_input = torch.zeros(1, 3, *self.target_size).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
        return output.shape[1]

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract features using the deep learning model."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        input_tensor = self.preprocess(rgb).unsqueeze(0).to(self.device)  # type: ignore[union-attr]

        # Extract features
        with torch.no_grad():
            features = self.model(input_tensor)

        # Move to CPU and convert to numpy
        features_np = features.cpu().numpy().flatten()

        # Apply L2 normalization
        return l2_normalize(features_np)

    def get_feature_names(self) -> List[str]:
        return [f"{self.name.lower()}_{i}" for i in range(self._feature_dim)]


class ResNet50Extractor(BaseDeepLearningExtractor):
    """ResNet50 feature extractor."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        weights_path: Optional[str] = None,
    ):
        super().__init__("resnet50", target_size, weights_path)

    def _build_model(self, weights_path: Optional[str]) -> nn.Module:
        if weights_path and os.path.exists(weights_path):
            print(f"Loading fine-tuned ResNet50 weights from: {weights_path}")
            model = resnet50(weights=None)
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                # Backbone weights saved without 'fc.' prefix
                model.load_state_dict(state_dict, strict=False)
                print("✓ Fine-tuned ResNet50 weights loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load fine-tuned weights: {e}")
                print("Using ImageNet weights instead")
                model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove the classification head (fc layer)
        # ResNet50 structure ends with avgpool -> flatten -> fc
        # We want the output of avgpool, flattened.
        # We can replace fc with Identity
        model.fc = nn.Identity()  # type: ignore[assignment]
        return model


class MobileNetV2Extractor(BaseDeepLearningExtractor):
    """MobileNetV2 feature extractor."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        weights_path: Optional[str] = None,
    ):
        super().__init__("mobilenetv2", target_size, weights_path)

    def _build_model(self, weights_path: Optional[str]) -> nn.Module:
        if weights_path and os.path.exists(weights_path):
            print(f"Loading fine-tuned MobileNetV2 weights from: {weights_path}")
            model = mobilenet_v2(weights=None)
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                state_dict = checkpoint.get("state_dict", checkpoint)
                model.load_state_dict(state_dict, strict=False)
                print("✓ Fine-tuned weights loaded")
            except Exception as e:
                print(f"Warning: Failed to load fine-tuned weights: {e}")
                model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        else:
            model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # MobileNetV2 classifier is a Sequential block.
        # The last layer is a Linear layer.
        # We want features from the pooling layer.
        # MobileNetV2 structure: features -> adaptive_avg_pool2d
        # -> classifier
        # We can just use model.features and add pooling.

        class MobileNetV2Features(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.features = original_model.features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.features(x)
                x = self.pool(x)
                x = torch.flatten(x, 1)
                return x

        return MobileNetV2Features(model)


class EfficientNetB1Extractor(BaseDeepLearningExtractor):
    """EfficientNet B1 feature extractor with ImageNet weights."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        weights_path: Optional[str] = None,
    ):
        super().__init__("efficientnetb1", target_size, weights_path)

    def _build_model(self, weights_path: Optional[str]) -> nn.Module:
        if weights_path and os.path.exists(weights_path):
            print(f"Loading fine-tuned EfficientNetB1 weights from: {weights_path}")
            model = efficientnet_b1(weights=None)
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                # Backbone weights saved without 'classifier.' prefix
                model.load_state_dict(state_dict, strict=False)
                print("✓ Fine-tuned EfficientNetB1 weights loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load weights: {e}")
                print("Using ImageNet pretrained weights instead")
                model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        else:
            model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)

        # Remove classifier to extract features
        # EfficientNet classifier is a Sequential, replace with Identity
        model.classifier = nn.Sequential(nn.Identity())
        return model


class ColorHistogramHSconcatResnet50(FeatureExtractor):
    """
    Concatenated feature extractor combining ColorHistogramHS and
    ResNet50. ColorHistogramHS features are weighted more heavily (3x)
    before concatenation.
    """

    def __init__(
        self,
        hist_weight: float = 3.0,
        bins: int = 32,
        hist_target_size: Tuple[int, int] = (128, 128),
        resnet_target_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__("colorhistogramhsconcatresnet50")
        self.hist_weight = hist_weight
        self.hist_extractor = ColorHistogramHSExtractor(
            bins=bins, target_size=hist_target_size
        )
        self.resnet_extractor = ResNet50Extractor(target_size=resnet_target_size)
        self._feature_dim = 64 + 2048

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract concatenated features with weighted ColorHistogramHS."""
        hist_features = self.hist_extractor.extract(image)
        resnet_features = self.resnet_extractor.extract(image)

        weighted_hist = hist_features * self.hist_weight
        concat_features = np.concatenate([weighted_hist, resnet_features])

        return l2_normalize(concat_features)

    def get_feature_names(self) -> List[str]:
        hist_names = [
            f"weighted_hist_{name}" for name in self.hist_extractor.get_feature_names()
        ]
        resnet_names = self.resnet_extractor.get_feature_names()
        return hist_names + resnet_names


def extract_features_from_dataset(
    segmented_image_path: str,
    metadata_path: str,
    output_json_path: str,
    extractors: List[FeatureExtractor],
) -> List[dict[str, Any]]:
    """
    Extract features from all segmented images and save to JSON.
    """
    with open(metadata_path, "r") as f:
        metadata_list = json.load(f)

    print(f"Found {len(metadata_list)} images in metadata")
    print(
        f"Applying {len(extractors)} feature extractors: {[e.name for e in extractors]}"
    )

    results: List[dict[str, Any]] = []

    for idx, metadata in enumerate(metadata_list):
        image_id = metadata["id"]
        image_path = os.path.join(segmented_image_path, f"{image_id}.jpg")

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping...")
            continue

        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            print(f"Warning: Failed to read {image_path}, skipping...")
            continue

        feature_data: dict[str, Any] = {"id": image_id, "features": {}}

        try:
            for extractor in extractors:
                features = extractor.extract(image)
                feature_data["features"][extractor.name.lower()] = {
                    "vector": features.tolist(),
                    "dimension": len(features),
                }

            results.append(feature_data)

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(metadata_list)} images...")

        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            continue

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    total_features = 0
    if results:
        total_features = sum(
            feat["dimension"] for feat in results[0]["features"].values()
        )

    print("\nFeature extraction complete!")
    print(f"Processed {len(results)} images")
    print(f"Feature types: {list(results[0]['features'].keys()) if results else []}")
    print(f"Total feature dimension: {total_features}")
    print(f"Results saved to: {output_json_path}")

    return results


# Fine-tuned extractor classes that point to the fine-tuned vectors in Qdrant
class ResNet50FinetunedExtractor(ResNet50Extractor):
    """ResNet50 extractor that uses fine-tuned weights and points to fine-tuned vectors in Qdrant."""

    def __init__(
        self,
        weights_path: Optional[str] = str(WEIGHTS_DIR / "ResNet50_finetuned.pth"),
    ):
        super().__init__(weights_path=weights_path)
        # Override name to match the vector name in Qdrant
        self.name = "ResNet50_finetuned"


class MobileNetV2FinetunedExtractor(MobileNetV2Extractor):
    """MobileNetV2 extractor that uses fine-tuned weights and points to fine-tuned vectors in Qdrant."""

    def __init__(
        self,
        weights_path: Optional[str] = str(WEIGHTS_DIR / "MobileNetV2_finetuned.pth"),
    ):
        super().__init__(weights_path=weights_path)
        # Override name to match the vector name in Qdrant
        self.name = "MobileNetV2_finetuned"


class EfficientNetB1FinetunedExtractor(EfficientNetB1Extractor):
    """EfficientNetB1 extractor that uses fine-tuned weights and points to fine-tuned vectors in Qdrant."""

    def __init__(
        self,
        weights_path: Optional[str] = str(WEIGHTS_DIR / "EfficientNetB1_finetuned.pth"),
    ):
        super().__init__(weights_path=weights_path)
        # Override name to match the vector name in Qdrant
        self.name = "EfficientNetB1_finetuned"


class EfficientNetB1TripletExtractor(EfficientNetB1Extractor):
    """EfficientNetB1 extractor that uses triplet loss fine-tuned weights."""

    def __init__(
        self,
        weights_path: Optional[str] = str(WEIGHTS_DIR / "EfficientNetB1_triplet.pth"),
    ):
        super().__init__(weights_path=weights_path)
        self.name = "efficientnetb1_triplet"

    def _build_model(self, weights_path: Optional[str]) -> nn.Module:
        if weights_path and os.path.exists(weights_path):
            print(
                f"Loading fine-tuned EfficientNetB1 Triplet weights from: {weights_path}"
            )
            model = efficientnet_b1(weights=None)
            try:
                state_dict = torch.load(weights_path, map_location=self.device)

                # Filter out classifier weights because of dimension mismatch
                # (128 vs 1000)
                filtered_state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if not k.startswith("classifier.")
                }

                model.load_state_dict(filtered_state_dict, strict=False)
                print(
                    "✓ Fine-tuned EfficientNetB1 Triplet weights loaded successfully (classifier excluded)"
                )
            except Exception as e:
                print(f"Warning: Failed to load weights: {e}")
                print("Using ImageNet pretrained weights instead")
                model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        else:
            model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)

        # Remove classifier to extract features
        model.classifier = nn.Sequential(nn.Identity())
        return model


# ============================================================================
# Vision Transformer (ViT) Implementation for CellViT/SAM/ViT-256
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
        x = self.proj(x).flatten(2).transpose(1, 2)
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
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0
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
    """Vision Transformer for feature extraction."""

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

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

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # Return cls token


class ViTExtractor(BaseDeepLearningExtractor):
    """Vision Transformer feature extractor with support for multiple pretrained weights."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        weights_type: str = "vit256_dino",
        target_size: Tuple[int, int] = (256, 256),
    ):
        """
        Args:
            weights_path: Path to pretrained weights file
            weights_type: Type of pretrained weights (cellvit_x20, cellvit_x40, sam_vit_b, sam_vit_l, sam_vit_h, vit256_dino)
            target_size: Target image size
        """
        self.weights_type = weights_type
        super().__init__(f"vit_{weights_type}", target_size, weights_path)

    def _build_model(self, weights_path: Optional[str]) -> nn.Module:
        # ViT-256 configuration (standard for CellViT/SAM/ViT-256)
        model = VisionTransformer(
            img_size=self.target_size[0],
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
        )

        if weights_path and os.path.exists(weights_path):
            print(f"Loading ViT weights from: {weights_path}")
            try:
                checkpoint = torch.load(
                    weights_path, map_location="cpu", weights_only=False
                )

                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

                # Remove classification head if present
                state_dict = {
                    k: v for k, v in state_dict.items() if not k.startswith("head")
                }

                model.load_state_dict(state_dict, strict=False)
                print(f"✓ ViT weights loaded successfully ({self.weights_type})")
            except Exception as e:
                print(f"Warning: Failed to load ViT weights: {e}")
                print("Using random initialization")
        else:
            print(
                f"No pretrained weights found at {weights_path}. Using random initialization."
            )

        return model


class ViTCellVitX20Extractor(ViTExtractor):
    """ViT with CellViT-256-x20 pretrained weights."""

    def __init__(self, weights_path: str = "pretrained/CellViT/CellViT-256-x20.pth"):
        super().__init__(weights_path=weights_path, weights_type="cellvit_x20")


class ViTCellVitX40Extractor(ViTExtractor):
    """ViT with CellViT-256-x40 pretrained weights."""

    def __init__(self, weights_path: str = "pretrained/CellViT/CellViT-256-x40.pth"):
        super().__init__(weights_path=weights_path, weights_type="cellvit_x40")


class ViTSAMBExtractor(ViTExtractor):
    """ViT with SAM Base (encoder-only) pretrained weights."""

    def __init__(self, weights_path: str = "pretrained/SAM/sam_vit_b.pth"):
        super().__init__(weights_path=weights_path, weights_type="sam_vit_b")


class ViTSAMLExtractor(ViTExtractor):
    """ViT with SAM Large (encoder-only) pretrained weights."""

    def __init__(self, weights_path: str = "pretrained/SAM/sam_vit_l.pth"):
        super().__init__(weights_path=weights_path, weights_type="sam_vit_l")


class ViTSAMHExtractor(ViTExtractor):
    """ViT with SAM Huge (encoder-only) pretrained weights."""

    def __init__(self, weights_path: str = "pretrained/SAM/sam_vit_h.pth"):
        super().__init__(weights_path=weights_path, weights_type="sam_vit_h")


class ViT256DinoExtractor(ViTExtractor):
    """ViT with ViT-256 DINO self-supervised pretrained weights."""

    def __init__(self, weights_path: str = "pretrained/ViT-256/vit256_small_dino.pth"):
        super().__init__(weights_path=weights_path, weights_type="vit256_dino")


class ViTFinetunedExtractor(ViTExtractor):
    """ViT extractor that uses fine-tuned weights and points to fine-tuned vectors in Qdrant."""

    def __init__(
        self,
        weights_path: str = str(WEIGHTS_DIR / "ViT_CellViT_finetuned.pth"),
        weights_type: str = "vit256_dino",
    ):
        super().__init__(weights_path=weights_path, weights_type=weights_type)
        # Override name to match the vector name in Qdrant
        self.name = "ViT_finetuned"

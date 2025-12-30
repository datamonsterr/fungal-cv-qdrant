"""
Training script for fine-tuning deep learning models on segmented myco fungi images.
Trains models to classify species, excluding one test strain per species.
"""
import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetV2B0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetv2_preprocess
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Dataset paths
SEGMENTED_IMAGE_DIR = "../Dataset/segmented_image"
SEGMENTED_METADATA_PATH = "../Dataset/segmented_image_metadata.json"
STRAIN_SPECIES_MAPPING_PATH = "../Dataset/strain_to_specy.csv"
WEIGHTS_BASE_DIR = "./weights"


def load_strain_to_species_mapping(csv_path: str) -> Dict[str, str]:
    """Load strain to species mapping from CSV."""
    df = pd.read_csv(csv_path)
    return dict(zip(df['Strain'], df['Species']))


def select_test_strains(
    available_strains: List[str],
    strain_to_specy: Dict[str, str]
) -> Dict[str, str]:
    """
    Select one strain per species for testing (same logic as evaluate_species.py).
    
    Args:
        available_strains: List of all available strains
        strain_to_specy: Mapping from strain to species
        
    Returns:
        Dictionary mapping species to selected test strain
    """
    species_to_strains = defaultdict(list)
    
    # Group available strains by species
    for strain in available_strains:
        if strain in strain_to_specy:
            species = strain_to_specy[strain]
            species_to_strains[species].append(strain)
    
    # Select one strain per species for testing
    test_strains = {}
    for species, strains in species_to_strains.items():
        if len(strains) > 1:
            test_strains[species] = strains[1]  # Take second strain
        else:
            test_strains[species] = strains[0]  # Take only available strain
    
    return test_strains


def load_dataset(
    metadata_path: str,
    image_dir: str,
    strain_to_specy: Dict[str, str],
    test_strains: Dict[str, str]
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load dataset and split into train/test based on test_strains.
    
    Args:
        metadata_path: Path to segmented image metadata JSON
        image_dir: Directory containing segmented images
        strain_to_specy: Mapping from strain to species
        test_strains: Dictionary mapping species to test strain
        
    Returns:
        Tuple of (train_image_paths, train_labels, test_image_paths, test_labels)
    """
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
    
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    # Create reverse mapping: test_strain -> species
    test_strain_set = set(test_strains.values())
    
    for item in metadata_list:
        image_id = item['id']
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        
        if not os.path.exists(image_path):
            continue
        
        strain = item['data']['strain']
        species = item['data']['specy']
        
        # Skip unknown species
        if species == 'unknown' or strain == 'unknown':
            continue
        
        # Assign to test set if this strain is selected as test strain
        if strain in test_strain_set:
            test_images.append(image_path)
            test_labels.append(species)
        else:
            train_images.append(image_path)
            train_labels.append(species)
    
    return train_images, train_labels, test_images, test_labels


def create_data_generators(
    train_images: List[str],
    train_labels: List[str],
    val_images: List[str],
    val_labels: List[str],
    label_encoder: LabelEncoder,
    preprocess_fn,
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224),
    augment: bool = True
):
    """
    Create TensorFlow data generators for training and validation.
    
    Args:
        train_images: List of training image paths
        train_labels: List of training labels
        val_images: List of validation image paths
        val_labels: List of validation labels
        label_encoder: Fitted label encoder
        preprocess_fn: Preprocessing function for the model
        batch_size: Batch size
        target_size: Target image size (height, width)
        augment: Whether to apply data augmentation
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    def load_and_preprocess_image(image_path, label):
        # Read image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size)
        
        # Apply preprocessing
        img = preprocess_fn(img)
        
        return img, label
    
    def augment_image(img, label):
        # Random flip
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        
        # Random rotation (via transpose and flip)
        if tf.random.uniform(()) > 0.5:
            img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Random brightness
        img = tf.image.random_brightness(img, 0.2)
        
        # Random contrast
        img = tf.image.random_contrast(img, 0.8, 1.2)
        
        return img, label
    
    # Encode labels
    train_labels_encoded = label_encoder.transform(train_labels)
    val_labels_encoded = label_encoder.transform(val_labels)
    
    # Create train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels_encoded))
    train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels_encoded))
    val_dataset = val_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset


def build_model(
    base_model_name: str,
    num_classes: int,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    freeze_base: bool = True
) -> Model:
    """
    Build a classification model with a pre-trained base.
    
    Args:
        base_model_name: Name of base model ('resnet50', 'mobilenetv2', 'efficientnetv2b0')
        num_classes: Number of species classes
        input_shape: Input image shape
        freeze_base: Whether to freeze base model layers
        
    Returns:
        Keras Model
    """
    # Create base model
    if base_model_name.lower() == 'resnet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name.lower() == 'mobilenetv2':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name.lower() == 'efficientnetv2b0':
        base_model = EfficientNetV2B0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze base model if requested
    base_model.trainable = not freeze_base
    
    # Build classification head
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name=f"{base_model_name}_classifier")
    
    return model


def plot_training_history(history, model_name: str, output_dir: str):
    """
    Plot and save training history charts.
    
    Args:
        history: Keras training history object
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'{model_name} - Training and Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Val Loss', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'{model_name} - Training and Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{model_name}_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to: {plot_path}")


def save_training_report(
    history,
    model_name: str,
    output_dir: str,
    train_size: int,
    val_size: int,
    test_size: int,
    num_classes: int,
    label_encoder: LabelEncoder,
    test_strains: Dict[str, str]
):
    """
    Save detailed training report.
    
    Args:
        history: Keras training history
        model_name: Name of the model
        output_dir: Directory to save report
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        num_classes: Number of classes
        label_encoder: Fitted label encoder
        test_strains: Dictionary of test strains
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f'{model_name}_training_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"TRAINING REPORT - {model_name}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_name}\n\n")
        
        f.write("Dataset Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training samples: {train_size}\n")
        f.write(f"Validation samples: {val_size}\n")
        f.write(f"Test samples: {test_size}\n")
        f.write(f"Number of species classes: {num_classes}\n\n")
        
        f.write("Species Classes:\n")
        f.write("-" * 80 + "\n")
        for i, species in enumerate(label_encoder.classes_):
            f.write(f"{i:3d}: {species}\n")
        f.write("\n")
        
        f.write("Test Strains (Excluded from Training):\n")
        f.write("-" * 80 + "\n")
        for species, strain in sorted(test_strains.items()):
            f.write(f"{species:<40} -> {strain}\n")
        f.write("\n")
        
        f.write("Training Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total epochs: {len(history.history['loss'])}\n")
        f.write(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
        f.write(f"Final training loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final validation loss: {history.history['val_loss'][-1]:.4f}\n")
        f.write(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}\n")
        f.write(f"Best validation loss: {min(history.history['val_loss']):.4f}\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Training report saved to: {report_path}")


def train_model(
    model_name: str,
    train_images: List[str],
    train_labels: List[str],
    test_images: List[str],
    test_labels: List[str],
    label_encoder: LabelEncoder,
    test_strains: Dict[str, str],
    batch_size: int = 32,
    epochs: int = 100,
    initial_learning_rate: float = 0.001,
    target_size: Tuple[int, int] = (224, 224)
):
    """
    Train a single model.
    
    Args:
        model_name: Name of the model to train
        train_images: Training image paths
        train_labels: Training labels
        test_images: Test image paths
        test_labels: Test labels
        label_encoder: Fitted label encoder
        test_strains: Dictionary of test strains
        batch_size: Batch size
        epochs: Number of epochs
        initial_learning_rate: Initial learning rate
        target_size: Target image size
    """
    print("\n" + "=" * 80)
    print(f"Training {model_name}")
    print("=" * 80)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(WEIGHTS_BASE_DIR, model_name.lower(), timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Split training data into train and validation
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        train_images, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )
    
    print(f"Train samples: {len(train_imgs)}")
    print(f"Validation samples: {len(val_imgs)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Get preprocessing function
    if model_name.lower() == 'resnet50':
        preprocess_fn = resnet_preprocess
    elif model_name.lower() == 'mobilenetv2':
        preprocess_fn = mobilenetv2_preprocess
    elif model_name.lower() == 'efficientnetv2b0':
        preprocess_fn = efficientnetv2_preprocess
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create data generators
    train_dataset, val_dataset = create_data_generators(
        train_imgs, train_lbls, val_imgs, val_lbls,
        label_encoder, preprocess_fn, batch_size, target_size
    )
    
    # Build model
    num_classes = len(label_encoder.classes_)
    model = build_model(
        model_name,
        num_classes,
        input_shape=(*target_size, 3),
        freeze_base=False  # CHANGED: Unfreeze base model for true fine-tuning!
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    checkpoint_path = os.path.join(output_dir, 'best_model.h5')
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model weights
    final_weights_path = os.path.join(output_dir, 'final_weights.h5')
    model.save_weights(final_weights_path)
    print(f"\nFinal weights saved to: {final_weights_path}")
    
    # Save label encoder
    label_encoder_path = os.path.join(output_dir, 'label_encoder.npy')
    np.save(label_encoder_path, label_encoder.classes_)
    print(f"Label encoder saved to: {label_encoder_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Plot training history
    plot_training_history(history, model_name, output_dir)
    
    # Save training report
    save_training_report(
        history, model_name, output_dir,
        len(train_imgs), len(val_imgs), len(test_images),
        num_classes, label_encoder, test_strains
    )
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'train_size': len(train_imgs),
        'val_size': len(val_imgs),
        'test_size': len(test_images),
        'num_classes': num_classes,
        'batch_size': batch_size,
        'epochs': len(history.history['loss']),
        'initial_learning_rate': initial_learning_rate,
        'target_size': target_size,
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'checkpoint_path': checkpoint_path,
        'final_weights_path': final_weights_path,
        'label_encoder_path': label_encoder_path
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    print(f"\n{model_name} training complete!")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"All files saved to: {output_dir}")
    
    return model, history, output_dir


def main():
    """Main training function."""
    print("=" * 80)
    print("MYCO FUNGI SPECIES CLASSIFICATION - MODEL TRAINING")
    print("=" * 80)
    
    # Load strain to species mapping
    print("\nLoading strain to species mapping...")
    strain_to_specy = load_strain_to_species_mapping(STRAIN_SPECIES_MAPPING_PATH)
    print(f"Loaded {len(strain_to_specy)} strain-to-species mappings")
    
    # Load metadata to get available strains
    print("\nLoading image metadata...")
    with open(SEGMENTED_METADATA_PATH, 'r') as f:
        metadata_list = json.load(f)
    
    available_strains = list(set([
        item['data']['strain'] 
        for item in metadata_list 
        if item['data']['strain'] != 'unknown' and item['data']['specy'] != 'unknown'
    ]))
    print(f"Found {len(available_strains)} unique strains in dataset")
    
    # Select test strains (one per species)
    print("\nSelecting test strains (one per species)...")
    test_strains = select_test_strains(available_strains, strain_to_specy)
    print(f"Selected {len(test_strains)} test strains")
    
    # Load dataset
    print("\nLoading and splitting dataset...")
    train_images, train_labels, test_images, test_labels = load_dataset(
        SEGMENTED_METADATA_PATH,
        SEGMENTED_IMAGE_DIR,
        strain_to_specy,
        test_strains
    )
    
    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    
    # Get unique species
    unique_species = sorted(set(train_labels + test_labels))
    print(f"Number of species: {len(unique_species)}")
    
    # Create and fit label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_species)
    
    # Training configuration
    models_to_train = ['ResNet50', 'MobileNetV2', 'EfficientNetV2B0']
    batch_size = 32
    epochs = 30
    initial_learning_rate = 0.001
    target_size = (224, 224)
    
    # Train each model
    results = {}
    for model_name in models_to_train:
        try:
            model, history, output_dir = train_model(
                model_name=model_name,
                train_images=train_images,
                train_labels=train_labels,
                test_images=test_images,
                test_labels=test_labels,
                label_encoder=label_encoder,
                test_strains=test_strains,
                batch_size=batch_size,
                epochs=epochs,
                initial_learning_rate=initial_learning_rate,
                target_size=target_size
            )
            results[model_name] = {
                'model': model,
                'history': history,
                'output_dir': output_dir
            }
        except Exception as e:
            print(f"\nError training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)
    for model_name, result in results.items():
        history = result['history']
        output_dir = result['output_dir']
        best_val_acc = max(history.history['val_accuracy'])
        print(f"\n{model_name}:")
        print(f"  Best validation accuracy: {best_val_acc:.4f}")
        print(f"  Weights saved to: {output_dir}")
    
    print("\n" + "=" * 80)
    print("All training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

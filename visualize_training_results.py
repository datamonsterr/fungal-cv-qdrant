"""
Visualize training results from previous training runs.
Shows training loss and accuracy charts for all trained models.
"""
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

WEIGHTS_BASE_DIR = "./weights"


def find_all_training_runs():
    """Find all training run directories with training history."""
    training_runs = []
    
    if not os.path.exists(WEIGHTS_BASE_DIR):
        return training_runs
    
    for model_name in os.listdir(WEIGHTS_BASE_DIR):
        model_dir = os.path.join(WEIGHTS_BASE_DIR, model_name)
        if not os.path.isdir(model_dir):
            continue
        
        for timestamp in os.listdir(model_dir):
            run_dir = os.path.join(model_dir, timestamp)
            if not os.path.isdir(run_dir):
                continue
            
            history_path = os.path.join(run_dir, 'training_history.json')
            if os.path.exists(history_path):
                training_runs.append({
                    'model_name': model_name,
                    'timestamp': timestamp,
                    'run_dir': run_dir,
                    'history_path': history_path
                })
    
    return sorted(training_runs, key=lambda x: (x['model_name'], x['timestamp']))


def plot_training_history_from_json(history_path: str, model_name: str, timestamp: str, output_path: str = None):
    """
    Plot training history from saved JSON file.
    
    Args:
        history_path: Path to training_history.json
        model_name: Name of the model
        timestamp: Timestamp of training run
        output_path: Optional path to save the plot
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Plot accuracy
    axes[0].plot(epochs, history['accuracy'], label='Train Accuracy', marker='o', linewidth=2)
    axes[0].plot(epochs, history['val_accuracy'], label='Val Accuracy', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title(f'{model_name} - Training and Validation Accuracy\n{timestamp}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Add annotations for best values
    best_train_acc = max(history['accuracy'])
    best_val_acc = max(history['val_accuracy'])
    best_train_epoch = history['accuracy'].index(best_train_acc) + 1
    best_val_epoch = history['val_accuracy'].index(best_val_acc) + 1
    
    axes[0].annotate(f'Best: {best_val_acc:.4f}\nEpoch: {best_val_epoch}',
                    xy=(best_val_epoch, best_val_acc),
                    xytext=(10, -30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot loss
    axes[1].plot(epochs, history['loss'], label='Train Loss', marker='o', linewidth=2)
    axes[1].plot(epochs, history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title(f'{model_name} - Training and Validation Loss\n{timestamp}', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Add annotations for best values
    best_train_loss = min(history['loss'])
    best_val_loss = min(history['val_loss'])
    best_train_loss_epoch = history['loss'].index(best_train_loss) + 1
    best_val_loss_epoch = history['val_loss'].index(best_val_loss) + 1
    
    axes[1].annotate(f'Best: {best_val_loss:.4f}\nEpoch: {best_val_loss_epoch}',
                    xy=(best_val_loss_epoch, best_val_loss),
                    xytext=(10, 30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return history


def compare_all_models():
    """Create comparison plots for all trained models."""
    training_runs = find_all_training_runs()
    
    if not training_runs:
        print("No training runs found in ./weights/")
        return
    
    print("=" * 80)
    print("TRAINING RESULTS VISUALIZATION")
    print("=" * 80)
    print(f"\nFound {len(training_runs)} training run(s):\n")
    
    all_histories = []
    
    for i, run in enumerate(training_runs, 1):
        print(f"{i}. {run['model_name']} - {run['timestamp']}")
        
        # Load history
        with open(run['history_path'], 'r') as f:
            history = json.load(f)
        
        # Print summary
        final_train_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        best_val_acc = max(history['val_accuracy'])
        best_val_acc_epoch = history['val_accuracy'].index(best_val_acc) + 1
        total_epochs = len(history['accuracy'])
        
        print(f"   Epochs: {total_epochs}")
        print(f"   Final Train Acc: {final_train_acc:.4f}")
        print(f"   Final Val Acc:   {final_val_acc:.4f}")
        print(f"   Best Val Acc:    {best_val_acc:.4f} (epoch {best_val_acc_epoch})")
        print()
        
        all_histories.append({
            'model_name': run['model_name'],
            'timestamp': run['timestamp'],
            'history': history
        })
        
        # Generate individual plot
        output_path = os.path.join(run['run_dir'], f"{run['model_name']}_training_history_detailed.png")
        plot_training_history_from_json(
            run['history_path'],
            run['model_name'],
            run['timestamp'],
            output_path
        )
    
    # Create comparison plot if multiple models
    if len(all_histories) > 1:
        print("\n" + "=" * 80)
        print("Creating comparison plot...")
        print("=" * 80)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = plt.cm.tab10(range(len(all_histories)))
        
        for i, run_data in enumerate(all_histories):
            model_name = run_data['model_name']
            timestamp = run_data['timestamp']
            history = run_data['history']
            epochs = range(1, len(history['accuracy']) + 1)
            
            label = f"{model_name} ({timestamp})"
            
            # Plot accuracy
            axes[0].plot(epochs, history['val_accuracy'], label=label, 
                        marker='o', linewidth=2, color=colors[i], markersize=4)
        
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Validation Accuracy', fontsize=12)
        axes[0].set_title('Model Comparison - Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        for i, run_data in enumerate(all_histories):
            model_name = run_data['model_name']
            timestamp = run_data['timestamp']
            history = run_data['history']
            epochs = range(1, len(history['loss']) + 1)
            
            label = f"{model_name} ({timestamp})"
            
            # Plot loss
            axes[1].plot(epochs, history['val_loss'], label=label,
                        marker='s', linewidth=2, color=colors[i], markersize=4)
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Validation Loss', fontsize=12)
        axes[1].set_title('Model Comparison - Validation Loss', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        comparison_path = os.path.join(WEIGHTS_BASE_DIR, 'model_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {comparison_path}")
        plt.close()
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated charts saved in weights/ directory")
    print("\nTo view charts:")
    print("  Linux: xdg-open weights/model_comparison.png")
    print("  macOS: open weights/model_comparison.png")
    print("=" * 80 + "\n")


def main():
    """Main function."""
    compare_all_models()


if __name__ == "__main__":
    main()

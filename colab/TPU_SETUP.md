# Running ViT Training on TPU v5e

## Setup Instructions for Google Colab

### 1. Create TPU Runtime
In Colab, go to: **Runtime > Change runtime type > TPU v5e**

### 2. Install PyTorch XLA
```bash
!pip install torch-xla -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### 3. Verify TPU Access
```python
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print(f"TPU device: {device}")
```

### 4. Configure Training Script
In `train_models_cellvit.py`, set:
```python
USE_TPU = True
TPU_CORES = 1  # Use 1 for single-core or 8 for multi-core training
```

### 5. Run Training
```bash
!python train_models_cellvit.py
```

## TPU-Specific Optimizations

### Batch Size
- TPUs work best with larger batch sizes (32-128)
- v5e has 16GB memory per core, so you can increase batch size significantly

### Data Loading
- Set `num_workers=0` for TPU (handled automatically in the script)
- Use `drop_last=True` to avoid uneven batches
- Consider using `persistent_workers=False`

### Model Checkpointing
When saving/loading on TPU, use:
```python
# Save
xm.save(model.state_dict(), 'model.pth')

# Load
state_dict = torch.load('model.pth')
model.load_state_dict(state_dict)
```

## Multi-Core Training (Advanced)

To use all 8 cores on v5e-8:
```python
TPU_CORES = 8
```

This will automatically distribute training across all cores using data parallelism.

## Troubleshooting

### Out of Memory
- Reduce batch size
- Reduce model size or image resolution
- Enable gradient checkpointing

### Slow Training
- Increase batch size (TPUs are optimized for larger batches)
- Ensure data preprocessing isn't a bottleneck
- Check that TPU is actually being used: `print(device)`

### Installation Issues
If torch-xla installation fails, try:
```bash
# Use the stable release
!pip install torch-xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

# Or check version compatibility
!python -c "import torch; print(torch.__version__)"
```

## Performance Comparison

Expected speedup on v5e vs CPU:
- Single core TPU: 5-10x faster than CPU
- 8-core TPU: 30-50x faster than CPU
- Comparable to or faster than single GPU (depending on GPU model)

## Cost Optimization

- Use TPU preemptible instances for lower cost
- Stop runtime when not training
- Consider batch prediction on TPU for inference

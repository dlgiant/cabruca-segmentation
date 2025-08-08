# Advanced Training Pipeline for Cabruca Segmentation

## Overview
Production-ready training infrastructure with experiment tracking, memory-efficient training for macOS, and comprehensive monitoring capabilities.

## Key Features

### 🚀 Training Capabilities
- **Multi-device Support**: CPU, CUDA GPU, Apple Metal (MPS)
- **Memory Efficiency**: Gradient accumulation, mixed precision training
- **Advanced Schedulers**: Cosine, OneCycle, Plateau, Warm Restarts
- **Early Stopping**: Automatic training termination on plateau
- **Checkpoint Management**: Save/resume with full state preservation

### 📊 Experiment Tracking
- **MLflow**: Local or remote tracking server
- **Weights & Biases**: Cloud-based experiment management
- **TensorBoard**: Real-time metrics visualization
- **Custom Dashboard**: Streamlit-based monitoring interface

### 💾 Memory Optimization
- **Gradient Accumulation**: Simulate larger batches
- **Mixed Precision**: FP16 training (CUDA only)
- **Dynamic Memory Management**: Automatic cache clearing
- **MPS Support**: Native Apple Silicon acceleration

## Quick Start

### Installation
```bash
# Install training dependencies
pip install -r requirements_training.txt

# Optional: Install MLflow
pip install mlflow

# Optional: Install Weights & Biases
pip install wandb
wandb login
```

### Basic Training
```bash
# Train with default configuration
python train.py --config configs/training_config.yaml

# Train on CPU with lightweight config
python train.py --config configs/training_config_light.yaml --device cpu

# Train on Apple Silicon (M1/M2)
python train.py --config configs/training_config.yaml --device mps
```

### Resume Training
```bash
# Resume from checkpoint
python train.py --config configs/training_config.yaml \
    --resume outputs/cabruca_20240108_120000/checkpoints/checkpoint_latest.pth
```

### Experiment Tracking
```bash
# Enable MLflow tracking
python train.py --config configs/training_config.yaml --use-mlflow

# Enable Weights & Biases
python train.py --config configs/training_config.yaml --use-wandb

# Enable both
python train.py --config configs/training_config.yaml --use-mlflow --use-wandb
```

### Hyperparameter Override
```bash
# Override learning rate and batch size
python train.py --config configs/training_config.yaml \
    --lr 0.001 --batch-size 8 --epochs 50

# Enable gradient accumulation for larger effective batch size
python train.py --config configs/training_config.yaml \
    --batch-size 2 --grad-accum 4  # Effective batch size = 8
```

## Configuration

### Training Configuration (YAML)
```yaml
model:
  num_instance_classes: 3
  num_semantic_classes: 6

data:
  batch_size: 4
  tile_size: 512
  num_workers: 4

optimizer:
  type: "adamw"
  lr: 0.0001
  weight_decay: 0.01

scheduler:
  type: "cosine"
  T_max: 50
  eta_min: 1e-7

training:
  num_epochs: 100
  gradient_accumulation_steps: 2
  gradient_clip: 1.0
  mixed_precision: false
  monitor_metric: "val/mAP_50"
  minimize_metric: false
  early_stopping_patience: 20

tracking:
  tensorboard: true
  mlflow: false
  wandb: false
```

### Memory-Efficient Settings for macOS
```yaml
# For M1/M2 Macs with limited memory
data:
  batch_size: 2
  tile_size: 256
  num_workers: 2

training:
  gradient_accumulation_steps: 4  # Effective batch = 8
  mixed_precision: false  # Not supported on MPS yet
  empty_cache_freq: 10

device: "mps"  # Use Metal Performance Shaders
```

## Monitoring

### Real-time Dashboard
```bash
# Launch Streamlit dashboard
python monitor.py

# Or directly
streamlit run src/training/training_monitor.py
```

Dashboard features:
- Live training curves
- System resource monitoring
- Checkpoint management
- Metric comparisons
- Configuration viewer

### TensorBoard
```bash
# Launch TensorBoard
tensorboard --logdir outputs/experiment_name/tensorboard
```

### MLflow UI
```bash
# Launch MLflow UI
mlflow ui --backend-store-uri file:./mlruns
```

## Training Strategies

### 1. Progressive Training
Start with smaller images and gradually increase resolution:
```bash
# Stage 1: Low resolution
python train.py --config configs/training_config.yaml \
    --experiment-name stage1_256 \
    --epochs 30

# Stage 2: High resolution
python train.py --config configs/training_config.yaml \
    --resume outputs/stage1_256/checkpoints/checkpoint_best.pth \
    --experiment-name stage2_512 \
    --lr 0.00001
```

### 2. Transfer Learning
```yaml
model:
  pretrained_checkpoint: "path/to/pretrained_model.pth"

optimizer:
  param_groups:
    - name: "backbone"
      lr: 0.00001  # Smaller LR for pretrained backbone
    - name: "heads"
      lr: 0.0001   # Normal LR for new heads
```

### 3. Memory-Constrained Training
For systems with limited memory:
```bash
python train.py --config configs/training_config_light.yaml \
    --batch-size 1 --grad-accum 8 \
    --device cpu
```

## Advanced Features

### Learning Rate Scheduling

#### Cosine Annealing
```yaml
scheduler:
  type: "cosine"
  T_max: 100
  eta_min: 1e-7
```

#### One Cycle
```yaml
scheduler:
  type: "onecycle"
  max_lr: 0.001
  pct_start: 0.3
```

#### Reduce on Plateau
```yaml
scheduler:
  type: "plateau"
  factor: 0.5
  patience: 10
```

### Multi-GPU Training (CUDA only)
```python
# In config
device: "cuda"
data:
  batch_size: 16  # Will be split across GPUs

# Run with DataParallel
python train.py --config configs/training_config.yaml
```

### Custom Metrics Monitoring
The training pipeline automatically tracks:
- **Loss Components**: Instance, semantic, crown, density
- **Detection Metrics**: mAP@0.5, mAP@0.75, mAP@[0.5:0.95]
- **Agroforestry Metrics**: Tree count accuracy, canopy IoU, shade distribution
- **System Metrics**: Memory usage, GPU utilization, training speed

## Troubleshooting

### Out of Memory Errors

#### On GPU:
```bash
# Reduce batch size
python train.py --config configs/training_config.yaml --batch-size 1

# Enable gradient accumulation
python train.py --config configs/training_config.yaml \
    --batch-size 1 --grad-accum 4

# Enable mixed precision (CUDA only)
python train.py --config configs/training_config.yaml --mixed-precision
```

#### On MPS (Apple Silicon):
```bash
# Use smaller tile size
python train.py --config configs/training_config_light.yaml \
    --device mps --batch-size 1

# Disable MPS and use CPU
python train.py --config configs/training_config.yaml --device cpu
```

### Slow Training

1. **Check Data Loading**:
```yaml
data:
  num_workers: 4  # Increase for faster loading
  pin_memory: true  # For CUDA
```

2. **Enable Compilation** (PyTorch 2.0+):
```bash
python train.py --config configs/training_config.yaml --compile
```

3. **Profile Training**:
```python
# Add to config
training:
  profile: true
  profile_batches: [10, 20]
```

### Experiment Tracking Issues

#### MLflow:
```bash
# Clear MLflow runs
rm -rf mlruns/

# Use remote tracking server
export MLFLOW_TRACKING_URI=http://localhost:5000
```

#### Weights & Biases:
```bash
# Login
wandb login

# Offline mode
export WANDB_MODE=offline
```

## Performance Benchmarks

| Configuration | Device | Batch Size | Time/Epoch | Memory |
|--------------|--------|------------|------------|---------|
| Standard | RTX 3090 | 8 | 5 min | 12 GB |
| Standard | M1 Max | 4 | 8 min | 16 GB |
| Light | M1 | 2 | 12 min | 8 GB |
| Light | CPU (i7) | 2 | 20 min | 16 GB |

## Directory Structure
```
outputs/
├── cabruca_20240108_120000/
│   ├── checkpoints/
│   │   ├── checkpoint_best.pth
│   │   ├── checkpoint_latest.pth
│   │   └── checkpoint_epoch_10.pth
│   ├── tensorboard/
│   │   └── events.out.tfevents...
│   ├── training_history.json
│   └── config.yaml
```

## Best Practices

1. **Start Small**: Begin with small batches and low resolution
2. **Monitor Metrics**: Use dashboard to track convergence
3. **Save Regularly**: Set appropriate checkpoint frequency
4. **Use Validation**: Monitor validation metrics for overfitting
5. **Experiment Tracking**: Always use at least TensorBoard
6. **Resource Management**: Monitor memory usage during training
7. **Gradual Unfreezing**: For transfer learning, unfreeze layers gradually

## Citation
If you use this training pipeline in your research, please cite:
```bibtex
@software{cabruca_segmentation,
  title = {Cabruca Segmentation Training Pipeline},
  year = {2024},
  url = {https://github.com/cabruca-segmentation}
}
```
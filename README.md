# 🌳 Cabruca Segmentation System

Advanced ML-based segmentation system for Cabruca agroforestry analysis, featuring dual-head architecture for instance and semantic segmentation, specialized for cacao and shade tree detection in Brazilian agroforestry systems.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)

## 🎯 Features

- **Dual-Head Architecture**: Combines Mask R-CNN for instance segmentation with DeepLab v3+ for semantic segmentation
- **Multi-Class Detection**: Identifies cacao trees, shade trees, understory vegetation, bare soil, and shadows
- **Crown Analysis**: Estimates tree crown diameter and canopy density
- **Apple Silicon Support**: Optimized for M1/M2/M3 Macs with Metal Performance Shaders (MPS)
- **Agroforestry Metrics**: Specialized evaluation metrics for plantation health assessment
- **GIS Integration**: QGIS plugin and GeoJSON export for spatial analysis
- **Interactive Visualization**: Streamlit dashboard and API for real-time analysis
- **Theobroma Integration**: Seamless integration with existing plantation management systems

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [API Documentation](#-api-documentation)
- [Performance Benchmarks](#-performance-benchmarks)
- [Docker Deployment](#-docker-deployment)
- [Adding New Imagery](#-adding-new-imagery-sources)
- [Best Practices](#-best-practices)
- [Contributing](#-contributing)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- macOS 11+ (for Apple Silicon support) or Linux
- 8GB+ RAM (16GB recommended)
- 10GB+ free disk space

### 1. Clone Repository

```bash
git clone https://github.com/dlgiant/cabruca-segmentation.git
cd cabruca-segmentation
```

### 2. Create Virtual Environment

```bash
python -m venv ml_env
source ml_env/bin/activate
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# All dependencies
pip install -r requirements.txt

# Optional: QGIS plugin dependencies
pip install qgis-plugin-tools
```

### 4. Download Pre-trained Models (Optional)

```bash
# Download pre-trained checkpoint
wget https://example.com/cabruca_model_v1.pth -O outputs/checkpoint_best.pth
```

## ⚡ Quick Start

### 1. Training a Model

```bash
# Prepare your dataset
python prepare_data.py --input data/raw --output data/processed

# Train model
python train.py --config configs/training_config.yaml --data data/processed

# Monitor training
python monitor.py
```

### 2. Running Inference

```bash
# Single image
python infer.py --model outputs/checkpoint_best.pth --images plantation.tif --visualize

# Batch processing
python infer.py --model outputs/checkpoint_best.pth --images data/test/ --export-geojson

# Interactive viewer
python viewer.py
```

### 3. API Server

```bash
# Start API server
python api_server.py --model outputs/checkpoint_best.pth

# Test API
python test_api.py
```

### 4. Streamlit Dashboard

```bash
streamlit run src/inference/interactive_viewer.py
```

## 📁 Project Structure

```
cabruca-segmentation/
├── src/
│   ├── models/                 # Model architectures
│   │   ├── cabruca_segmentation_model.py
│   │   └── components.py
│   ├── data/                   # Data loading and processing
│   │   ├── dataset.py
│   │   └── augmentation.py
│   ├── training/               # Training pipeline
│   │   ├── advanced_trainer.py
│   │   └── scheduler.py
│   ├── inference/              # Inference and visualization
│   │   ├── batch_inference.py
│   │   └── interactive_viewer.py
│   ├── evaluation/             # Metrics and evaluation
│   │   └── agroforestry_metrics.py
│   ├── integration/            # External system integration
│   │   └── theobroma_integration.py
│   └── api/                    # API endpoints
│       ├── inference_api.py
│       └── client.py
├── configs/                    # Configuration files
│   └── training_config.yaml
├── qgis_plugin/               # QGIS integration
│   └── cabruca_qgis_plugin.py
├── notebooks/                  # Jupyter tutorials
│   ├── 01_getting_started.ipynb
│   ├── 02_training_custom_model.ipynb
│   └── 03_analysis_workflow.ipynb
├── docker/                     # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts/                    # Utility scripts
│   ├── prepare_data.py
│   └── benchmark.py
├── data/                       # Data directory
│   ├── raw/                   # Original imagery
│   ├── processed/             # Preprocessed data
│   └── geojson/              # Geographic boundaries
└── tests/                      # Unit tests
    └── test_model.py

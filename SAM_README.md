# Segment Anything Model (SAM) for Tree Segmentation

This module implements tree segmentation using Meta's Segment Anything Model (SAM), optimized for detecting and classifying cacao and shade trees in agroforestry systems.

## Features

✅ **Multi-class Tree Detection**: Distinguishes between cacao trees and shade trees
✅ **Multiple Prompting Strategies**: 
   - Point prompts for precise tree center selection
   - Bounding box prompts for region-based detection  
   - Automatic segmentation without manual prompts
✅ **Metal Performance Shaders Optimization**: Leverages Apple Silicon GPU acceleration
✅ **Custom Prompt Engineering**: Tree-specific prompt generation based on vegetation indices
✅ **Fine-tuning Support**: Adapt model to your specific tree detection needs

## Setup

### 1. Install Dependencies

```bash
# Install PyTorch with Metal support (for Apple Silicon)
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt

# Install Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 2. Download SAM Model

Run the setup script to download model checkpoints and configure SAM:

```bash
# Download ViT-B model (recommended for starting)
python setup_sam.py --model vit_b

# Or download larger models for better accuracy
python setup_sam.py --model vit_l  # Large model
python setup_sam.py --model vit_h  # Huge model (best accuracy, requires more memory)
```

The script will:
- Check for Metal Performance Shaders support
- Download the SAM checkpoint (~375MB for ViT-B)
- Create configuration files
- Test the model setup

## Usage

### Basic Tree Segmentation

```python
from src.models.sam_model import SAMConfig, SAMTreeSegmenter

# Initialize SAM
config = SAMConfig(model_type="vit_b")
segmenter = SAMTreeSegmenter(config)
segmenter.load_model()

# Load your image
import cv2
image = cv2.imread("path/to/drone_image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Automatic segmentation
results = segmenter.segment_automatic(image)
print(f"Found {results['total_trees']} trees")
print(f"  Cacao: {results['cacao_count']}")
print(f"  Shade: {results['shade_count']}")
```

### Command Line Interface

```bash
# Segment a single image
python run_sam_segmentation.py --image path/to/image.jpg --mode automatic

# Process entire directory
python run_sam_segmentation.py --input-dir data/images --output-dir output/segmented

# Interactive point selection
python run_sam_segmentation.py --image image.jpg --mode points --interactive

# Using bounding boxes
python run_sam_segmentation.py --image image.jpg --mode boxes
```

## Prompting Strategies

### 1. Point Prompts
Best for sparse tree detection or when you need precise control:
```python
# Generate automatic point prompts
points = segmenter.prompt_engineer.generate_point_prompts(image, tree_type='cacao')

# Or provide manual points
points = [(x1, y1), (x2, y2), ...]  # Tree center coordinates
results = segmenter.segment_with_points(image, points)
```

### 2. Bounding Box Prompts
Ideal for pre-detected regions or coarse localization:
```python
# Generate box prompts automatically
boxes = segmenter.prompt_engineer.generate_box_prompts(image)

# Or provide manual boxes
boxes = [[x1, y1, x2, y2], ...]  # [left, top, right, bottom]
results = segmenter.segment_with_boxes(image, boxes)
```

### 3. Automatic Segmentation
Let SAM find all trees automatically:
```python
results = segmenter.segment_automatic(image)
```

## Tree Classification

The system classifies detected trees into two categories:

- **Cacao Trees**: Smaller crown size, darker green color, typically 5m height
- **Shade Trees**: Larger crown size, lighter green color, typically 15m height

Classification is based on:
1. Mask area (crown size)
2. Color features (vegetation indices)
3. Shape characteristics (compactness, aspect ratio)

## Configuration

Edit `configs/sam_config.json` to customize:

```json
{
  "model": {
    "type": "vit_b",
    "device": "auto"
  },
  "tree_detection": {
    "num_classes": 2,
    "confidence_threshold": 0.7,
    "min_tree_area": 100,
    "max_tree_area": 50000
  },
  "prompt_engineering": {
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95
  }
}
```

## Performance Optimization

### Apple Silicon (M1/M2/M3)
The system automatically detects and uses Metal Performance Shaders:
```python
✅ Metal Performance Shaders (MPS) is available!
✅ MPS test successful - matrix multiplication works!
```

### Memory Management
- Use `vit_b` for systems with <16GB RAM
- Use `vit_l` for systems with 16-32GB RAM  
- Use `vit_h` for systems with >32GB RAM

### Batch Processing
Adjust batch size based on available memory:
```python
config = SAMConfig(
    batch_size=1,  # Reduce if running out of memory
    enable_mixed_precision=True  # Enable for faster processing
)
```

## Fine-tuning

To fine-tune SAM on your tree dataset:

```python
# Prepare your data
train_data = {
    'images': [...],  # List of images
    'masks': [...],   # Ground truth masks
    'labels': [...]   # Tree type labels
}

# Fine-tune model
segmenter.fine_tune(
    train_data=train_data,
    epochs=10
)

# Save fine-tuned model
segmenter.save_model("models/sam_trees_finetuned.pth")
```

## Troubleshooting

### Issue: "segment-anything not installed"
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Issue: "MPS backend out of memory"
Reduce batch size or use smaller model:
```python
config = SAMConfig(model_type="vit_b", batch_size=1)
```

### Issue: Poor tree detection
Try adjusting thresholds:
```python
config = SAMConfig(
    pred_iou_thresh=0.85,  # Lower for more detections
    min_mask_region_area=50  # Smaller minimum tree size
)
```

## Model Variants

| Model | Parameters | Speed | Accuracy | Memory |
|-------|-----------|-------|----------|---------|
| ViT-B | 91M | Fast | Good | ~4GB |
| ViT-L | 308M | Medium | Better | ~8GB |
| ViT-H | 636M | Slow | Best | ~16GB |

## Directory Structure

```
cabruca-segmentation/
├── src/models/
│   └── sam_model.py          # Main SAM implementation
├── configs/
│   └── sam_config.json       # Configuration file
├── models/sam_checkpoints/   # Downloaded model checkpoints
├── setup_sam.py              # Setup and download script
├── run_sam_segmentation.py   # CLI for segmentation
└── output/sam_results/       # Segmentation results
```

## Citation

If you use this implementation, please cite:

```bibtex
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
```

## Next Steps

1. Run setup: `python setup_sam.py`
2. Test on sample image: `python run_sam_segmentation.py --image sample.jpg`
3. Process your dataset: `python run_sam_segmentation.py --input-dir data/`
4. Fine-tune on labeled data for better accuracy
5. Integrate with geospatial pipeline for farm-scale analysis

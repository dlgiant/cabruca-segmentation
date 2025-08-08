# Annotation Tools and Data Preparation Pipeline

This directory contains a comprehensive suite of tools for annotating satellite/drone imagery and preparing training data for cacao tree detection models.

## 📋 Overview

The annotation pipeline provides:

1. **Web-based Manual Annotation** - Streamlit app for interactive labeling
2. **Semi-automated Annotation** - SAM (Segment Anything Model) integration for zero-shot segmentation
3. **Format Conversion** - Convert between LabelMe, custom JSON, and COCO formats
4. **Dataset Splitting** - Stratified train/val/test splits with class balancing
5. **Class Imbalance Handling** - Multiple strategies for dealing with imbalanced classes

## 🚀 Quick Start

### Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Download SAM model checkpoint (if using SAM annotation)
cd models/sam
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../..
```

### Running the Full Pipeline

```bash
# Run complete annotation and data preparation pipeline
python annotation_tools/annotation_pipeline.py --image-dir data/raw --balance-strategy mixed
```

## 🛠️ Tools

### 1. Streamlit Annotation App

Interactive web interface for manual annotation.

```bash
# Launch the annotation app
streamlit run annotation_tools/streamlit_app/annotation_app.py
```

Features:
- Multi-class annotation (background, cacao, shade trees)
- Multiple drawing modes (polygon, rectangle, circle, point)
- Real-time statistics and visualization
- Export to JSON and LabelMe formats

### 2. SAM Auto-Annotator

Leverage SAM's zero-shot capabilities for semi-automated annotation.

```python
from annotation_tools.sam_annotation.sam_auto_annotator import SAMAutoAnnotator, AnnotationConfig

# Configure SAM
config = AnnotationConfig(
    model_type="vit_h",
    checkpoint_path="models/sam/sam_vit_h_4b8939.pth",
    cacao_min_area=50,
    cacao_max_area=500,
    shade_tree_min_area=500,
    shade_tree_max_area=5000
)

# Initialize and run
annotator = SAMAutoAnnotator(config)
annotations = annotator.batch_annotate("data/raw", "data/annotations/sam")
```

Features:
- Automatic mask generation
- Class prediction based on object properties
- Interactive refinement with user points
- Batch processing

### 3. COCO Format Converter

Convert various annotation formats to COCO for training.

```python
from annotation_tools.coco_conversion.annotation_converter import COCOConverter

converter = COCOConverter()

# Convert LabelMe annotations
converter.batch_convert("data/annotations/labelme", format_type="labelme")

# Convert custom JSON annotations
converter.batch_convert("data/annotations/sam", format_type="custom")

# Save COCO dataset
converter.save_coco_dataset("data/processed/annotations_coco.json")
```

Supported formats:
- LabelMe JSON
- Custom JSON (from Streamlit app or SAM)
- Auto-detection of format

### 4. Dataset Splitter

Create balanced train/validation/test splits.

```python
from annotation_tools.data_splitting.dataset_splitter import DatasetSplitter

splitter = DatasetSplitter("data/processed/annotations_coco.json")

# Analyze class distribution
stats = splitter.analyze_class_distribution()

# Create splits with balancing
output_paths = splitter.create_split_datasets(
    output_dir="data/processed/splits",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    balance_strategy="mixed"  # Options: oversample, undersample, mixed, None
)

# Visualize distribution
splitter.visualize_class_distribution("data/processed/class_distribution.png")
```

## 📊 Class Balancing Strategies

### 1. Oversampling
- Duplicates minority class samples
- Increases dataset size
- May lead to overfitting

### 2. Undersampling
- Removes majority class samples
- Reduces dataset size
- May lose important information

### 3. Mixed Strategy
- Oversample minority to median
- Undersample majority classes
- Balanced approach

### 4. Data Augmentation (planned)
- Generate synthetic samples
- Apply transformations
- Preserve data diversity

## 🗂️ Directory Structure

```
annotation_tools/
├── streamlit_app/           # Web-based annotation interface
│   └── annotation_app.py
├── sam_annotation/          # SAM-based semi-automated annotation
│   └── sam_auto_annotator.py
├── coco_conversion/         # Format conversion tools
│   └── annotation_converter.py
├── data_splitting/          # Dataset splitting and balancing
│   └── dataset_splitter.py
├── annotation_pipeline.py   # Main pipeline orchestrator
└── README.md               # This file
```

## 💾 Output Formats

### COCO Format
```json
{
  "info": {...},
  "licenses": [...],
  "images": [
    {
      "id": 1,
      "file_name": "image.jpg",
      "width": 1024,
      "height": 768
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1,y1,x2,y2,...]],
      "area": 500.0,
      "bbox": [x,y,width,height],
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 0, "name": "background"},
    {"id": 1, "name": "cacao"},
    {"id": 2, "name": "shade_tree"}
  ]
}
```

## 🎯 Class Definitions

- **Background (0)**: Everything that is not a tree (ground, buildings, etc.)
- **Cacao (1)**: Cacao trees, typically smaller with dense canopy
- **Shade Tree (2)**: Larger shade trees providing canopy cover

## 📈 Pipeline Workflow

1. **Data Collection**: Gather satellite/drone imagery
2. **Initial Annotation**: Use Streamlit app or SAM for first-pass labeling
3. **Refinement**: Review and correct annotations
4. **Format Conversion**: Convert all annotations to COCO format
5. **Dataset Splitting**: Create train/val/test splits with stratification
6. **Class Balancing**: Apply balancing strategy to handle imbalance
7. **Export**: Save processed datasets for model training

## 🔧 Command Line Interface

### Full Pipeline
```bash
python annotation_pipeline.py --image-dir data/raw --balance-strategy mixed
```

### Individual Steps
```bash
# SAM annotation only
python annotation_pipeline.py --run-sam --image-dir data/raw

# Convert annotations to COCO
python annotation_pipeline.py --convert-only \
    --annotation-dirs data/annotations/sam data/annotations/labelme

# Split existing COCO dataset
python annotation_pipeline.py --split-only \
    --coco-path data/processed/annotations_coco.json \
    --balance-strategy oversample
```

## 📝 Tips for Quality Annotations

1. **Consistency**: Maintain consistent labeling criteria across all images
2. **Edge Cases**: Document how to handle ambiguous cases
3. **Validation**: Cross-check annotations with multiple annotators
4. **Iterative Refinement**: Use model predictions to identify labeling errors
5. **Class Balance**: Ensure adequate representation of all classes

## 🐛 Troubleshooting

### Common Issues

1. **SAM Memory Error**: Reduce `points_per_side` or use smaller images
2. **Class Imbalance**: Try different balancing strategies
3. **Format Conversion Errors**: Check JSON structure matches expected format
4. **Streamlit App Slow**: Reduce image resolution for annotation

## 📚 References

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [LabelMe Annotation Tool](https://github.com/wkentaro/labelme)
- [Imbalanced-learn](https://imbalanced-learn.org/)

## 🤝 Contributing

To add new annotation formats or balancing strategies:

1. Create new module in appropriate directory
2. Follow existing code structure and documentation
3. Add tests for new functionality
4. Update this README with usage examples

## 📄 License

This project is part of the Cabruca Segmentation system for sustainable cacao farming.

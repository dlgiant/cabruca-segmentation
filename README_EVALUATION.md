# Agroforestry Evaluation Metrics

## Overview
Comprehensive evaluation system for Cabruca agroforestry segmentation models with domain-specific metrics designed for agricultural and ecological assessment.

## Implemented Metrics

### 1. Tree Count Accuracy
- **Cacao Tree Count Accuracy**: Measures accuracy of cacao tree detection
- **Shade Tree Count Accuracy**: Measures accuracy of shade tree detection
- **Species Classification Accuracy**: Evaluates correct classification of tree species
- **Per-species Precision/Recall/F1**: Individual metrics for each tree type

### 2. Canopy Coverage Analysis
- **Coverage Percentage**: Total canopy area as percentage of image
- **Coverage Error**: Absolute and relative error vs ground truth
- **Canopy IoU**: Intersection over Union for canopy regions
- **Species-specific Coverage**: Separate metrics for cacao and shade trees
- **Crown Diameter Statistics**: Mean, std, min, max crown diameters

### 3. Tree Spacing Uniformity
- **Nearest Neighbor Distance**: Average and standard deviation
- **Clark-Evans Index**: Measure of spatial pattern (clustered/random/uniform)
  - R < 0.8: Clustered distribution
  - 0.8 ≤ R ≤ 1.2: Random distribution
  - R > 1.2: Uniform distribution
- **Ripley's K Function**: Multi-scale spatial analysis
- **Voronoi Cell Analysis**: Regularity of tree spacing
- **Coefficient of Variation**: Spacing consistency metric

### 4. Shade Distribution Analysis
- **Shade Tree Density**: Trees per unit area
- **Shade-to-Cacao Ratio**: Proportion of shade vs productive trees
- **Shade Ratio Optimality**: Comparison to ideal ratio (1:5)
- **Cacao Shade Coverage**: Percentage of cacao trees within effective shade radius
- **Shade Distribution Score**: Uniformity of shade coverage
- **Average Distance to Shade**: Mean distance from cacao to nearest shade tree

### 5. Instance Segmentation Metrics (mAP)
- **mAP@[0.5:0.95]**: COCO-style mean Average Precision
- **mAP@0.5**: Standard object detection metric
- **mAP@0.75**: Strict IoU threshold
- **Per-class AP**: Separate AP for cacao and shade trees
- **Detection Precision/Recall/F1**: Overall detection performance

### 6. Crown IoU Metrics
- **Individual Crown IoU**: Per-tree crown segmentation accuracy
- **Crown IoU Statistics**: Mean, std, median, min, max
- **IoU Threshold Analysis**: Percentage above 0.5, 0.7, 0.9
- **Crown Area Accuracy**: Size estimation accuracy

### 7. Plantation Data Comparison
- **Detection Rate**: Percentage of known trees detected
- **Position Accuracy**: Average and max position error (meters)
- **Distance Threshold Analysis**: Trees within 0.5m, 1m, 2m, 5m
- **Species Classification**: Accuracy vs plantation records

## Usage

### Basic Evaluation
```python
from evaluation.agroforestry_metrics import AgroforestryMetrics

# Initialize evaluator
evaluator = AgroforestryMetrics(
    iou_threshold=0.5,
    distance_threshold=2.0  # meters
)

# Run evaluation
metrics = evaluator.evaluate_batch(
    predictions=model_outputs,
    ground_truth=annotations,
    plantation_data='data/plantation-data.json'
)

# Print key metrics
print(f"mAP@0.5: {metrics['mAP_50']:.3f}")
print(f"Canopy Coverage: {metrics['predicted_canopy_coverage']:.2%}")
print(f"Shade Distribution Score: {metrics['shade_distribution_score']:.3f}")
```

### Individual Metric Calculation
```python
# Tree count accuracy
tree_metrics = evaluator.calculate_tree_count_accuracy(predictions, ground_truth)

# Canopy coverage
canopy_metrics = evaluator.calculate_canopy_coverage(predictions, ground_truth)

# Spacing uniformity
spacing_metrics = evaluator.calculate_spacing_uniformity(predictions)

# Shade distribution
shade_metrics = evaluator.analyze_shade_distribution(predictions, ground_truth)
```

### Visualization
```python
from evaluation.agroforestry_metrics import visualize_evaluation_results

# Create comprehensive visualization
fig = visualize_evaluation_results(
    metrics=all_metrics,
    save_path='evaluation_results.png'
)
```

## Metric Interpretation

### Tree Count Accuracy
- **Good**: > 90% accuracy
- **Acceptable**: 80-90% accuracy
- **Needs Improvement**: < 80% accuracy

### Canopy Coverage
- **Optimal for Cabruca**: 40-60% coverage
- **Over-shaded**: > 70% coverage
- **Under-shaded**: < 30% coverage

### Spacing Uniformity (Clark-Evans Index)
- **Uniform** (R > 1.2): Well-planned plantation
- **Random** (0.8 ≤ R ≤ 1.2): Natural distribution
- **Clustered** (R < 0.8): Uneven distribution

### Shade Distribution
- **Optimal Ratio**: 1 shade tree per 4-6 cacao trees
- **Effective Coverage**: > 80% of cacao within shade radius
- **Distribution Score**: > 0.7 indicates good uniformity

### Instance Segmentation (mAP)
- **Excellent**: mAP > 0.7
- **Good**: mAP 0.5-0.7
- **Fair**: mAP 0.3-0.5
- **Poor**: mAP < 0.3

## Plantation Data Format

The system supports comparison with existing plantation data in JSON format:

```json
{
  "trees": [
    {
      "id": 1,
      "species": "cacao",
      "coordinates": [100.5, 200.3],
      "crown_diameter_m": 3.5
    },
    {
      "id": 2,
      "species": "shade",
      "coordinates": [120.8, 190.5],
      "crown_diameter_m": 15.2
    }
  ]
}
```

## Output Metrics Dictionary

The evaluation returns a comprehensive dictionary with all metrics:

```python
{
    # Tree count metrics
    'cacao_count_accuracy': 0.95,
    'shade_count_accuracy': 0.90,
    'species_classification_accuracy': 0.93,
    
    # Canopy metrics
    'predicted_canopy_coverage': 0.45,
    'canopy_iou': 0.82,
    'avg_crown_diameter': 8.5,
    
    # Spacing metrics
    'avg_nearest_neighbor_distance': 6.2,
    'clark_evans_index': 1.15,
    'spatial_pattern': 'random',
    
    # Shade metrics
    'shade_to_cacao_ratio': 0.22,
    'cacao_shade_coverage': 0.85,
    'shade_distribution_score': 0.73,
    
    # Detection metrics
    'mAP': 0.65,
    'mAP_50': 0.78,
    'detection_f1': 0.81,
    
    # Crown metrics
    'crown_iou_mean': 0.71,
    'crown_area_accuracy': 0.88,
    
    # Plantation comparison
    'plantation_detection_rate': 0.92,
    'avg_position_error': 1.3
}
```

## Visualization Components

The visualization includes 8 subplots:
1. **Tree Count Accuracy**: Bar chart by species
2. **Species Classification**: Precision/Recall/F1 comparison
3. **Canopy Coverage**: Pie chart comparison
4. **Spacing Uniformity**: Horizontal bar metrics
5. **Instance Segmentation mAP**: Performance at different IoU thresholds
6. **Crown IoU Statistics**: Distribution analysis
7. **Shade Distribution**: Coverage and optimality metrics
8. **Plantation Comparison**: Detection and accuracy metrics

## Performance Considerations

- **Batch Processing**: Evaluate multiple images together for efficiency
- **IoU Calculation**: Use vectorized operations for mask IoU
- **Spatial Analysis**: Cache distance matrices for repeated calculations
- **Memory Management**: Process large images in tiles

## Integration with Training

```python
# In training loop
for epoch in range(num_epochs):
    train_model()
    
    # Validation with agroforestry metrics
    val_predictions = model.predict(val_data)
    metrics = evaluator.evaluate_batch(
        val_predictions, 
        val_ground_truth,
        plantation_data
    )
    
    # Log metrics
    logger.log(metrics)
    
    # Early stopping based on domain metrics
    if metrics['shade_distribution_score'] > best_shade_score:
        save_best_model()
```

## Future Enhancements

- [ ] Temporal analysis for growth monitoring
- [ ] Carbon sequestration estimation
- [ ] Disease detection metrics
- [ ] Yield prediction correlations
- [ ] Multi-temporal change detection
- [ ] Biodiversity indices
- [ ] Water stress indicators
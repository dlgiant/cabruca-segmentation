"""
Test script for agroforestry evaluation metrics.
"""

import numpy as np
import torch
import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append('src')

from evaluation.agroforestry_metrics import AgroforestryMetrics, visualize_evaluation_results


def create_sample_predictions():
    """Create sample predictions for testing."""
    predictions = {
        'instances': {
            'boxes': np.array([
                [98, 198, 103, 203],    # Near tree 1
                [108, 203, 113, 208],    # Near tree 2
                [118, 188, 123, 193],    # Near tree 3 (shade)
                [93, 208, 98, 213],      # Near tree 4
                [113, 213, 118, 218],    # Near tree 5
                [138, 208, 143, 213],    # Near tree 6 (shade)
                [103, 193, 108, 198],    # Near tree 7
                [123, 198, 128, 203],    # Near tree 8
                [128, 203, 133, 208],    # Near tree 9
                [158, 193, 163, 198],    # Near tree 10 (shade)
                [145, 200, 150, 205],    # False positive
            ]),
            'labels': np.array([1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1]),  # 1=cacao, 2=shade
            'scores': np.array([0.95, 0.92, 0.88, 0.91, 0.89, 0.87, 0.93, 0.90, 0.94, 0.86, 0.65]),
            'masks': np.random.rand(11, 512, 512) > 0.7  # Random masks for testing
        },
        'semantic_map': np.random.randint(0, 6, (512, 512)),  # Random semantic map
        'crown_map': np.random.rand(512, 512) * 10,  # Random crown diameters
        'canopy_density': 0.42
    }
    
    # Make semantic map more realistic
    # Create circular patterns for trees
    h, w = 512, 512
    semantic_map = np.zeros((h, w), dtype=np.int32)
    
    # Add tree regions
    for box, label in zip(predictions['instances']['boxes'], predictions['instances']['labels']):
        x1, y1, x2, y2 = box.astype(int)
        # Scale to image size
        x1, x2 = x1 * 2, x2 * 2
        y1, y2 = y1 * 2, y2 * 2
        
        # Clip to image bounds
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if label == 1:  # Cacao
            semantic_map[y1:y2, x1:x2] = 1
        elif label == 2:  # Shade
            semantic_map[y1:y2, x1:x2] = 2
    
    # Add some understory and bare soil
    semantic_map[semantic_map == 0] = np.random.choice([0, 3, 4], size=(semantic_map == 0).sum(), p=[0.5, 0.3, 0.2])
    
    predictions['semantic_map'] = semantic_map
    
    return predictions


def create_sample_ground_truth():
    """Create sample ground truth for testing."""
    ground_truth = {
        'instances': {
            'boxes': np.array([
                [100, 200, 105, 205],   # Tree 1
                [110, 205, 115, 210],   # Tree 2
                [120, 190, 125, 195],   # Tree 3 (shade)
                [95, 210, 100, 215],    # Tree 4
                [115, 215, 120, 220],   # Tree 5
                [140, 210, 145, 215],   # Tree 6 (shade)
                [105, 195, 110, 200],   # Tree 7
                [125, 200, 130, 205],   # Tree 8
                [130, 205, 135, 210],   # Tree 9
                [160, 195, 165, 200],   # Tree 10 (shade)
            ]),
            'labels': np.array([1, 1, 2, 1, 1, 2, 1, 1, 1, 2]),  # 1=cacao, 2=shade
            'masks': np.random.rand(10, 512, 512) > 0.7  # Random masks for testing
        },
        'semantic_map': np.random.randint(0, 6, (512, 512)),  # Random semantic map
        'crown_map': np.random.rand(512, 512) * 10,  # Random crown diameters
        'density_gt': 0.45
    }
    
    # Make semantic map similar to predictions
    h, w = 512, 512
    semantic_map = np.zeros((h, w), dtype=np.int32)
    
    for box, label in zip(ground_truth['instances']['boxes'], ground_truth['instances']['labels']):
        x1, y1, x2, y2 = box.astype(int)
        x1, x2 = x1 * 2, x2 * 2
        y1, y2 = y1 * 2, y2 * 2
        
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if label == 1:
            semantic_map[y1:y2, x1:x2] = 1
        elif label == 2:
            semantic_map[y1:y2, x1:x2] = 2
    
    semantic_map[semantic_map == 0] = np.random.choice([0, 3, 4], size=(semantic_map == 0).sum(), p=[0.5, 0.3, 0.2])
    
    ground_truth['semantic_map'] = semantic_map
    
    return ground_truth


def test_tree_count_accuracy():
    """Test tree count accuracy metrics."""
    print("\n" + "="*50)
    print("Testing Tree Count Accuracy")
    print("="*50)
    
    evaluator = AgroforestryMetrics()
    predictions = create_sample_predictions()
    ground_truth = create_sample_ground_truth()
    
    metrics = evaluator.calculate_tree_count_accuracy(predictions, ground_truth)
    
    print("Tree Count Metrics:")
    print(f"  Cacao count accuracy: {metrics['cacao_count_accuracy']:.2%}")
    print(f"  Shade count accuracy: {metrics['shade_count_accuracy']:.2%}")
    print(f"  Total count accuracy: {metrics['total_count_accuracy']:.2%}")
    print(f"  Species classification accuracy: {metrics['species_classification_accuracy']:.2%}")
    print(f"\nPer-species metrics:")
    print(f"  Cacao - Precision: {metrics['cacao_precision']:.3f}, Recall: {metrics['cacao_recall']:.3f}, F1: {metrics['cacao_f1']:.3f}")
    print(f"  Shade - Precision: {metrics['shade_precision']:.3f}, Recall: {metrics['shade_recall']:.3f}, F1: {metrics['shade_f1']:.3f}")
    
    return metrics


def test_canopy_coverage():
    """Test canopy coverage metrics."""
    print("\n" + "="*50)
    print("Testing Canopy Coverage")
    print("="*50)
    
    evaluator = AgroforestryMetrics()
    predictions = create_sample_predictions()
    ground_truth = create_sample_ground_truth()
    
    metrics = evaluator.calculate_canopy_coverage(predictions, ground_truth)
    
    print("Canopy Coverage Metrics:")
    print(f"  Predicted coverage: {metrics['predicted_canopy_coverage']:.2%}")
    print(f"  Ground truth coverage: {metrics['ground_truth_canopy_coverage']:.2%}")
    print(f"  Coverage error: {metrics['canopy_coverage_error']:.4f}")
    print(f"  Canopy IoU: {metrics['canopy_iou']:.3f}")
    print(f"  Average crown diameter: {metrics.get('avg_crown_diameter', 0):.2f} m")
    
    return metrics


def test_spacing_uniformity():
    """Test tree spacing uniformity metrics."""
    print("\n" + "="*50)
    print("Testing Spacing Uniformity")
    print("="*50)
    
    evaluator = AgroforestryMetrics()
    predictions = create_sample_predictions()
    
    metrics = evaluator.calculate_spacing_uniformity(predictions)
    
    print("Spacing Uniformity Metrics:")
    print(f"  Avg nearest neighbor distance: {metrics['avg_nearest_neighbor_distance']:.2f} m")
    print(f"  Std nearest neighbor distance: {metrics['std_nearest_neighbor_distance']:.2f} m")
    print(f"  CV nearest neighbor: {metrics['cv_nearest_neighbor']:.3f}")
    print(f"  Clark-Evans Index: {metrics['clark_evans_index']:.3f}")
    print(f"  Spatial pattern: {metrics['spatial_pattern']}")
    
    if 'voronoi_area_mean' in metrics:
        print(f"  Voronoi cell area mean: {metrics['voronoi_area_mean']:.2f} m²")
        print(f"  Voronoi cell area CV: {metrics.get('voronoi_area_cv', 0):.3f}")
    
    return metrics


def test_shade_distribution():
    """Test shade distribution analysis."""
    print("\n" + "="*50)
    print("Testing Shade Distribution")
    print("="*50)
    
    evaluator = AgroforestryMetrics()
    predictions = create_sample_predictions()
    ground_truth = create_sample_ground_truth()
    
    metrics = evaluator.analyze_shade_distribution(predictions, ground_truth)
    
    print("Shade Distribution Metrics:")
    print(f"  Shade tree density: {metrics.get('shade_tree_density', 0):.4f} trees/m²")
    print(f"  Shade to cacao ratio: {metrics.get('shade_to_cacao_ratio', 0):.3f}")
    print(f"  Shade ratio optimality: {metrics.get('shade_ratio_optimality', 0):.2%}")
    print(f"  Cacao shade coverage: {metrics.get('cacao_shade_coverage', 0):.2%}")
    print(f"  Avg cacao to shade distance: {metrics.get('avg_cacao_to_shade_distance', 0):.2f} m")
    print(f"  Shade distribution score: {metrics.get('shade_distribution_score', 0):.3f}")
    
    return metrics


def test_instance_metrics():
    """Test instance segmentation metrics (mAP)."""
    print("\n" + "="*50)
    print("Testing Instance Segmentation Metrics")
    print("="*50)
    
    evaluator = AgroforestryMetrics()
    predictions = create_sample_predictions()
    ground_truth = create_sample_ground_truth()
    
    metrics = evaluator.calculate_instance_metrics(predictions, ground_truth)
    
    print("Instance Segmentation Metrics:")
    print(f"  mAP@[0.5:0.95]: {metrics['mAP']:.3f}")
    print(f"  mAP@0.5: {metrics['mAP_50']:.3f}")
    print(f"  mAP@0.75: {metrics['mAP_75']:.3f}")
    print(f"  AP (cacao): {metrics['AP_cacao']:.3f}")
    print(f"  AP (shade): {metrics['AP_shade']:.3f}")
    print(f"  Detection precision: {metrics['detection_precision']:.3f}")
    print(f"  Detection recall: {metrics['detection_recall']:.3f}")
    print(f"  Detection F1: {metrics['detection_f1']:.3f}")
    
    return metrics


def test_crown_iou():
    """Test crown IoU metrics."""
    print("\n" + "="*50)
    print("Testing Crown IoU")
    print("="*50)
    
    evaluator = AgroforestryMetrics()
    predictions = create_sample_predictions()
    ground_truth = create_sample_ground_truth()
    
    metrics = evaluator.calculate_crown_iou(predictions, ground_truth)
    
    print("Crown IoU Metrics:")
    print(f"  Mean IoU: {metrics.get('crown_iou_mean', 0):.3f}")
    print(f"  Std IoU: {metrics.get('crown_iou_std', 0):.3f}")
    print(f"  Median IoU: {metrics.get('crown_iou_median', 0):.3f}")
    print(f"  Crown area accuracy: {metrics.get('crown_area_accuracy', 0):.3f}")
    
    for thresh in [50, 70, 90]:
        key = f'crown_iou_above_{thresh}'
        if key in metrics:
            print(f"  Crowns with IoU > {thresh/100:.1f}: {metrics[key]:.2%}")
    
    return metrics


def test_plantation_comparison():
    """Test comparison with plantation data."""
    print("\n" + "="*50)
    print("Testing Plantation Data Comparison")
    print("="*50)
    
    evaluator = AgroforestryMetrics()
    predictions = create_sample_predictions()
    plantation_file = "data/plantation-data.json"
    
    metrics = evaluator.compare_with_plantation_data(predictions, plantation_file)
    
    if metrics:
        print("Plantation Comparison Metrics:")
        print(f"  Detection rate: {metrics.get('plantation_detection_rate', 0):.2%}")
        print(f"  Precision: {metrics.get('plantation_precision', 0):.3f}")
        print(f"  Recall: {metrics.get('plantation_recall', 0):.3f}")
        print(f"  F1 score: {metrics.get('plantation_f1', 0):.3f}")
        
        if 'avg_position_error' in metrics:
            print(f"  Avg position error: {metrics['avg_position_error']:.2f} m")
            print(f"  Max position error: {metrics['max_position_error']:.2f} m")
        
        for thresh in [0.5, 1.0, 2.0, 5.0]:
            key = f'trees_within_{thresh}m'
            if key in metrics:
                print(f"  Trees within {thresh}m: {metrics[key]:.2%}")
    else:
        print("  No plantation data found or comparison failed")
    
    return metrics


def test_complete_evaluation():
    """Test complete evaluation pipeline."""
    print("\n" + "="*50)
    print("Testing Complete Evaluation Pipeline")
    print("="*50)
    
    evaluator = AgroforestryMetrics(iou_threshold=0.5, distance_threshold=2.0)
    predictions = create_sample_predictions()
    ground_truth = create_sample_ground_truth()
    plantation_file = "data/plantation-data.json"
    
    # Run complete evaluation
    all_metrics = evaluator.evaluate_batch(predictions, ground_truth, plantation_file)
    
    print("\nComplete Evaluation Summary:")
    print("-"*40)
    
    # Summary statistics
    key_metrics = [
        ('Total trees detected', len(predictions['instances']['labels'])),
        ('Total trees ground truth', len(ground_truth['instances']['labels'])),
        ('Overall accuracy', all_metrics.get('total_count_accuracy', 0)),
        ('mAP@0.5', all_metrics.get('mAP_50', 0)),
        ('Canopy IoU', all_metrics.get('canopy_iou', 0)),
        ('Shade distribution score', all_metrics.get('shade_distribution_score', 0)),
    ]
    
    for metric_name, value in key_metrics:
        if isinstance(value, float):
            print(f"{metric_name}: {value:.3f}")
        else:
            print(f"{metric_name}: {value}")
    
    # Create visualization
    print("\nGenerating visualization...")
    visualize_evaluation_results(all_metrics, save_path="evaluation_results.png")
    print("Visualization saved to evaluation_results.png")
    
    return all_metrics


def main():
    """Run all metric tests."""
    print("\n" + "="*60)
    print(" AGROFORESTRY METRICS TEST SUITE")
    print("="*60)
    
    test_results = {}
    
    # Run individual tests
    tests = [
        ("Tree Count Accuracy", test_tree_count_accuracy),
        ("Canopy Coverage", test_canopy_coverage),
        ("Spacing Uniformity", test_spacing_uniformity),
        ("Shade Distribution", test_shade_distribution),
        ("Instance Metrics (mAP)", test_instance_metrics),
        ("Crown IoU", test_crown_iou),
        ("Plantation Comparison", test_plantation_comparison),
        ("Complete Evaluation", test_complete_evaluation)
    ]
    
    for test_name, test_func in tests:
        try:
            metrics = test_func()
            test_results[test_name] = "PASSED"
        except Exception as e:
            print(f"\n✗ {test_name} failed: {str(e)}")
            test_results[test_name] = "FAILED"
    
    # Summary
    print("\n" + "="*60)
    print(" TEST SUMMARY")
    print("="*60)
    
    for test_name, status in test_results.items():
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{test_name:.<40} {symbol} {status}")
    
    passed = sum(1 for s in test_results.values() if s == "PASSED")
    total = len(test_results)
    
    print("\n" + "-"*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All agroforestry metrics tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
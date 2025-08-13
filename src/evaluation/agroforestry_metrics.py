"""
Agroforestry-specific evaluation metrics for Cabruca system assessment.
Includes tree detection metrics, canopy analysis, and spatial distribution evaluation.
"""

import json
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import agentops
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.spatial import Voronoi, distance_matrix, voronoi_plot_2d
from scipy.stats import chi2
from shapely.geometry import Point, Polygon
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")


class AgroforestryMetrics:
    """
    Comprehensive evaluation metrics for agroforestry segmentation assessment.
    """

    def __init__(self, iou_threshold: float = 0.5, distance_threshold: float = 2.0):
        """
        Initialize metrics calculator.

        Args:
            iou_threshold: IoU threshold for matching predictions with ground truth
            distance_threshold: Maximum distance (meters) for tree matching
        """
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.metrics_history = []

    @agentops.tool(name="AgroforestryEvaluator")
    def evaluate_batch(
        self,
        predictions: Dict,
        ground_truth: Dict,
        plantation_data: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate a batch of predictions against ground truth.

        Args:
            predictions: Model predictions
            ground_truth: Ground truth annotations
            plantation_data: Path to plantation data JSON file

        Returns:
            Dictionary containing all evaluation metrics
        """
        metrics = {}

        # Tree count accuracy
        tree_metrics = self.calculate_tree_count_accuracy(predictions, ground_truth)
        metrics.update(tree_metrics)

        # Canopy coverage metrics
        canopy_metrics = self.calculate_canopy_coverage(predictions, ground_truth)
        metrics.update(canopy_metrics)

        # Tree spacing uniformity
        spacing_metrics = self.calculate_spacing_uniformity(predictions)
        metrics.update(spacing_metrics)

        # Shade distribution analysis
        shade_metrics = self.analyze_shade_distribution(predictions, ground_truth)
        metrics.update(shade_metrics)

        # Instance segmentation metrics (mAP)
        instance_metrics = self.calculate_instance_metrics(predictions, ground_truth)
        metrics.update(instance_metrics)

        # IoU for tree crowns
        crown_metrics = self.calculate_crown_iou(predictions, ground_truth)
        metrics.update(crown_metrics)

        # Compare with plantation data if provided
        if plantation_data:
            plantation_metrics = self.compare_with_plantation_data(
                predictions, plantation_data
            )
            metrics.update(plantation_metrics)

        # Store metrics for tracking
        self.metrics_history.append(metrics)

        return metrics

    def calculate_tree_count_accuracy(
        self, predictions: Dict, ground_truth: Dict
    ) -> Dict:
        """
        Calculate tree count accuracy for different species.

        Returns metrics:
        - Cacao tree count accuracy
        - Shade tree count accuracy
        - Total tree count accuracy
        - Species classification accuracy
        """
        metrics = {}

        # Extract predicted and ground truth counts
        pred_labels = predictions.get("instances", {}).get("labels", np.array([]))
        gt_labels = ground_truth.get("instances", {}).get("labels", np.array([]))

        # Count by species (1: cacao, 2: shade tree)
        pred_cacao = np.sum(pred_labels == 1)
        pred_shade = np.sum(pred_labels == 2)
        pred_total = len(pred_labels)

        gt_cacao = np.sum(gt_labels == 1)
        gt_shade = np.sum(gt_labels == 2)
        gt_total = len(gt_labels)

        # Calculate accuracies
        metrics["cacao_count_accuracy"] = 1.0 - abs(pred_cacao - gt_cacao) / max(
            gt_cacao, 1
        )
        metrics["shade_count_accuracy"] = 1.0 - abs(pred_shade - gt_shade) / max(
            gt_shade, 1
        )
        metrics["total_count_accuracy"] = 1.0 - abs(pred_total - gt_total) / max(
            gt_total, 1
        )

        # Calculate species classification accuracy using Hungarian matching
        if len(pred_labels) > 0 and len(gt_labels) > 0:
            matched_labels = self._match_instances(predictions, ground_truth)
            if matched_labels:
                correct_species = sum(1 for p, g in matched_labels if p == g)
                metrics["species_classification_accuracy"] = correct_species / len(
                    matched_labels
                )
            else:
                metrics["species_classification_accuracy"] = 0.0
        else:
            metrics["species_classification_accuracy"] = 0.0

        # Calculate precision and recall for each species
        for species_id, species_name in [(1, "cacao"), (2, "shade")]:
            tp = sum(
                1 for p, g in matched_labels if p == species_id and g == species_id
            )
            fp = sum(
                1 for p, g in matched_labels if p == species_id and g != species_id
            )
            fn = sum(
                1 for p, g in matched_labels if p != species_id and g == species_id
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics[f"{species_name}_precision"] = precision
            metrics[f"{species_name}_recall"] = recall
            metrics[f"{species_name}_f1"] = f1

        return metrics

    def calculate_canopy_coverage(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """
        Calculate canopy coverage percentage and related metrics.
        """
        metrics = {}

        # Get semantic segmentation masks
        pred_semantic = predictions.get("semantic_map", None)
        gt_semantic = ground_truth.get("semantic_map", None)

        if pred_semantic is not None and gt_semantic is not None:
            # Calculate canopy pixels (cacao + shade trees)
            pred_canopy = (pred_semantic == 1) | (pred_semantic == 2)
            gt_canopy = (gt_semantic == 1) | (gt_semantic == 2)

            # Coverage percentage
            pred_coverage = np.sum(pred_canopy) / pred_canopy.size
            gt_coverage = np.sum(gt_canopy) / gt_canopy.size

            metrics["predicted_canopy_coverage"] = pred_coverage
            metrics["ground_truth_canopy_coverage"] = gt_coverage
            metrics["canopy_coverage_error"] = abs(pred_coverage - gt_coverage)
            metrics["canopy_coverage_relative_error"] = abs(
                pred_coverage - gt_coverage
            ) / max(gt_coverage, 0.01)

            # Calculate IoU for canopy coverage
            intersection = np.sum(pred_canopy & gt_canopy)
            union = np.sum(pred_canopy | gt_canopy)
            metrics["canopy_iou"] = intersection / union if union > 0 else 0

            # Separate coverage by species
            pred_cacao_coverage = np.sum(pred_semantic == 1) / pred_semantic.size
            pred_shade_coverage = np.sum(pred_semantic == 2) / pred_semantic.size
            gt_cacao_coverage = np.sum(gt_semantic == 1) / gt_semantic.size
            gt_shade_coverage = np.sum(gt_semantic == 2) / gt_semantic.size

            metrics["cacao_coverage_accuracy"] = 1.0 - abs(
                pred_cacao_coverage - gt_cacao_coverage
            )
            metrics["shade_coverage_accuracy"] = 1.0 - abs(
                pred_shade_coverage - gt_shade_coverage
            )

        # Crown-based canopy metrics
        if "crown_map" in predictions:
            crown_map = predictions["crown_map"]
            metrics["avg_crown_diameter"] = np.mean(crown_map[crown_map > 0])
            metrics["std_crown_diameter"] = np.std(crown_map[crown_map > 0])
            metrics["total_crown_area"] = (
                np.sum(crown_map > 0) * 0.25
            )  # Assuming 0.5m pixel resolution

        return metrics

    def calculate_spacing_uniformity(self, predictions: Dict) -> Dict:
        """
        Calculate tree spacing uniformity metrics using spatial statistics.
        """
        metrics = {}

        # Get tree positions
        tree_positions = self._extract_tree_positions(predictions)

        if len(tree_positions) < 3:
            metrics["spacing_uniformity"] = 0.0
            metrics["avg_nearest_neighbor_distance"] = 0.0
            return metrics

        # Calculate pairwise distances
        positions_array = np.array(tree_positions)
        dist_matrix = distance_matrix(positions_array, positions_array)

        # Nearest neighbor distances (excluding self)
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_distances = np.min(dist_matrix, axis=1)

        # Average nearest neighbor distance
        avg_nn_distance = np.mean(nearest_distances)
        std_nn_distance = np.std(nearest_distances)

        metrics["avg_nearest_neighbor_distance"] = avg_nn_distance
        metrics["std_nearest_neighbor_distance"] = std_nn_distance
        metrics["cv_nearest_neighbor"] = (
            std_nn_distance / avg_nn_distance if avg_nn_distance > 0 else 0
        )

        # Clark-Evans Index for spatial uniformity
        # R = observed mean distance / expected mean distance
        area = self._calculate_polygon_area(positions_array)
        density = len(tree_positions) / area if area > 0 else 0
        expected_distance = 0.5 * np.sqrt(1 / density) if density > 0 else 0

        clark_evans_r = (
            avg_nn_distance / expected_distance if expected_distance > 0 else 0
        )
        metrics["clark_evans_index"] = clark_evans_r

        # Interpretation: R = 1 (random), R < 1 (clustered), R > 1 (uniform)
        if clark_evans_r < 0.8:
            metrics["spatial_pattern"] = "clustered"
        elif clark_evans_r > 1.2:
            metrics["spatial_pattern"] = "uniform"
        else:
            metrics["spatial_pattern"] = "random"

        # Ripley's K function for multi-scale analysis
        k_values = self._calculate_ripleys_k(positions_array)
        metrics["ripleys_k"] = k_values

        # Voronoi cell analysis for spacing regularity
        if len(tree_positions) >= 4:
            try:
                vor = Voronoi(positions_array)
                # Calculate area of finite Voronoi cells
                cell_areas = []
                for region_index in vor.point_region:
                    region = vor.regions[region_index]
                    if len(region) > 0 and -1 not in region:
                        polygon = [vor.vertices[i] for i in region]
                        area = self._polygon_area(polygon)
                        if area > 0:
                            cell_areas.append(area)

                if cell_areas:
                    metrics["voronoi_area_mean"] = np.mean(cell_areas)
                    metrics["voronoi_area_std"] = np.std(cell_areas)
                    metrics["voronoi_area_cv"] = np.std(cell_areas) / np.mean(
                        cell_areas
                    )
            except:
                pass

        return metrics

    def analyze_shade_distribution(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """
        Analyze shade tree distribution and effectiveness.
        """
        metrics = {}

        # Extract shade tree positions
        pred_instances = predictions.get("instances", {})
        pred_labels = pred_instances.get("labels", np.array([]))
        pred_boxes = pred_instances.get("boxes", np.array([]))

        shade_positions = []
        cacao_positions = []

        for i, label in enumerate(pred_labels):
            if len(pred_boxes) > i:
                box = pred_boxes[i]
                center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                if label == 2:  # Shade tree
                    shade_positions.append(center)
                elif label == 1:  # Cacao tree
                    cacao_positions.append(center)

        if len(shade_positions) == 0:
            metrics["shade_distribution_score"] = 0.0
            return metrics

        # Calculate shade coverage effectiveness
        if len(cacao_positions) > 0:
            shade_effectiveness = self._calculate_shade_effectiveness(
                shade_positions, cacao_positions
            )
            metrics.update(shade_effectiveness)

        # Shade tree density
        total_area = self._estimate_image_area(predictions)
        shade_density = len(shade_positions) / total_area if total_area > 0 else 0
        metrics["shade_tree_density"] = shade_density

        # Shade to cacao ratio
        shade_cacao_ratio = (
            len(shade_positions) / len(cacao_positions)
            if len(cacao_positions) > 0
            else 0
        )
        metrics["shade_to_cacao_ratio"] = shade_cacao_ratio

        # Optimal ratio is typically 1:4 to 1:6 for cabruca systems
        optimal_ratio = 1 / 5  # 1 shade tree per 5 cacao trees
        metrics["shade_ratio_optimality"] = (
            1.0 - abs(shade_cacao_ratio - optimal_ratio) / optimal_ratio
        )

        # Shade distribution uniformity using coefficient of variation
        if len(shade_positions) > 1:
            shade_array = np.array(shade_positions)
            dist_matrix = distance_matrix(shade_array, shade_array)
            np.fill_diagonal(dist_matrix, np.inf)
            nn_distances = np.min(dist_matrix, axis=1)

            shade_spacing_cv = np.std(nn_distances) / np.mean(nn_distances)
            metrics["shade_spacing_uniformity"] = 1.0 / (1.0 + shade_spacing_cv)

        return metrics

    def calculate_instance_metrics(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """
        Calculate mAP and other instance segmentation metrics.
        """
        metrics = {}

        # Extract predictions and ground truth
        pred_boxes = predictions.get("instances", {}).get("boxes", np.array([]))
        pred_labels = predictions.get("instances", {}).get("labels", np.array([]))
        pred_scores = predictions.get("instances", {}).get("scores", np.array([]))
        pred_masks = predictions.get("instances", {}).get("masks", np.array([]))

        gt_boxes = ground_truth.get("instances", {}).get("boxes", np.array([]))
        gt_labels = ground_truth.get("instances", {}).get("labels", np.array([]))
        gt_masks = ground_truth.get("instances", {}).get("masks", np.array([]))

        # Calculate mAP for different IoU thresholds
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        ap_scores = []

        for iou_thresh in iou_thresholds:
            ap = self._calculate_ap_at_iou(
                pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresh
            )
            ap_scores.append(ap)

        # COCO-style metrics
        metrics["mAP"] = np.mean(ap_scores)  # mAP@[0.5:0.95]
        metrics["mAP_50"] = ap_scores[0] if ap_scores else 0  # mAP@0.5
        metrics["mAP_75"] = ap_scores[5] if len(ap_scores) > 5 else 0  # mAP@0.75

        # Calculate per-class AP
        for class_id, class_name in [(1, "cacao"), (2, "shade")]:
            class_ap = self._calculate_class_ap(
                pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, class_id
            )
            metrics[f"AP_{class_name}"] = class_ap

        # Mask-based metrics if available
        if len(pred_masks) > 0 and len(gt_masks) > 0:
            mask_iou = self._calculate_mask_iou(pred_masks, gt_masks)
            metrics["mask_mIoU"] = np.mean(mask_iou) if len(mask_iou) > 0 else 0

        # Detection metrics
        matched = self._match_predictions_to_gt(
            pred_boxes, gt_boxes, self.iou_threshold
        )
        tp = len(matched)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics["detection_precision"] = precision
        metrics["detection_recall"] = recall
        metrics["detection_f1"] = f1

        return metrics

    def calculate_crown_iou(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """
        Calculate IoU specifically for individual tree crowns.
        """
        metrics = {}

        pred_masks = predictions.get("instances", {}).get("masks", np.array([]))
        gt_masks = ground_truth.get("instances", {}).get("masks", np.array([]))

        if len(pred_masks) == 0 or len(gt_masks) == 0:
            metrics["crown_iou_mean"] = 0.0
            return metrics

        # Match predictions to ground truth
        iou_matrix = self._calculate_iou_matrix(pred_masks, gt_masks)

        # Hungarian matching for optimal assignment
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched_ious = []
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] > 0.1:  # Minimum overlap threshold
                matched_ious.append(iou_matrix[i, j])

        if matched_ious:
            metrics["crown_iou_mean"] = np.mean(matched_ious)
            metrics["crown_iou_std"] = np.std(matched_ious)
            metrics["crown_iou_median"] = np.median(matched_ious)
            metrics["crown_iou_min"] = np.min(matched_ious)
            metrics["crown_iou_max"] = np.max(matched_ious)

            # Percentage of crowns with IoU > thresholds
            for thresh in [0.5, 0.7, 0.9]:
                pct = np.sum(np.array(matched_ious) > thresh) / len(matched_ious)
                metrics[f"crown_iou_above_{int(thresh*100)}"] = pct
        else:
            metrics["crown_iou_mean"] = 0.0

        # Crown size accuracy
        pred_areas = [np.sum(mask) for mask in pred_masks]
        gt_areas = [np.sum(mask) for mask in gt_masks]

        if pred_areas and gt_areas:
            pred_mean_area = np.mean(pred_areas)
            gt_mean_area = np.mean(gt_areas)
            metrics["crown_area_accuracy"] = (
                1.0 - abs(pred_mean_area - gt_mean_area) / gt_mean_area
            )

        return metrics

    def compare_with_plantation_data(
        self, predictions: Dict, plantation_file: str
    ) -> Dict:
        """
        Compare predictions with existing plantation data from JSON file.
        """
        metrics = {}

        try:
            with open(plantation_file, "r") as f:
                plantation_data = json.load(f)
        except:
            return metrics

        # Extract plantation tree coordinates
        plantation_trees = []
        plantation_species = []

        if "trees" in plantation_data:
            for tree in plantation_data["trees"]:
                if "coordinates" in tree:
                    plantation_trees.append(tree["coordinates"])
                    plantation_species.append(tree.get("species", "unknown"))
        elif "features" in plantation_data:  # GeoJSON format
            for feature in plantation_data["features"]:
                if "geometry" in feature:
                    coords = feature["geometry"].get("coordinates", [])
                    if coords:
                        plantation_trees.append(coords[:2])  # [lon, lat]
                        species = feature.get("properties", {}).get(
                            "species", "unknown"
                        )
                        plantation_species.append(species)

        if not plantation_trees:
            return metrics

        # Extract predicted tree positions
        pred_positions = self._extract_tree_positions(predictions)

        # Match predicted trees to plantation data
        if pred_positions and plantation_trees:
            matched_trees = self._match_trees_by_distance(
                pred_positions, plantation_trees, self.distance_threshold
            )

            total_plantation = len(plantation_trees)
            total_predicted = len(pred_positions)
            total_matched = len(matched_trees)

            metrics["plantation_detection_rate"] = (
                total_matched / total_plantation if total_plantation > 0 else 0
            )
            metrics["plantation_precision"] = (
                total_matched / total_predicted if total_predicted > 0 else 0
            )
            metrics["plantation_recall"] = (
                total_matched / total_plantation if total_plantation > 0 else 0
            )

            # F1 score
            if metrics["plantation_precision"] + metrics["plantation_recall"] > 0:
                metrics["plantation_f1"] = (
                    2 * metrics["plantation_precision"] * metrics["plantation_recall"]
                ) / (metrics["plantation_precision"] + metrics["plantation_recall"])
            else:
                metrics["plantation_f1"] = 0.0

            # Calculate position accuracy for matched trees
            if matched_trees:
                distances = [d for _, _, d in matched_trees]
                metrics["avg_position_error"] = np.mean(distances)
                metrics["std_position_error"] = np.std(distances)
                metrics["max_position_error"] = np.max(distances)

                # Percentage within different distance thresholds
                for thresh in [0.5, 1.0, 2.0, 5.0]:
                    within_thresh = sum(1 for d in distances if d <= thresh)
                    metrics[f"trees_within_{thresh}m"] = within_thresh / len(distances)

        # Species classification accuracy if available
        if "instances" in predictions and plantation_species:
            pred_labels = predictions["instances"].get("labels", [])
            species_mapping = {"cacao": 1, "shade": 2}

            if matched_trees and pred_labels is not None:
                correct_species = 0
                for pred_idx, plant_idx, _ in matched_trees:
                    if pred_idx < len(pred_labels) and plant_idx < len(
                        plantation_species
                    ):
                        pred_species = pred_labels[pred_idx]
                        true_species = species_mapping.get(
                            plantation_species[plant_idx].lower(), 0
                        )
                        if pred_species == true_species:
                            correct_species += 1

                metrics["plantation_species_accuracy"] = correct_species / len(
                    matched_trees
                )

        return metrics

    # Helper methods

    def _extract_tree_positions(self, predictions: Dict) -> List[Tuple[float, float]]:
        """Extract tree center positions from predictions."""
        positions = []
        boxes = predictions.get("instances", {}).get("boxes", np.array([]))

        for box in boxes:
            if len(box) >= 4:
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                positions.append((center_x, center_y))

        return positions

    def _match_instances(
        self, predictions: Dict, ground_truth: Dict
    ) -> List[Tuple[int, int]]:
        """Match predicted instances to ground truth based on IoU."""
        pred_boxes = predictions.get("instances", {}).get("boxes", np.array([]))
        pred_labels = predictions.get("instances", {}).get("labels", np.array([]))
        gt_boxes = ground_truth.get("instances", {}).get("boxes", np.array([]))
        gt_labels = ground_truth.get("instances", {}).get("labels", np.array([]))

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return []

        # Calculate IoU matrix
        iou_matrix = self._calculate_box_iou_matrix(pred_boxes, gt_boxes)

        # Hungarian matching
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched = []
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] >= self.iou_threshold:
                if i < len(pred_labels) and j < len(gt_labels):
                    matched.append((pred_labels[i], gt_labels[j]))

        return matched

    def _calculate_box_iou_matrix(
        self, boxes1: np.ndarray, boxes2: np.ndarray
    ) -> np.ndarray:
        """Calculate IoU matrix between two sets of boxes."""
        n1 = len(boxes1)
        n2 = len(boxes2)
        iou_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                iou_matrix[i, j] = self._box_iou(boxes1[i], boxes2[j])

        return iou_matrix

    def _box_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _calculate_iou_matrix(
        self, masks1: np.ndarray, masks2: np.ndarray
    ) -> np.ndarray:
        """Calculate IoU matrix between two sets of masks."""
        n1 = len(masks1)
        n2 = len(masks2)
        iou_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                intersection = np.sum(masks1[i] & masks2[j])
                union = np.sum(masks1[i] | masks2[j])
                iou_matrix[i, j] = intersection / union if union > 0 else 0

        return iou_matrix

    def _calculate_polygon_area(self, points: np.ndarray) -> float:
        """Calculate area of polygon formed by convex hull of points."""
        from scipy.spatial import ConvexHull

        if len(points) < 3:
            return 0.0

        try:
            hull = ConvexHull(points)
            return hull.volume  # In 2D, volume is actually area
        except:
            return 0.0

    def _polygon_area(self, vertices: List) -> float:
        """Calculate area of a polygon using Shoelace formula."""
        n = len(vertices)
        if n < 3:
            return 0

        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]

        return abs(area) / 2

    def _calculate_ripleys_k(
        self, points: np.ndarray, radii: List[float] = None
    ) -> Dict:
        """Calculate Ripley's K function for spatial pattern analysis."""
        if radii is None:
            max_dist = np.max(distance_matrix(points, points))
            radii = np.linspace(0, max_dist / 4, 10)

        n = len(points)
        area = self._calculate_polygon_area(points)

        k_values = {}
        for r in radii:
            count = 0
            dist_matrix = distance_matrix(points, points)
            count = np.sum(dist_matrix <= r) - n  # Exclude self-pairs
            k_values[f"K_{r:.1f}"] = (area * count) / (n * (n - 1))

        return k_values

    def _calculate_shade_effectiveness(
        self, shade_positions: List, cacao_positions: List
    ) -> Dict:
        """Calculate how effectively shade trees cover cacao trees."""
        metrics = {}

        if not shade_positions or not cacao_positions:
            return metrics

        shade_array = np.array(shade_positions)
        cacao_array = np.array(cacao_positions)

        # For each cacao tree, find distance to nearest shade tree
        distances_to_shade = []
        for cacao_pos in cacao_array:
            distances = np.sqrt(np.sum((shade_array - cacao_pos) ** 2, axis=1))
            min_distance = np.min(distances)
            distances_to_shade.append(min_distance)

        # Effective shade radius (typically 10-15 meters for mature shade trees)
        effective_radius = 12.0

        # Percentage of cacao trees within effective shade radius
        shaded_cacao = sum(1 for d in distances_to_shade if d <= effective_radius)
        metrics["cacao_shade_coverage"] = shaded_cacao / len(cacao_positions)

        # Average distance from cacao to nearest shade
        metrics["avg_cacao_to_shade_distance"] = np.mean(distances_to_shade)
        metrics["std_cacao_to_shade_distance"] = np.std(distances_to_shade)

        # Shade distribution score (higher is better)
        # Based on how evenly distributed shade trees are among cacao
        if len(distances_to_shade) > 1:
            cv = np.std(distances_to_shade) / np.mean(distances_to_shade)
            metrics["shade_distribution_score"] = 1.0 / (1.0 + cv)
        else:
            metrics["shade_distribution_score"] = 0.0

        return metrics

    def _estimate_image_area(self, predictions: Dict) -> float:
        """Estimate total area of the image in square meters."""
        # Assuming 0.5m pixel resolution (can be adjusted)
        pixel_size = 0.5  # meters per pixel

        if "semantic_map" in predictions:
            h, w = predictions["semantic_map"].shape
            return h * w * pixel_size * pixel_size

        return 1000.0  # Default area

    def _match_trees_by_distance(
        self, pred_positions: List, gt_positions: List, threshold: float
    ) -> List[Tuple[int, int, float]]:
        """Match trees based on distance threshold."""
        matched = []
        pred_array = np.array(pred_positions)
        gt_array = np.array(gt_positions)

        dist_matrix = distance_matrix(pred_array, gt_array)

        # Greedy matching
        used_pred = set()
        used_gt = set()

        while True:
            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            min_dist = dist_matrix[min_idx]

            if min_dist > threshold:
                break

            if min_idx[0] not in used_pred and min_idx[1] not in used_gt:
                matched.append((min_idx[0], min_idx[1], min_dist))
                used_pred.add(min_idx[0])
                used_gt.add(min_idx[1])

            dist_matrix[min_idx] = np.inf

        return matched

    def _calculate_ap_at_iou(
        self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold
    ):
        """Calculate Average Precision at specific IoU threshold."""
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return 0.0

        # Sort predictions by score
        sorted_idx = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sorted_idx]
        pred_labels = pred_labels[sorted_idx]
        pred_scores = pred_scores[sorted_idx]

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))

        gt_matched = set()

        for i, (box, label) in enumerate(zip(pred_boxes, pred_labels)):
            best_iou = 0
            best_gt = -1

            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if j in gt_matched:
                    continue

                if label == gt_label:
                    iou = self._box_iou(box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = j

            if best_iou >= iou_threshold:
                tp[i] = 1
                gt_matched.add(best_gt)
            else:
                fp[i] = 1

        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / len(gt_boxes)

        # Calculate AP using 11-point interpolation
        ap = 0
        for r in np.linspace(0, 1, 11):
            prec_at_recall = precision[recall >= r]
            if len(prec_at_recall) > 0:
                ap += np.max(prec_at_recall) / 11

        return ap

    def _calculate_class_ap(
        self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, class_id
    ):
        """Calculate AP for a specific class."""
        # Filter by class
        pred_mask = pred_labels == class_id
        gt_mask = gt_labels == class_id

        if not np.any(pred_mask) or not np.any(gt_mask):
            return 0.0

        return self._calculate_ap_at_iou(
            pred_boxes[pred_mask],
            pred_labels[pred_mask],
            pred_scores[pred_mask],
            gt_boxes[gt_mask],
            gt_labels[gt_mask],
            0.5,
        )

    def _match_predictions_to_gt(self, pred_boxes, gt_boxes, iou_threshold):
        """Match predictions to ground truth based on IoU."""
        matched = []

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return matched

        iou_matrix = self._calculate_box_iou_matrix(pred_boxes, gt_boxes)

        # Greedy matching
        for i in range(len(pred_boxes)):
            if len(gt_boxes) > 0:
                best_j = np.argmax(iou_matrix[i])
                if iou_matrix[i, best_j] >= iou_threshold:
                    matched.append((i, best_j))
                    iou_matrix[:, best_j] = 0  # Mark as used

        return matched

    def _calculate_mask_iou(self, pred_masks, gt_masks):
        """Calculate IoU for masks."""
        ious = []

        iou_matrix = self._calculate_iou_matrix(pred_masks, gt_masks)

        # Hungarian matching
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] > 0:
                ious.append(iou_matrix[i, j])

        return ious


def visualize_evaluation_results(metrics: Dict, save_path: str = None):
    """
    Create comprehensive visualization of evaluation metrics.
    """
    fig = plt.figure(figsize=(20, 12))

    # Tree count accuracy
    ax1 = plt.subplot(2, 4, 1)
    categories = ["Cacao", "Shade", "Total"]
    accuracies = [
        metrics.get("cacao_count_accuracy", 0),
        metrics.get("shade_count_accuracy", 0),
        metrics.get("total_count_accuracy", 0),
    ]
    bars = ax1.bar(categories, accuracies, color=["brown", "green", "blue"])
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Tree Count Accuracy")
    ax1.set_ylim([0, 1])
    for bar, acc in zip(bars, accuracies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.2%}",
            ha="center",
        )

    # Species classification metrics
    ax2 = plt.subplot(2, 4, 2)
    species_metrics = ["Precision", "Recall", "F1"]
    cacao_scores = [
        metrics.get("cacao_precision", 0),
        metrics.get("cacao_recall", 0),
        metrics.get("cacao_f1", 0),
    ]
    shade_scores = [
        metrics.get("shade_precision", 0),
        metrics.get("shade_recall", 0),
        metrics.get("shade_f1", 0),
    ]
    x = np.arange(len(species_metrics))
    width = 0.35
    ax2.bar(x - width / 2, cacao_scores, width, label="Cacao", color="brown")
    ax2.bar(x + width / 2, shade_scores, width, label="Shade", color="green")
    ax2.set_xticks(x)
    ax2.set_xticklabels(species_metrics)
    ax2.set_ylabel("Score")
    ax2.set_title("Species Classification Metrics")
    ax2.legend()
    ax2.set_ylim([0, 1])

    # Canopy coverage
    ax3 = plt.subplot(2, 4, 3)
    coverage_data = {
        "Predicted": metrics.get("predicted_canopy_coverage", 0),
        "Ground Truth": metrics.get("ground_truth_canopy_coverage", 0),
    }
    ax3.pie(
        coverage_data.values(),
        labels=coverage_data.keys(),
        autopct="%1.1f%%",
        colors=["lightgreen", "darkgreen"],
    )
    ax3.set_title("Canopy Coverage Comparison")

    # Spacing uniformity
    ax4 = plt.subplot(2, 4, 4)
    spacing_metrics = {
        "Avg NN Distance": metrics.get("avg_nearest_neighbor_distance", 0),
        "Std NN Distance": metrics.get("std_nearest_neighbor_distance", 0),
        "Clark-Evans Index": metrics.get("clark_evans_index", 0),
    }
    ax4.barh(
        list(spacing_metrics.keys()),
        list(spacing_metrics.values()),
        color=["blue", "orange", "green"],
    )
    ax4.set_xlabel("Value")
    ax4.set_title("Spacing Uniformity Metrics")

    # Instance segmentation mAP
    ax5 = plt.subplot(2, 4, 5)
    map_scores = {
        "mAP@0.5": metrics.get("mAP_50", 0),
        "mAP@0.75": metrics.get("mAP_75", 0),
        "mAP@[0.5:0.95]": metrics.get("mAP", 0),
    }
    ax5.bar(map_scores.keys(), map_scores.values(), color="purple")
    ax5.set_ylabel("Average Precision")
    ax5.set_title("Instance Segmentation mAP")
    ax5.set_ylim([0, 1])
    for i, (k, v) in enumerate(map_scores.items()):
        ax5.text(i, v + 0.01, f"{v:.3f}", ha="center")

    # Crown IoU distribution
    ax6 = plt.subplot(2, 4, 6)
    crown_metrics = [
        metrics.get("crown_iou_mean", 0),
        metrics.get("crown_iou_std", 0),
        metrics.get("crown_iou_median", 0),
    ]
    labels = ["Mean", "Std", "Median"]
    ax6.bar(labels, crown_metrics, color=["blue", "red", "green"])
    ax6.set_ylabel("IoU")
    ax6.set_title("Crown IoU Statistics")
    ax6.set_ylim([0, 1])

    # Shade distribution
    ax7 = plt.subplot(2, 4, 7)
    shade_data = {
        "Shade Coverage": metrics.get("cacao_shade_coverage", 0),
        "Distribution Score": metrics.get("shade_distribution_score", 0),
        "Ratio Optimality": metrics.get("shade_ratio_optimality", 0),
    }
    ax7.bar(
        shade_data.keys(),
        shade_data.values(),
        color=["darkgreen", "lightgreen", "olive"],
    )
    ax7.set_ylabel("Score")
    ax7.set_title("Shade Distribution Analysis")
    ax7.set_ylim([0, 1])

    # Plantation comparison
    ax8 = plt.subplot(2, 4, 8)
    if "plantation_detection_rate" in metrics:
        plantation_metrics = {
            "Detection Rate": metrics.get("plantation_detection_rate", 0),
            "Precision": metrics.get("plantation_precision", 0),
            "Recall": metrics.get("plantation_recall", 0),
            "F1": metrics.get("plantation_f1", 0),
        }
        ax8.bar(plantation_metrics.keys(), plantation_metrics.values(), color="orange")
        ax8.set_ylabel("Score")
        ax8.set_title("Plantation Data Comparison")
        ax8.set_ylim([0, 1])
    else:
        ax8.text(0.5, 0.5, "No plantation data", ha="center", va="center")
        ax8.set_title("Plantation Data Comparison")

    plt.suptitle("Agroforestry Evaluation Metrics", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    return fig

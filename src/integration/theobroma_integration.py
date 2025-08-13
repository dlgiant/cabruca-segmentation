"""
Integration module for connecting Cabruca ML segmentation with theobroma-digital project.
Compares ML detections with existing coordinates and updates plantation data.
"""

import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

warnings.filterwarnings("ignore")

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.agroforestry_metrics import AgroforestryMetrics
from inference.batch_inference import (
    BatchInferenceEngine,
    InferenceResult,
    TreeInstance,
)


@dataclass
class TreeComparison:
    """Container for tree comparison results."""

    plantation_id: int
    ml_tree_id: int
    species_match: bool
    position_error: float
    crown_diameter_diff: float
    ml_confidence: float
    status: str  # 'matched', 'ml_only', 'plantation_only'


class TheobromaIntegration:
    """
    Integration layer between ML segmentation and theobroma-digital project.
    """

    def __init__(self, model_path: str, plantation_data_path: str):
        """
        Initialize integration module.

        Args:
            model_path: Path to trained ML model
            plantation_data_path: Path to existing plantation-data.json
        """
        self.model_path = model_path
        self.plantation_data_path = plantation_data_path

        # Load ML model
        self.inference_engine = BatchInferenceEngine(
            model_path=model_path, device="auto"
        )

        # Load existing plantation data
        self.plantation_data = self._load_plantation_data()

        # Metrics evaluator
        self.evaluator = AgroforestryMetrics()

    def _load_plantation_data(self) -> Dict:
        """Load existing plantation data from JSON."""
        with open(self.plantation_data_path, "r") as f:
            return json.load(f)

    def compare_with_ml_detection(
        self, image_path: str, distance_threshold: float = 2.0
    ) -> Dict:
        """
        Compare ML-detected trees with existing plantation coordinates.

        Args:
            image_path: Path to plantation image
            distance_threshold: Maximum distance (meters) for matching

        Returns:
            Comparison results and statistics
        """
        # Run ML inference
        ml_result = self.inference_engine.process_single(image_path)

        # Extract plantation trees
        plantation_trees = self._extract_plantation_trees()

        # Match trees
        comparisons = self._match_trees(
            ml_result.trees, plantation_trees, distance_threshold
        )

        # Calculate statistics
        stats = self._calculate_comparison_stats(comparisons)

        # Generate comparison report
        report = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "ml_trees_detected": len(ml_result.trees),
            "plantation_trees_expected": len(plantation_trees),
            "comparisons": [asdict(c) for c in comparisons],
            "statistics": stats,
            "ml_result": ml_result,
        }

        return report

    def _extract_plantation_trees(self) -> List[Dict]:
        """Extract tree information from plantation data."""
        trees = []

        if "trees" in self.plantation_data:
            trees = self.plantation_data["trees"]
        elif "features" in self.plantation_data:  # GeoJSON format
            for feature in self.plantation_data["features"]:
                tree = {
                    "id": feature.get("properties", {}).get("id"),
                    "species": feature.get("properties", {}).get("species"),
                    "coordinates": feature.get("geometry", {}).get("coordinates", []),
                    "crown_diameter_m": feature.get("properties", {}).get(
                        "crown_diameter", 0
                    ),
                }
                trees.append(tree)

        return trees

    def _match_trees(
        self,
        ml_trees: List[TreeInstance],
        plantation_trees: List[Dict],
        threshold: float,
    ) -> List[TreeComparison]:
        """
        Match ML-detected trees with plantation coordinates.

        Uses Hungarian algorithm for optimal matching.
        """
        comparisons = []

        if not ml_trees or not plantation_trees:
            # Handle empty cases
            for ml_tree in ml_trees:
                comparisons.append(
                    TreeComparison(
                        plantation_id=-1,
                        ml_tree_id=ml_tree.id,
                        species_match=False,
                        position_error=float("inf"),
                        crown_diameter_diff=float("inf"),
                        ml_confidence=ml_tree.confidence,
                        status="ml_only",
                    )
                )

            for plant_tree in plantation_trees:
                comparisons.append(
                    TreeComparison(
                        plantation_id=plant_tree.get("id", -1),
                        ml_tree_id=-1,
                        species_match=False,
                        position_error=float("inf"),
                        crown_diameter_diff=float("inf"),
                        ml_confidence=0.0,
                        status="plantation_only",
                    )
                )

            return comparisons

        # Build distance matrix
        ml_positions = np.array([tree.centroid for tree in ml_trees])
        plant_positions = np.array(
            [tree.get("coordinates", [0, 0])[:2] for tree in plantation_trees]
        )

        dist_matrix = distance_matrix(ml_positions, plant_positions)

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        matched_ml = set()
        matched_plant = set()

        # Process matches
        for ml_idx, plant_idx in zip(row_ind, col_ind):
            distance = dist_matrix[ml_idx, plant_idx]

            if distance <= threshold:
                ml_tree = ml_trees[ml_idx]
                plant_tree = plantation_trees[plant_idx]

                # Check species match
                ml_species = ml_tree.species
                plant_species = plant_tree.get("species", "").lower()
                species_match = ml_species == plant_species

                # Calculate crown diameter difference
                ml_crown = ml_tree.crown_diameter
                plant_crown = plant_tree.get("crown_diameter_m", 0)
                crown_diff = abs(ml_crown - plant_crown)

                comparisons.append(
                    TreeComparison(
                        plantation_id=plant_tree.get("id", plant_idx),
                        ml_tree_id=ml_tree.id,
                        species_match=species_match,
                        position_error=distance,
                        crown_diameter_diff=crown_diff,
                        ml_confidence=ml_tree.confidence,
                        status="matched",
                    )
                )

                matched_ml.add(ml_idx)
                matched_plant.add(plant_idx)

        # Add unmatched ML trees
        for ml_idx, ml_tree in enumerate(ml_trees):
            if ml_idx not in matched_ml:
                comparisons.append(
                    TreeComparison(
                        plantation_id=-1,
                        ml_tree_id=ml_tree.id,
                        species_match=False,
                        position_error=float("inf"),
                        crown_diameter_diff=float("inf"),
                        ml_confidence=ml_tree.confidence,
                        status="ml_only",
                    )
                )

        # Add unmatched plantation trees
        for plant_idx, plant_tree in enumerate(plantation_trees):
            if plant_idx not in matched_plant:
                comparisons.append(
                    TreeComparison(
                        plantation_id=plant_tree.get("id", plant_idx),
                        ml_tree_id=-1,
                        species_match=False,
                        position_error=float("inf"),
                        crown_diameter_diff=float("inf"),
                        ml_confidence=0.0,
                        status="plantation_only",
                    )
                )

        return comparisons

    def _calculate_comparison_stats(self, comparisons: List[TreeComparison]) -> Dict:
        """Calculate statistics from tree comparisons."""
        matched = [c for c in comparisons if c.status == "matched"]
        ml_only = [c for c in comparisons if c.status == "ml_only"]
        plant_only = [c for c in comparisons if c.status == "plantation_only"]

        stats = {
            "total_comparisons": len(comparisons),
            "matched_trees": len(matched),
            "ml_only_trees": len(ml_only),
            "plantation_only_trees": len(plant_only),
            "match_rate": len(matched) / max(len(comparisons), 1),
        }

        if matched:
            # Position accuracy
            position_errors = [c.position_error for c in matched]
            stats["avg_position_error"] = np.mean(position_errors)
            stats["max_position_error"] = np.max(position_errors)
            stats["position_rmse"] = np.sqrt(np.mean(np.array(position_errors) ** 2))

            # Species accuracy
            species_matches = [c.species_match for c in matched]
            stats["species_accuracy"] = sum(species_matches) / len(species_matches)

            # Crown diameter accuracy
            crown_diffs = [c.crown_diameter_diff for c in matched]
            stats["avg_crown_diff"] = np.mean(crown_diffs)
            stats["crown_rmse"] = np.sqrt(np.mean(np.array(crown_diffs) ** 2))

            # Confidence statistics
            confidences = [c.ml_confidence for c in matched]
            stats["avg_confidence"] = np.mean(confidences)
            stats["min_confidence"] = np.min(confidences)

        # Detection metrics
        if len(plant_only) > 0 or len(matched) > 0:
            stats["detection_recall"] = len(matched) / (len(matched) + len(plant_only))
        else:
            stats["detection_recall"] = 0.0

        if len(ml_only) > 0 or len(matched) > 0:
            stats["detection_precision"] = len(matched) / (len(matched) + len(ml_only))
        else:
            stats["detection_precision"] = 0.0

        # F1 score
        if stats["detection_precision"] + stats["detection_recall"] > 0:
            stats["detection_f1"] = (
                2 * stats["detection_precision"] * stats["detection_recall"]
            ) / (stats["detection_precision"] + stats["detection_recall"])
        else:
            stats["detection_f1"] = 0.0

        return stats

    def update_plantation_data(
        self, comparison_report: Dict, confidence_threshold: float = 0.7
    ) -> Dict:
        """
        Update plantation data with ML-derived attributes.

        Args:
            comparison_report: Results from compare_with_ml_detection
            confidence_threshold: Minimum confidence to accept ML updates

        Returns:
            Updated plantation data
        """
        updated_data = self.plantation_data.copy()
        ml_result = comparison_report["ml_result"]
        comparisons = comparison_report["comparisons"]

        # Create lookup for comparisons
        comp_by_plant_id = {
            c["plantation_id"]: c for c in comparisons if c["status"] == "matched"
        }
        comp_by_ml_id = {
            c["ml_tree_id"]: c for c in comparisons if c["status"] == "matched"
        }

        # Update existing trees
        if "trees" in updated_data:
            for tree in updated_data["trees"]:
                tree_id = tree.get("id")
                if tree_id in comp_by_plant_id:
                    comp = comp_by_plant_id[tree_id]

                    # Find corresponding ML tree
                    ml_tree = next(
                        (t for t in ml_result.trees if t.id == comp["ml_tree_id"]), None
                    )

                    if ml_tree and ml_tree.confidence >= confidence_threshold:
                        # Update with ML attributes
                        tree["ml_detected"] = True
                        tree["ml_confidence"] = float(ml_tree.confidence)
                        tree["ml_crown_diameter"] = float(ml_tree.crown_diameter)
                        tree["ml_crown_area"] = float(ml_tree.crown_area)
                        tree["ml_species"] = ml_tree.species
                        tree["position_error"] = float(comp["position_error"])
                        tree["last_detected"] = datetime.now().isoformat()

                        # Update coordinates if position error is large
                        if comp["position_error"] > 1.0:  # More than 1 meter off
                            tree["ml_coordinates"] = list(ml_tree.centroid)

        # Add newly detected trees (ML only)
        new_trees = []
        for comp in comparisons:
            if comp["status"] == "ml_only":
                ml_tree = next(
                    (t for t in ml_result.trees if t.id == comp["ml_tree_id"]), None
                )

                if ml_tree and ml_tree.confidence >= confidence_threshold:
                    new_tree = {
                        "id": f"ml_{ml_tree.id}_{datetime.now().timestamp()}",
                        "species": ml_tree.species,
                        "coordinates": list(ml_tree.centroid),
                        "crown_diameter_m": float(ml_tree.crown_diameter),
                        "crown_area_m2": float(ml_tree.crown_area),
                        "ml_confidence": float(ml_tree.confidence),
                        "source": "ml_detection",
                        "detected_date": datetime.now().isoformat(),
                    }
                    new_trees.append(new_tree)

        if new_trees:
            if "trees" not in updated_data:
                updated_data["trees"] = []
            updated_data["trees"].extend(new_trees)

        # Add ML analysis metadata
        if "ml_analysis" not in updated_data:
            updated_data["ml_analysis"] = []

        analysis_record = {
            "timestamp": comparison_report["timestamp"],
            "image_path": comparison_report["image_path"],
            "statistics": comparison_report["statistics"],
            "ml_trees_detected": comparison_report["ml_trees_detected"],
            "new_trees_added": len(new_trees),
        }
        updated_data["ml_analysis"].append(analysis_record)

        return updated_data

    def generate_health_metrics(self, ml_result: InferenceResult) -> Dict:
        """
        Generate plantation health metrics from ML analysis.

        Args:
            ml_result: ML inference result

        Returns:
            Dictionary of health metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "plantation_health": {},
            "tree_health": {},
            "canopy_health": {},
            "recommendations": [],
        }

        # Tree health metrics
        if ml_result.trees:
            crown_diameters = [t.crown_diameter for t in ml_result.trees]

            metrics["tree_health"] = {
                "total_trees": len(ml_result.trees),
                "cacao_trees": sum(1 for t in ml_result.trees if t.species == "cacao"),
                "shade_trees": sum(1 for t in ml_result.trees if t.species == "shade"),
                "avg_crown_diameter": float(np.mean(crown_diameters)),
                "crown_diameter_std": float(np.std(crown_diameters)),
                "detection_confidence": float(
                    np.mean([t.confidence for t in ml_result.trees])
                ),
            }

            # Assess crown health
            expected_crown_diameter = {"cacao": 3.5, "shade": 15.0}

            for species in ["cacao", "shade"]:
                species_trees = [t for t in ml_result.trees if t.species == species]
                if species_trees:
                    avg_crown = np.mean([t.crown_diameter for t in species_trees])
                    expected = expected_crown_diameter[species]
                    health_score = min(avg_crown / expected, 1.0)

                    metrics["tree_health"][f"{species}_crown_health"] = float(
                        health_score
                    )

                    if health_score < 0.7:
                        metrics["recommendations"].append(
                            f"{species.title()} trees show small crown diameter "
                            f"({avg_crown:.1f}m vs expected {expected}m). "
                            "Consider nutrient supplementation."
                        )

        # Canopy health metrics
        metrics["canopy_health"] = {
            "canopy_density": float(ml_result.canopy_density),
            "canopy_coverage": float(
                ml_result.metrics.get("cacao_tree_coverage", 0)
                + ml_result.metrics.get("shade_tree_coverage", 0)
            ),
            "understory_coverage": float(
                ml_result.metrics.get("understory_coverage", 0)
            ),
            "bare_soil_percentage": float(
                ml_result.metrics.get("bare_soil_coverage", 0)
            ),
        }

        # Assess canopy health
        optimal_canopy_density = 0.45  # 45% is optimal for cabruca
        canopy_health_score = (
            1.0
            - abs(ml_result.canopy_density - optimal_canopy_density)
            / optimal_canopy_density
        )
        metrics["canopy_health"]["health_score"] = float(canopy_health_score)

        if ml_result.canopy_density < 0.3:
            metrics["recommendations"].append(
                "Low canopy density detected. Consider planting additional shade trees."
            )
        elif ml_result.canopy_density > 0.6:
            metrics["recommendations"].append(
                "High canopy density may reduce cacao productivity. Consider selective pruning."
            )

        # Plantation-wide health score
        if ml_result.trees:
            shade_ratio = metrics["tree_health"]["shade_trees"] / max(
                metrics["tree_health"]["cacao_trees"], 1
            )
            optimal_ratio = 0.2  # 1:5 shade to cacao
            ratio_score = 1.0 - abs(shade_ratio - optimal_ratio) / optimal_ratio

            overall_health = np.mean(
                [
                    canopy_health_score,
                    ratio_score,
                    metrics["tree_health"].get("cacao_crown_health", 0.5),
                    metrics["tree_health"].get("shade_crown_health", 0.5),
                ]
            )

            metrics["plantation_health"] = {
                "overall_score": float(overall_health),
                "shade_ratio": float(shade_ratio),
                "ratio_score": float(ratio_score),
                "status": (
                    "healthy"
                    if overall_health > 0.7
                    else "moderate" if overall_health > 0.4 else "poor"
                ),
            }

            if shade_ratio < 0.15:
                metrics["recommendations"].append(
                    "Insufficient shade coverage. Plant more shade trees for optimal cacao growth."
                )
            elif shade_ratio > 0.25:
                metrics["recommendations"].append(
                    "Excessive shade may reduce cacao yield. Consider removing some shade trees."
                )

        # Bare soil assessment
        if metrics["canopy_health"]["bare_soil_percentage"] > 0.3:
            metrics["recommendations"].append(
                f"High bare soil percentage ({metrics['canopy_health']['bare_soil_percentage']:.1%}). "
                "Consider cover crops or mulching to prevent erosion."
            )

        return metrics

    def create_enhanced_visualization(
        self, image_path: str, ml_result: InferenceResult, plantation_data: Dict
    ) -> Dict:
        """
        Create enhanced visualization combining ML masks with plantation data.

        Args:
            image_path: Path to original image
            ml_result: ML inference result
            plantation_data: Updated plantation data

        Returns:
            Dictionary with visualization paths
        """
        import cv2
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Original image with plantation coordinates
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Plantation Coordinates")

        if "trees" in plantation_data:
            for tree in plantation_data["trees"]:
                coord = tree.get("coordinates", [])
                if len(coord) >= 2:
                    color = "green" if tree.get("species") == "cacao" else "brown"
                    circle = Circle(
                        (coord[0], coord[1]), 5, color=color, fill=False, linewidth=2
                    )
                    axes[0, 0].add_patch(circle)
        axes[0, 0].axis("off")

        # 2. ML detection overlay
        from inference.batch_inference import VisualizationTools

        overlay = VisualizationTools.create_overlay(image, ml_result, alpha=0.4)
        axes[0, 1].imshow(overlay)
        axes[0, 1].set_title("ML Detection")
        axes[0, 1].axis("off")

        # 3. Comparison view
        axes[0, 2].imshow(image)
        axes[0, 2].set_title("Comparison (Green=Match, Red=Mismatch)")

        # Show matched and unmatched trees
        comparison = self.compare_with_ml_detection(image_path)
        for comp in comparison["comparisons"]:
            if comp["status"] == "matched":
                # Draw matched trees in green
                ml_tree = next(
                    (t for t in ml_result.trees if t.id == comp["ml_tree_id"]), None
                )
                if ml_tree:
                    cx, cy = ml_tree.centroid
                    circle = Circle(
                        (cx, cy),
                        ml_tree.crown_diameter / 2,
                        color="green",
                        fill=False,
                        linewidth=2,
                    )
                    axes[0, 2].add_patch(circle)
            elif comp["status"] == "ml_only":
                # Draw ML-only trees in yellow
                ml_tree = next(
                    (t for t in ml_result.trees if t.id == comp["ml_tree_id"]), None
                )
                if ml_tree:
                    cx, cy = ml_tree.centroid
                    circle = Circle(
                        (cx, cy),
                        ml_tree.crown_diameter / 2,
                        color="yellow",
                        fill=False,
                        linewidth=2,
                    )
                    axes[0, 2].add_patch(circle)
        axes[0, 2].axis("off")

        # 4. Segmentation mask
        from inference.batch_inference import BatchInferenceEngine

        semantic_colored = BatchInferenceEngine.SEMANTIC_COLORS[ml_result.semantic_map]
        axes[1, 0].imshow(semantic_colored.astype(np.uint8))
        axes[1, 0].set_title("Semantic Segmentation")
        axes[1, 0].axis("off")

        # 5. Confidence heatmap
        confidence_map = np.zeros(image.shape[:2])
        for tree in ml_result.trees:
            cx, cy = [int(x) for x in tree.centroid]
            radius = int(tree.crown_diameter / 2)
            cv2.circle(confidence_map, (cx, cy), radius, tree.confidence, -1)

        im = axes[1, 1].imshow(confidence_map, cmap="RdYlGn", vmin=0, vmax=1)
        axes[1, 1].set_title("Detection Confidence")
        axes[1, 1].axis("off")
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # 6. Health metrics panel
        axes[1, 2].axis("off")
        health_metrics = self.generate_health_metrics(ml_result)

        health_text = f"""
        Plantation Health Report
        {'='*30}
        
        Overall Health: {health_metrics['plantation_health'].get('overall_score', 0):.2%}
        Status: {health_metrics['plantation_health'].get('status', 'N/A').upper()}
        
        Tree Counts:
        - Cacao: {health_metrics['tree_health'].get('cacao_trees', 0)}
        - Shade: {health_metrics['tree_health'].get('shade_trees', 0)}
        - Ratio: {health_metrics['plantation_health'].get('shade_ratio', 0):.2f}
        
        Canopy Health:
        - Density: {health_metrics['canopy_health'].get('canopy_density', 0):.2%}
        - Coverage: {health_metrics['canopy_health'].get('canopy_coverage', 0):.2%}
        - Bare Soil: {health_metrics['canopy_health'].get('bare_soil_percentage', 0):.2%}
        
        Detection Stats:
        - Matched Trees: {comparison['statistics'].get('matched_trees', 0)}
        - Position RMSE: {comparison['statistics'].get('position_rmse', 0):.2f}m
        - Species Accuracy: {comparison['statistics'].get('species_accuracy', 0):.2%}
        """

        axes[1, 2].text(
            0.05,
            0.95,
            health_text,
            transform=axes[1, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.suptitle(
            "Theobroma Digital - ML Integration Analysis",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save visualization
        output_path = (
            Path(image_path).parent
            / f"{Path(image_path).stem}_integration_analysis.png"
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return {
            "visualization_path": str(output_path),
            "comparison_stats": comparison["statistics"],
            "health_metrics": health_metrics,
        }


class PlantationDataUpdater:
    """
    Updates and maintains plantation data with ML insights.
    """

    @staticmethod
    def merge_ml_insights(original_data: Dict, ml_updates: Dict) -> Dict:
        """
        Merge ML insights with original plantation data.

        Args:
            original_data: Original plantation data
            ml_updates: ML-derived updates

        Returns:
            Merged data with version tracking
        """
        merged = original_data.copy()

        # Add version tracking
        if "version_history" not in merged:
            merged["version_history"] = []

        version_entry = {
            "version": len(merged["version_history"]) + 1,
            "timestamp": datetime.now().isoformat(),
            "source": "ml_integration",
            "changes": [],
        }

        # Track changes
        changes = []

        # Update trees
        if "trees" in ml_updates:
            tree_map = {t.get("id"): t for t in merged.get("trees", [])}

            for ml_tree in ml_updates["trees"]:
                tree_id = ml_tree.get("id")

                if tree_id in tree_map:
                    # Update existing tree
                    original = tree_map[tree_id].copy()
                    tree_map[tree_id].update(ml_tree)

                    # Track changes
                    changed_fields = []
                    for key in ml_tree:
                        if key not in original or original[key] != ml_tree[key]:
                            changed_fields.append(key)

                    if changed_fields:
                        changes.append(
                            {
                                "type": "update",
                                "tree_id": tree_id,
                                "fields": changed_fields,
                            }
                        )
                else:
                    # New tree
                    if "trees" not in merged:
                        merged["trees"] = []
                    merged["trees"].append(ml_tree)
                    changes.append({"type": "add", "tree_id": tree_id})

        version_entry["changes"] = changes
        merged["version_history"].append(version_entry)

        # Update metadata
        merged["last_ml_update"] = datetime.now().isoformat()
        if "ml_analysis" in ml_updates:
            if "ml_analysis" not in merged:
                merged["ml_analysis"] = []
            merged["ml_analysis"].extend(ml_updates["ml_analysis"])

        return merged

    @staticmethod
    def export_enhanced_geojson(
        plantation_data: Dict, ml_result: InferenceResult, output_path: str
    ):
        """
        Export enhanced GeoJSON with ML attributes.

        Args:
            plantation_data: Updated plantation data
            ml_result: ML inference result
            output_path: Path to save GeoJSON
        """
        features = []

        # Add trees with ML attributes
        for tree in plantation_data.get("trees", []):
            properties = {
                "id": tree.get("id"),
                "species": tree.get("species"),
                "crown_diameter": tree.get("crown_diameter_m", 0),
                "ml_detected": tree.get("ml_detected", False),
                "ml_confidence": tree.get("ml_confidence", 0),
                "ml_crown_diameter": tree.get("ml_crown_diameter", 0),
                "position_error": tree.get("position_error", None),
                "last_detected": tree.get("last_detected", None),
            }

            # Use ML coordinates if available and more accurate
            if (
                tree.get("ml_coordinates")
                and tree.get("position_error", float("inf")) > 1.0
            ):
                coordinates = tree["ml_coordinates"]
            else:
                coordinates = tree.get("coordinates", [0, 0])

            feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": coordinates},
                "properties": properties,
            }
            features.append(feature)

        # Create GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "ml_model": "cabruca_segmentation",
                "total_trees": len(features),
            },
        }

        # Save to file
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)

        print(f"Enhanced GeoJSON saved to {output_path}")


def integrate_with_theobroma(
    model_path: str, plantation_data_path: str, image_path: str, output_dir: str
):
    """
    Main integration function.

    Args:
        model_path: Path to ML model
        plantation_data_path: Path to plantation data JSON
        image_path: Path to plantation image
        output_dir: Directory for outputs
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize integration
    integration = TheobromaIntegration(model_path, plantation_data_path)

    # Compare ML with plantation data
    print("Comparing ML detection with plantation data...")
    comparison = integration.compare_with_ml_detection(image_path)

    # Save comparison report
    comparison_path = output_path / "comparison_report.json"
    with open(comparison_path, "w") as f:
        json.dump(
            {k: v for k, v in comparison.items() if k != "ml_result"}, f, indent=2
        )

    print(f"Comparison Statistics:")
    stats = comparison["statistics"]
    print(f"  - Matched trees: {stats['matched_trees']}")
    print(f"  - Detection F1: {stats['detection_f1']:.2%}")
    print(f"  - Position RMSE: {stats.get('position_rmse', 0):.2f}m")
    print(f"  - Species accuracy: {stats.get('species_accuracy', 0):.2%}")

    # Update plantation data
    print("\nUpdating plantation data with ML attributes...")
    updated_data = integration.update_plantation_data(comparison)

    # Save updated data
    updated_path = output_path / "plantation_data_updated.json"
    with open(updated_path, "w") as f:
        json.dump(updated_data, f, indent=2)

    # Generate health metrics
    print("\nGenerating health metrics...")
    health_metrics = integration.generate_health_metrics(comparison["ml_result"])

    health_path = output_path / "health_metrics.json"
    with open(health_path, "w") as f:
        json.dump(health_metrics, f, indent=2)

    print(
        f"Plantation Health: {health_metrics['plantation_health'].get('status', 'N/A').upper()}"
    )
    print(
        f"Overall Score: {health_metrics['plantation_health'].get('overall_score', 0):.2%}"
    )

    # Create visualization
    print("\nCreating enhanced visualization...")
    viz_result = integration.create_enhanced_visualization(
        image_path, comparison["ml_result"], updated_data
    )

    print(f"Visualization saved to: {viz_result['visualization_path']}")

    # Export enhanced GeoJSON
    geojson_path = output_path / "enhanced_plantation.geojson"
    PlantationDataUpdater.export_enhanced_geojson(
        updated_data, comparison["ml_result"], str(geojson_path)
    )

    # Print recommendations
    if health_metrics["recommendations"]:
        print("\nRecommendations:")
        for rec in health_metrics["recommendations"]:
            print(f"  • {rec}")

    print(f"\n✅ Integration complete! Results saved to {output_path}")

    return {
        "comparison": comparison,
        "updated_data": updated_data,
        "health_metrics": health_metrics,
        "visualization": viz_result,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Integrate ML with Theobroma Digital")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to ML model checkpoint"
    )
    parser.add_argument(
        "--plantation-data",
        type=str,
        required=True,
        help="Path to plantation-data.json",
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to plantation image"
    )
    parser.add_argument(
        "--output", type=str, default="integration_results", help="Output directory"
    )

    args = parser.parse_args()

    integrate_with_theobroma(args.model, args.plantation_data, args.image, args.output)

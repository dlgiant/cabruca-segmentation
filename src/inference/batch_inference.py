"""
Batch inference pipeline for Cabruca segmentation.
Processes multiple images efficiently with comprehensive analysis.
"""

import json
import os
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import agentops
import cv2
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.windows import Window
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from tqdm import tqdm

warnings.filterwarnings("ignore")

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.agroforestry_metrics import AgroforestryMetrics
from models.cabruca_segmentation_model import create_cabruca_model


@dataclass
class TreeInstance:
    """Data class for individual tree instances."""

    id: int
    species: str
    confidence: float
    bbox: List[float]
    centroid: Tuple[float, float]
    crown_diameter: float
    crown_area: float
    geometry: Optional[Polygon] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        if self.geometry:
            d["geometry"] = self.geometry.wkt
        return d


@dataclass
class InferenceResult:
    """Container for inference results."""

    image_path: str
    timestamp: str
    trees: List[TreeInstance]
    semantic_map: np.ndarray
    crown_map: np.ndarray
    canopy_density: float
    metrics: Dict
    georef_info: Optional[Dict] = None

    def summary(self) -> Dict:
        """Generate summary statistics."""
        return {
            "total_trees": len(self.trees),
            "cacao_trees": sum(1 for t in self.trees if t.species == "cacao"),
            "shade_trees": sum(1 for t in self.trees if t.species == "shade"),
            "avg_crown_diameter": (
                np.mean([t.crown_diameter for t in self.trees]) if self.trees else 0
            ),
            "total_crown_area": sum(t.crown_area for t in self.trees),
            "canopy_density": self.canopy_density,
            "image_path": self.image_path,
            "timestamp": self.timestamp,
        }


class BatchInferenceEngine:
    """
    Batch inference engine for processing multiple images.
    """

    # Class definitions
    INSTANCE_CLASSES = {1: "cacao", 2: "shade"}
    SEMANTIC_CLASSES = {
        0: "background",
        1: "cacao_tree",
        2: "shade_tree",
        3: "understory",
        4: "bare_soil",
        5: "shadows",
    }

    # Color schemes
    SEMANTIC_COLORS = np.array(
        [
            [255, 255, 255],  # Background - White
            [34, 139, 34],  # Cacao - Forest Green
            [0, 100, 0],  # Shade - Dark Green
            [144, 238, 144],  # Understory - Light Green
            [139, 69, 19],  # Bare soil - Brown
            [105, 105, 105],  # Shadows - Gray
        ]
    )

    def __init__(
        self, model_path: str, config: Optional[Dict] = None, device: str = "auto"
    ):
        """
        Initialize batch inference engine.

        Args:
            model_path: Path to trained model checkpoint
            config: Model configuration
            device: Device for inference ('auto', 'cpu', 'cuda', 'mps')
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path, config)
        self.config = config or {}

        # Inference settings
        self.batch_size = self.config.get("batch_size", 1)
        self.tile_size = self.config.get("tile_size", 512)
        self.overlap = self.config.get("overlap", 64)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)

        # Metrics evaluator
        self.evaluator = AgroforestryMetrics()

    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _load_model(self, model_path: str, config: Optional[Dict]) -> torch.nn.Module:
        """Load trained model."""
        if config:
            model = create_cabruca_model(config.get("model", {}))
        else:
            from models.cabruca_segmentation_model import CabrucaSegmentationModel

            model = CabrucaSegmentationModel()

        # Load checkpoint
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        return model

    @agentops.operation
    def process_batch(
        self, image_paths: List[str], output_dir: str = None
    ) -> List[InferenceResult]:
        """
        Process a batch of images.

        Args:
            image_paths: List of image file paths
            output_dir: Directory to save results

        Returns:
            List of InferenceResult objects
        """
        results = []

        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Process each image
        for image_path in tqdm(image_paths, desc="Processing images"):
            result = self.process_single(image_path)
            results.append(result)

            # Save results if output directory specified
            if output_dir:
                self._save_results(result, output_path)

        # Generate batch report
        if output_dir:
            self._generate_batch_report(results, output_path)

        return results

    @agentops.operation
    def process_single(self, image_path: str) -> InferenceResult:
        """
        Process a single image.

        Args:
            image_path: Path to image file

        Returns:
            InferenceResult object
        """
        # Load image
        image, georef_info = self._load_image(image_path)

        # Check if tiling is needed
        h, w = image.shape[:2]
        if h > self.tile_size * 1.5 or w > self.tile_size * 1.5:
            outputs = self._process_tiled(image)
        else:
            outputs = self._process_whole(image)

        # Post-process outputs
        trees = self._extract_trees(outputs)

        # Create result object
        result = InferenceResult(
            image_path=image_path,
            timestamp=datetime.now().isoformat(),
            trees=trees,
            semantic_map=outputs.get("semantic_map", np.zeros((h, w))),
            crown_map=outputs.get("crown_map", np.zeros((h, w))),
            canopy_density=outputs.get("canopy_density", 0.0),
            metrics=self._calculate_metrics(outputs),
            georef_info=georef_info,
        )

        return result

    def _load_image(self, image_path: str) -> Tuple[np.ndarray, Optional[Dict]]:
        """Load image and georeferencing information."""
        georef_info = None

        if image_path.endswith((".tif", ".tiff")):
            # Load GeoTIFF
            with rasterio.open(image_path) as src:
                image = src.read()
                if image.shape[0] > 3:
                    image = image[:3]
                image = np.transpose(image, (1, 2, 0))

                # Store georeferencing info
                georef_info = {
                    "transform": src.transform,
                    "crs": src.crs,
                    "bounds": src.bounds,
                }
        else:
            # Load regular image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image / image.max() * 255).astype(np.uint8)

        return image, georef_info

    def _process_whole(self, image: np.ndarray) -> Dict:
        """Process entire image at once."""
        # Preprocess
        input_tensor = self._preprocess_image(image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)

        # Post-process
        return self._postprocess_outputs(outputs, image.shape[:2])

    def _process_tiled(self, image: np.ndarray) -> Dict:
        """Process large image using tiling."""
        h, w = image.shape[:2]

        # Initialize output arrays
        semantic_map = np.zeros((h, w), dtype=np.uint8)
        crown_map = np.zeros((h, w), dtype=np.float32)
        instance_list = []

        # Generate tiles
        tiles = []
        for y in range(0, h - self.overlap, self.tile_size - self.overlap):
            for x in range(0, w - self.overlap, self.tile_size - self.overlap):
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                tiles.append((x, y, x_end, y_end))

        # Process tiles
        for x, y, x_end, y_end in tqdm(tiles, desc="Processing tiles", leave=False):
            tile = image[y:y_end, x:x_end]

            # Process tile
            tile_outputs = self._process_whole(tile)

            # Merge semantic map
            if "semantic_map" in tile_outputs:
                semantic_map[y:y_end, x:x_end] = tile_outputs["semantic_map"]

            # Merge crown map
            if "crown_map" in tile_outputs:
                crown_map[y:y_end, x:x_end] = tile_outputs["crown_map"]

            # Adjust instance coordinates
            if "instances" in tile_outputs:
                for inst in tile_outputs["instances"]:
                    inst["bbox"][0] += x
                    inst["bbox"][1] += y
                    inst["bbox"][2] += x
                    inst["bbox"][3] += y
                    instance_list.append(inst)

        # Remove duplicate instances from overlapping regions
        instance_list = self._remove_duplicate_instances(instance_list)

        return {
            "semantic_map": semantic_map,
            "crown_map": crown_map,
            "instances": instance_list,
            "canopy_density": np.sum(semantic_map <= 2) / semantic_map.size,
        }

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        return image_tensor

    def _postprocess_outputs(self, outputs: Dict, image_shape: Tuple[int, int]) -> Dict:
        """Post-process model outputs."""
        results = {}

        # Semantic segmentation
        if "semantic" in outputs:
            semantic_pred = torch.argmax(outputs["semantic"], dim=1).squeeze()
            results["semantic_map"] = semantic_pred.cpu().numpy()

        # Instance segmentation
        if "instances" in outputs:
            instances = []
            instance_outputs = outputs["instances"]

            if isinstance(instance_outputs, list) and len(instance_outputs) > 0:
                inst = instance_outputs[0]

                boxes = inst.get("boxes", torch.tensor([])).cpu().numpy()
                labels = inst.get("labels", torch.tensor([])).cpu().numpy()
                scores = inst.get("scores", torch.tensor([])).cpu().numpy()
                masks = inst.get("masks", torch.tensor([])).cpu().numpy()

                for i in range(len(boxes)):
                    if scores[i] >= self.confidence_threshold:
                        instances.append(
                            {
                                "bbox": boxes[i],
                                "label": labels[i],
                                "score": scores[i],
                                "mask": masks[i] if i < len(masks) else None,
                            }
                        )

            results["instances"] = instances

        # Crown map
        if "crown_diameters" in outputs:
            crown_pred = outputs["crown_diameters"].squeeze()
            results["crown_map"] = crown_pred.cpu().numpy()

        # Canopy density
        if "canopy_density" in outputs:
            results["canopy_density"] = outputs["canopy_density"].cpu().item()

        return results

    def _extract_trees(self, outputs: Dict) -> List[TreeInstance]:
        """Extract tree instances from model outputs."""
        trees = []

        if "instances" not in outputs:
            return trees

        for i, inst in enumerate(outputs["instances"]):
            # Get species
            label = inst["label"]
            species = self.INSTANCE_CLASSES.get(label, "unknown")

            # Calculate centroid
            bbox = inst["bbox"]
            centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            # Estimate crown diameter
            if inst.get("mask") is not None:
                mask = inst["mask"]
                crown_area = np.sum(mask > 0.5)
                crown_diameter = 2 * np.sqrt(crown_area / np.pi)
            else:
                # Estimate from bounding box
                crown_diameter = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                crown_area = np.pi * (crown_diameter / 2) ** 2

            # Create polygon from mask if available
            geometry = None
            if inst.get("mask") is not None:
                contours, _ = cv2.findContours(
                    (inst["mask"] > 0.5).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if len(largest_contour) >= 3:
                        points = largest_contour.squeeze()
                        if len(points.shape) == 2 and points.shape[0] >= 3:
                            geometry = Polygon(points)

            # Create tree instance
            tree = TreeInstance(
                id=i,
                species=species,
                confidence=inst["score"],
                bbox=bbox.tolist(),
                centroid=centroid,
                crown_diameter=crown_diameter,
                crown_area=crown_area,
                geometry=geometry,
            )

            trees.append(tree)

        return trees

    def _remove_duplicate_instances(
        self, instances: List[Dict], iou_threshold: float = 0.5
    ) -> List[Dict]:
        """Remove duplicate instances from overlapping tiles."""
        if len(instances) <= 1:
            return instances

        # Sort by confidence
        instances = sorted(instances, key=lambda x: x.get("score", 0), reverse=True)

        keep = []
        for inst in instances:
            is_duplicate = False

            for kept in keep:
                iou = self._calculate_bbox_iou(inst["bbox"], kept["bbox"])
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep.append(inst)

        return keep

    def _calculate_bbox_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes."""
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

    def _calculate_metrics(self, outputs: Dict) -> Dict:
        """Calculate inference metrics."""
        metrics = {}

        # Tree counts
        if "instances" in outputs:
            labels = [inst["label"] for inst in outputs["instances"]]
            metrics["total_trees"] = len(labels)
            metrics["cacao_trees"] = sum(1 for l in labels if l == 1)
            metrics["shade_trees"] = sum(1 for l in labels if l == 2)

        # Semantic distribution
        if "semantic_map" in outputs:
            semantic = outputs["semantic_map"]
            total_pixels = semantic.size

            for class_id, class_name in self.SEMANTIC_CLASSES.items():
                class_pixels = np.sum(semantic == class_id)
                metrics[f"{class_name}_coverage"] = class_pixels / total_pixels

        # Crown statistics
        if "crown_map" in outputs:
            crown = outputs["crown_map"]
            crown_values = crown[crown > 0]
            if len(crown_values) > 0:
                metrics["avg_crown_diameter"] = float(np.mean(crown_values))
                metrics["std_crown_diameter"] = float(np.std(crown_values))

        # Canopy density
        metrics["canopy_density"] = outputs.get("canopy_density", 0.0)

        return metrics

    def _save_results(self, result: InferenceResult, output_dir: Path):
        """Save inference results."""
        # Create subdirectory for this image
        image_name = Path(result.image_path).stem
        image_dir = output_dir / image_name
        image_dir.mkdir(exist_ok=True)

        # Save JSON summary
        summary_path = image_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.summary(), f, indent=2)

        # Save tree inventory
        if result.trees:
            inventory_path = image_dir / "tree_inventory.json"
            inventory = [tree.to_dict() for tree in result.trees]
            with open(inventory_path, "w") as f:
                json.dump(inventory, f, indent=2)

        # Save semantic map
        semantic_path = image_dir / "semantic_map.npy"
        np.save(semantic_path, result.semantic_map)

        # Save crown map
        crown_path = image_dir / "crown_map.npy"
        np.save(crown_path, result.crown_map)

    def _generate_batch_report(self, results: List[InferenceResult], output_dir: Path):
        """Generate comprehensive batch processing report."""
        # Aggregate statistics
        total_trees = sum(len(r.trees) for r in results)
        total_cacao = sum(
            sum(1 for t in r.trees if t.species == "cacao") for r in results
        )
        total_shade = sum(
            sum(1 for t in r.trees if t.species == "shade") for r in results
        )

        avg_canopy_density = np.mean([r.canopy_density for r in results])

        # Create report
        report = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_images": len(results),
            "total_trees_detected": total_trees,
            "total_cacao_trees": total_cacao,
            "total_shade_trees": total_shade,
            "average_canopy_density": float(avg_canopy_density),
            "image_summaries": [r.summary() for r in results],
        }

        # Save report
        report_path = output_dir / "batch_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Batch report saved to {report_path}")
        print(f"Processed {len(results)} images")
        print(
            f"Total trees detected: {total_trees} ({total_cacao} cacao, {total_shade} shade)"
        )
        print(f"Average canopy density: {avg_canopy_density:.2%}")


class VisualizationTools:
    """
    Advanced visualization tools for segmentation results.
    """

    @staticmethod
    def create_overlay(
        image: np.ndarray, result: InferenceResult, alpha: float = 0.4
    ) -> np.ndarray:
        """
        Create overlay of predictions on original image.

        Args:
            image: Original image
            result: Inference result
            alpha: Transparency for overlay

        Returns:
            Overlaid image
        """
        overlay = image.copy()

        # Apply semantic segmentation with transparency
        semantic_colored = BatchInferenceEngine.SEMANTIC_COLORS[result.semantic_map]
        mask = result.semantic_map > 0
        overlay[mask] = (
            overlay[mask] * (1 - alpha) + semantic_colored[mask] * alpha
        ).astype(np.uint8)

        # Draw tree instances
        for tree in result.trees:
            # Draw bounding box
            x1, y1, x2, y2 = [int(x) for x in tree.bbox]
            color = (0, 255, 0) if tree.species == "cacao" else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Add label
            label = f"{tree.species} ({tree.confidence:.2f})"
            cv2.putText(
                overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            # Draw centroid
            cx, cy = [int(x) for x in tree.centroid]
            cv2.circle(overlay, (cx, cy), 3, color, -1)

        return overlay

    @staticmethod
    def create_heatmap(
        result: InferenceResult, heatmap_type: str = "density"
    ) -> np.ndarray:
        """
        Create heatmap visualization.

        Args:
            result: Inference result
            heatmap_type: Type of heatmap ('density', 'crown', 'species')

        Returns:
            Heatmap image
        """
        h, w = result.semantic_map.shape

        if heatmap_type == "density":
            # Tree density heatmap
            heatmap = np.zeros((h, w), dtype=np.float32)

            # Add gaussian for each tree
            for tree in result.trees:
                cx, cy = [int(x) for x in tree.centroid]
                radius = int(tree.crown_diameter / 2)

                # Create gaussian kernel
                y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
                kernel = np.exp(-(x**2 + y**2) / (2 * (radius / 2) ** 2))

                # Add to heatmap
                y1, y2 = max(0, cy - radius), min(h, cy + radius + 1)
                x1, x2 = max(0, cx - radius), min(w, cx + radius + 1)

                kernel_y1 = radius - (cy - y1)
                kernel_y2 = radius + (y2 - cy)
                kernel_x1 = radius - (cx - x1)
                kernel_x2 = radius + (cx - x2)

                heatmap[y1:y2, x1:x2] += kernel[
                    kernel_y1:kernel_y2, kernel_x1:kernel_x2
                ]

            # Normalize
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

        elif heatmap_type == "crown":
            # Crown diameter heatmap
            heatmap = result.crown_map
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

        elif heatmap_type == "species":
            # Species distribution heatmap
            heatmap = np.zeros((h, w, 3), dtype=np.float32)

            # Cacao in green channel, shade in red channel
            for tree in result.trees:
                cx, cy = [int(x) for x in tree.centroid]
                radius = int(tree.crown_diameter / 2)

                channel = 1 if tree.species == "cacao" else 0

                cv2.circle(heatmap[:, :, channel], (cx, cy), radius, 1.0, -1)

            # Normalize each channel
            for i in range(3):
                if heatmap[:, :, i].max() > 0:
                    heatmap[:, :, i] = heatmap[:, :, i] / heatmap[:, :, i].max()

        # Apply colormap
        if len(heatmap.shape) == 2:
            heatmap = plt.cm.hot(heatmap)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)
        else:
            heatmap = (heatmap * 255).astype(np.uint8)

        return heatmap

    @staticmethod
    def create_comparison_figure(
        image: np.ndarray, result: InferenceResult, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive comparison figure.

        Args:
            image: Original image
            result: Inference result
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # Overlay
        overlay = VisualizationTools.create_overlay(image, result)
        axes[0, 1].imshow(overlay)
        axes[0, 1].set_title("Segmentation Overlay")
        axes[0, 1].axis("off")

        # Semantic segmentation
        semantic_colored = BatchInferenceEngine.SEMANTIC_COLORS[result.semantic_map]
        axes[0, 2].imshow(semantic_colored.astype(np.uint8))
        axes[0, 2].set_title("Semantic Segmentation")
        axes[0, 2].axis("off")

        # Tree density heatmap
        density_heatmap = VisualizationTools.create_heatmap(result, "density")
        axes[1, 0].imshow(density_heatmap)
        axes[1, 0].set_title("Tree Density Heatmap")
        axes[1, 0].axis("off")

        # Crown diameter heatmap
        crown_heatmap = VisualizationTools.create_heatmap(result, "crown")
        axes[1, 1].imshow(crown_heatmap)
        axes[1, 1].set_title("Crown Diameter Heatmap")
        axes[1, 1].axis("off")

        # Statistics panel
        axes[1, 2].axis("off")
        stats_text = f"""
        Tree Inventory:
        - Total Trees: {len(result.trees)}
        - Cacao Trees: {sum(1 for t in result.trees if t.species == 'cacao')}
        - Shade Trees: {sum(1 for t in result.trees if t.species == 'shade')}
        
        Canopy Metrics:
        - Canopy Density: {result.canopy_density:.2%}
        - Avg Crown Diameter: {np.mean([t.crown_diameter for t in result.trees]):.2f}m
        
        Land Cover:
        - Tree Cover: {result.metrics.get('cacao_tree_coverage', 0) + result.metrics.get('shade_tree_coverage', 0):.2%}
        - Understory: {result.metrics.get('understory_coverage', 0):.2%}
        - Bare Soil: {result.metrics.get('bare_soil_coverage', 0):.2%}
        """
        axes[1, 2].text(
            0.1,
            0.9,
            stats_text,
            transform=axes[1, 2].transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.suptitle(
            f"Cabruca Segmentation Analysis - {Path(result.image_path).name}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig


class ReportGenerator:
    """
    Generate comprehensive reports from inference results.
    """

    @staticmethod
    def generate_tree_inventory(
        results: List[InferenceResult], output_path: str
    ) -> pd.DataFrame:
        """
        Generate tree inventory report.

        Args:
            results: List of inference results
            output_path: Path to save report

        Returns:
            DataFrame with tree inventory
        """
        inventory_data = []

        for result in results:
            image_name = Path(result.image_path).name

            for tree in result.trees:
                inventory_data.append(
                    {
                        "image": image_name,
                        "tree_id": tree.id,
                        "species": tree.species,
                        "confidence": tree.confidence,
                        "centroid_x": tree.centroid[0],
                        "centroid_y": tree.centroid[1],
                        "crown_diameter_m": tree.crown_diameter,
                        "crown_area_m2": tree.crown_area,
                        "timestamp": result.timestamp,
                    }
                )

        df = pd.DataFrame(inventory_data)

        # Add summary statistics
        summary = (
            df.groupby("species")
            .agg(
                {
                    "tree_id": "count",
                    "crown_diameter_m": ["mean", "std", "min", "max"],
                    "crown_area_m2": ["sum", "mean"],
                    "confidence": "mean",
                }
            )
            .round(2)
        )

        # Save to Excel with multiple sheets
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Tree Inventory", index=False)
            summary.to_excel(writer, sheet_name="Summary Statistics")

        print(f"Tree inventory saved to {output_path}")

        return df

    @staticmethod
    def export_to_geojson(
        result: InferenceResult, output_path: str, crs: str = "EPSG:4326"
    ):
        """
        Export results to GeoJSON format.

        Args:
            result: Inference result
            output_path: Path to save GeoJSON
            crs: Coordinate reference system
        """
        features = []

        for tree in result.trees:
            # Create feature
            if tree.geometry:
                geometry = tree.geometry
            else:
                # Create circle from centroid and radius
                geometry = Point(tree.centroid).buffer(tree.crown_diameter / 2)

            feature = {
                "type": "Feature",
                "geometry": geometry.__geo_interface__,
                "properties": {
                    "id": tree.id,
                    "species": tree.species,
                    "confidence": tree.confidence,
                    "crown_diameter": tree.crown_diameter,
                    "crown_area": tree.crown_area,
                },
            }
            features.append(feature)

        # Create GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "crs": {"init": crs},
            "features": features,
        }

        # Apply georeferencing if available
        if result.georef_info:
            # Transform coordinates using georef info
            # This is simplified - actual implementation would use rasterio transforms
            pass

        # Save to file
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)

        print(f"GeoJSON exported to {output_path}")

    @staticmethod
    def generate_analysis_report(results: List[InferenceResult], output_dir: str):
        """
        Generate comprehensive analysis report with visualizations.

        Args:
            results: List of inference results
            output_dir: Directory to save report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate tree inventory
        inventory_path = output_path / "tree_inventory.xlsx"
        inventory_df = ReportGenerator.generate_tree_inventory(
            results, str(inventory_path)
        )

        # Generate summary statistics
        summary = {
            "total_images": len(results),
            "total_trees": sum(len(r.trees) for r in results),
            "species_distribution": inventory_df["species"].value_counts().to_dict(),
            "avg_crown_diameter": inventory_df["crown_diameter_m"].mean(),
            "total_crown_area": inventory_df["crown_area_m2"].sum(),
            "avg_canopy_density": np.mean([r.canopy_density for r in results]),
        }

        # Save summary
        summary_path = output_path / "analysis_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Generate visualizations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Species distribution
        species_counts = inventory_df["species"].value_counts()
        axes[0, 0].pie(
            species_counts.values,
            labels=species_counts.index,
            autopct="%1.1f%%",
            colors=["green", "brown"],
        )
        axes[0, 0].set_title("Species Distribution")

        # Crown diameter distribution
        axes[0, 1].hist(inventory_df["crown_diameter_m"], bins=20, edgecolor="black")
        axes[0, 1].set_xlabel("Crown Diameter (m)")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Crown Diameter Distribution")

        # Confidence scores
        axes[1, 0].boxplot(
            [
                inventory_df[inventory_df["species"] == s]["confidence"].values
                for s in inventory_df["species"].unique()
            ],
            labels=inventory_df["species"].unique(),
        )
        axes[1, 0].set_ylabel("Confidence Score")
        axes[1, 0].set_title("Detection Confidence by Species")

        # Canopy density by image
        image_density = pd.DataFrame(
            [
                {"image": Path(r.image_path).name, "density": r.canopy_density}
                for r in results
            ]
        )
        axes[1, 1].bar(range(len(image_density)), image_density["density"])
        axes[1, 1].set_xlabel("Image Index")
        axes[1, 1].set_ylabel("Canopy Density")
        axes[1, 1].set_title("Canopy Density by Image")
        axes[1, 1].set_ylim([0, 1])

        plt.suptitle("Cabruca Analysis Report", fontsize=16, fontweight="bold")
        plt.tight_layout()

        # Save figure
        fig_path = output_path / "analysis_report.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")

        print(f"Analysis report saved to {output_dir}")


def main():
    """Main inference pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch Inference for Cabruca Segmentation"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="Image file paths or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Model configuration file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for inference",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for processing"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )
    parser.add_argument(
        "--export-geojson", action="store_true", help="Export results to GeoJSON"
    )

    args = parser.parse_args()

    # Load configuration if provided
    config = None
    if args.config:
        with open(args.config, "r") as f:
            import yaml

            config = yaml.safe_load(f)

    # Initialize inference engine
    engine = BatchInferenceEngine(
        model_path=args.model, config=config, device=args.device
    )

    # Collect image paths
    image_paths = []
    for path_str in args.images:
        path = Path(path_str)
        if path.is_dir():
            # Add all images in directory
            for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
                image_paths.extend(path.glob(ext))
        elif path.is_file():
            image_paths.append(path)

    image_paths = [str(p) for p in image_paths]

    print(f"Processing {len(image_paths)} images...")

    # Process images
    results = engine.process_batch(image_paths, args.output)

    # Generate visualizations if requested
    if args.visualize:
        vis_dir = Path(args.output) / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            image_name = Path(result.image_path).stem

            # Load original image
            image, _ = engine._load_image(result.image_path)

            # Create comparison figure
            fig_path = vis_dir / f"{image_name}_analysis.png"
            VisualizationTools.create_comparison_figure(image, result, str(fig_path))

            plt.close("all")

    # Export to GeoJSON if requested
    if args.export_geojson:
        geojson_dir = Path(args.output) / "geojson"
        geojson_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            image_name = Path(result.image_path).stem
            geojson_path = geojson_dir / f"{image_name}.geojson"
            ReportGenerator.export_to_geojson(result, str(geojson_path))

    # Generate final report
    ReportGenerator.generate_analysis_report(results, args.output)

    print(f"\nInference complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()

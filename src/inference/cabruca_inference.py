"""
Inference and visualization tools for Cabruca segmentation model.
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
import geopandas as gpd
from shapely.geometry import Polygon, Point
import os
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cabruca_segmentation_model import CabrucaSegmentationModel, create_cabruca_model
from data_processing.cabruca_dataset import CabrucaDataset


class CabrucaInference:
    """
    Inference pipeline for Cabruca segmentation model.
    """
    
    # Color mappings for visualization
    INSTANCE_COLORS = {
        1: (34, 139, 34),    # Cacao tree - Forest Green
        2: (0, 100, 0)       # Shade tree - Dark Green
    }
    
    SEMANTIC_COLORS = {
        0: (0, 0, 0),        # Background - Black
        1: (34, 139, 34),    # Cacao tree - Forest Green
        2: (0, 100, 0),      # Shade tree - Dark Green
        3: (144, 238, 144),  # Understory - Light Green
        4: (139, 69, 19),    # Bare soil - Brown
        5: (105, 105, 105)   # Shadows - Gray
    }
    
    def __init__(self, model_path: str, config: Optional[Dict] = None, device: str = None):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            config: Model configuration
            device: Device to use for inference
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path, config)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Inference settings
        self.tile_size = config.get('tile_size', 512) if config else 512
        self.overlap = config.get('overlap', 64) if config else 64
        self.batch_size = config.get('batch_size', 1) if config else 1
    
    def _load_model(self, model_path: str, config: Optional[Dict]) -> CabrucaSegmentationModel:
        """Load trained model."""
        if config:
            model = create_cabruca_model(config)
        else:
            model = CabrucaSegmentationModel()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def predict_image(self, image_path: str, use_tiles: bool = True) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            use_tiles: Whether to process large images in tiles
            
        Returns:
            Dictionary containing all predictions
        """
        if use_tiles and self._is_large_image(image_path):
            return self._predict_tiled(image_path)
        else:
            return self._predict_single(image_path)
    
    def _is_large_image(self, image_path: str) -> bool:
        """Check if image is too large for single processing."""
        with rasterio.open(image_path) as src:
            return src.height > 2048 or src.width > 2048
    
    def _predict_single(self, image_path: str) -> Dict:
        """Predict on a single image without tiling."""
        # Load and preprocess image
        image = self._load_image(image_path)
        image_tensor = self._preprocess_image(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Post-process outputs
        results = self._postprocess_outputs(outputs, image.shape[:2])
        
        # Calculate metrics
        metrics = self.model.calculate_tree_metrics(outputs)
        
        results['metrics'] = metrics
        results['image_path'] = image_path
        
        return results
    
    def _predict_tiled(self, image_path: str) -> Dict:
        """Predict on large image using tiling."""
        tiles_results = []
        
        with rasterio.open(image_path) as src:
            height, width = src.shape
            meta = src.meta.copy()
            
            # Create tiles
            tiles = []
            for y in range(0, height - self.overlap, self.tile_size - self.overlap):
                for x in range(0, width - self.overlap, self.tile_size - self.overlap):
                    window = Window(
                        col_off=x,
                        row_off=y,
                        width=min(self.tile_size, width - x),
                        height=min(self.tile_size, height - y)
                    )
                    tiles.append((x, y, window))
            
            # Process tiles in batches
            for i in tqdm(range(0, len(tiles), self.batch_size), desc="Processing tiles"):
                batch_tiles = tiles[i:i+self.batch_size]
                batch_images = []
                
                for x, y, window in batch_tiles:
                    tile_data = src.read(window=window)
                    tile_data = np.transpose(tile_data, (1, 2, 0))
                    
                    if tile_data.shape[2] > 3:
                        tile_data = tile_data[:, :, :3]
                    
                    tile_tensor = self._preprocess_image(tile_data)
                    batch_images.append(tile_tensor)
                
                # Run batch inference
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                with torch.no_grad():
                    batch_outputs = self.model(batch_tensor)
                
                # Store results
                for j, (x, y, window) in enumerate(batch_tiles):
                    tile_outputs = self._extract_tile_outputs(batch_outputs, j)
                    tiles_results.append({
                        'x': x,
                        'y': y,
                        'window': window,
                        'outputs': tile_outputs
                    })
        
        # Merge tile results
        merged_results = self._merge_tile_results(tiles_results, (height, width))
        merged_results['image_path'] = image_path
        
        return merged_results
    
    def _extract_tile_outputs(self, batch_outputs: Dict, idx: int) -> Dict:
        """Extract outputs for a single tile from batch."""
        tile_outputs = {}
        
        if 'semantic' in batch_outputs:
            tile_outputs['semantic'] = batch_outputs['semantic'][idx]
        
        if 'crown_diameters' in batch_outputs:
            tile_outputs['crown_diameters'] = batch_outputs['crown_diameters'][idx]
        
        if 'canopy_density' in batch_outputs:
            tile_outputs['canopy_density'] = batch_outputs['canopy_density'][idx]
        
        # Instance outputs need special handling
        if 'instances' in batch_outputs:
            if isinstance(batch_outputs['instances'], list):
                tile_outputs['instances'] = batch_outputs['instances'][idx]
            else:
                tile_outputs['instances'] = {
                    k: v[idx] if len(v) > idx else None 
                    for k, v in batch_outputs['instances'].items()
                }
        
        return tile_outputs
    
    def _merge_tile_results(self, tiles_results: List[Dict], 
                           image_shape: Tuple[int, int]) -> Dict:
        """Merge results from multiple tiles."""
        height, width = image_shape
        
        # Initialize merged arrays
        semantic_map = np.zeros((height, width), dtype=np.uint8)
        crown_map = np.zeros((height, width), dtype=np.float32)
        instance_masks = []
        instance_boxes = []
        instance_labels = []
        
        # Merge tiles
        for tile_result in tiles_results:
            x, y = tile_result['x'], tile_result['y']
            window = tile_result['window']
            outputs = tile_result['outputs']
            
            # Merge semantic segmentation
            if 'semantic' in outputs:
                semantic_pred = torch.argmax(outputs['semantic'], dim=0).cpu().numpy()
                semantic_map[y:y+window.height, x:x+window.width] = semantic_pred
            
            # Merge crown diameters
            if 'crown_diameters' in outputs:
                crown_pred = outputs['crown_diameters'].squeeze().cpu().numpy()
                crown_map[y:y+window.height, x:x+window.width] = crown_pred
            
            # Merge instances (with offset)
            if 'instances' in outputs and outputs['instances']:
                if 'masks' in outputs['instances']:
                    for mask, box, label in zip(
                        outputs['instances'].get('masks', []),
                        outputs['instances'].get('boxes', []),
                        outputs['instances'].get('labels', [])
                    ):
                        # Adjust coordinates
                        adjusted_box = box.clone()
                        adjusted_box[[0, 2]] += x
                        adjusted_box[[1, 3]] += y
                        
                        # Create full-size mask
                        full_mask = np.zeros((height, width), dtype=np.uint8)
                        mask_np = mask.cpu().numpy()
                        full_mask[y:y+window.height, x:x+window.width] = mask_np
                        
                        instance_masks.append(full_mask)
                        instance_boxes.append(adjusted_box.cpu().numpy())
                        instance_labels.append(label.cpu().item())
        
        # Remove duplicate instances (from overlapping tiles)
        if instance_masks:
            unique_instances = self._remove_duplicate_instances(
                instance_masks, instance_boxes, instance_labels
            )
        else:
            unique_instances = {
                'masks': [],
                'boxes': [],
                'labels': []
            }
        
        return {
            'semantic_map': semantic_map,
            'crown_map': crown_map,
            'instances': unique_instances,
            'metrics': self._calculate_merged_metrics(
                semantic_map, crown_map, unique_instances
            )
        }
    
    def _remove_duplicate_instances(self, masks, boxes, labels, iou_threshold=0.5):
        """Remove duplicate instances from overlapping tiles."""
        if not masks:
            return {'masks': [], 'boxes': [], 'labels': []}
        
        # Calculate IoU between all pairs
        keep = []
        n = len(masks)
        
        for i in range(n):
            is_duplicate = False
            for j in keep:
                # Calculate IoU
                intersection = np.logical_and(masks[i], masks[j]).sum()
                union = np.logical_or(masks[i], masks[j]).sum()
                iou = intersection / (union + 1e-6)
                
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep.append(i)
        
        return {
            'masks': [masks[i] for i in keep],
            'boxes': [boxes[i] for i in keep],
            'labels': [labels[i] for i in keep]
        }
    
    def _calculate_merged_metrics(self, semantic_map, crown_map, instances):
        """Calculate metrics from merged results."""
        metrics = {}
        
        # Tree counts
        labels = instances.get('labels', [])
        metrics['cacao_count'] = sum(1 for l in labels if l == 1)
        metrics['shade_tree_count'] = sum(1 for l in labels if l == 2)
        metrics['total_trees'] = len(labels)
        
        # Crown statistics
        crown_values = crown_map[crown_map > 0]
        if len(crown_values) > 0:
            metrics['avg_crown_diameter'] = float(np.mean(crown_values))
            metrics['max_crown_diameter'] = float(np.max(crown_values))
            metrics['min_crown_diameter'] = float(np.min(crown_values))
        
        # Land cover distribution
        total_pixels = semantic_map.size
        for class_id in range(6):
            class_pixels = (semantic_map == class_id).sum()
            class_name = ['background', 'cacao_tree', 'shade_tree', 
                         'understory', 'bare_soil', 'shadows'][class_id]
            metrics[f'{class_name}_coverage'] = float(class_pixels / total_pixels)
        
        # Canopy density (estimated from tree coverage)
        tree_pixels = ((semantic_map == 1) | (semantic_map == 2)).sum()
        metrics['canopy_density'] = float(tree_pixels / total_pixels)
        
        return metrics
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file."""
        if image_path.endswith(('.tif', '.tiff')):
            with rasterio.open(image_path) as src:
                image = src.read()
                if image.shape[0] > 3:
                    image = image[:3]
                image = np.transpose(image, (1, 2, 0))
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image.dtype != np.uint8:
            image = (image / image.max() * 255).astype(np.uint8)
        
        return image
    
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
        if 'semantic' in outputs:
            semantic_pred = torch.argmax(outputs['semantic'], dim=1).squeeze()
            results['semantic_map'] = semantic_pred.cpu().numpy()
        
        # Instance segmentation
        if 'instances' in outputs:
            instances = outputs['instances']
            if isinstance(instances, list) and len(instances) > 0:
                instance_data = instances[0]
                results['instances'] = {
                    'boxes': instance_data.get('boxes', torch.tensor([])).cpu().numpy(),
                    'labels': instance_data.get('labels', torch.tensor([])).cpu().numpy(),
                    'scores': instance_data.get('scores', torch.tensor([])).cpu().numpy(),
                    'masks': instance_data.get('masks', torch.tensor([])).cpu().numpy()
                }
            else:
                results['instances'] = {
                    'boxes': np.array([]),
                    'labels': np.array([]),
                    'scores': np.array([]),
                    'masks': np.array([])
                }
        
        # Crown diameters
        if 'crown_diameters' in outputs:
            crown_pred = outputs['crown_diameters'].squeeze()
            results['crown_map'] = crown_pred.cpu().numpy()
        
        # Canopy density
        if 'canopy_density' in outputs:
            results['canopy_density'] = outputs['canopy_density'].cpu().item()
        
        return results
    
    def visualize_results(self, results: Dict, image_path: str = None, 
                         save_path: str = None, show: bool = True) -> plt.Figure:
        """
        Visualize segmentation results.
        
        Args:
            results: Prediction results
            image_path: Path to original image
            save_path: Path to save visualization
            show: Whether to display the figure
            
        Returns:
            Matplotlib figure
        """
        # Load original image if provided
        if image_path:
            image = self._load_image(image_path)
        elif 'image_path' in results:
            image = self._load_image(results['image_path'])
        else:
            h, w = results.get('semantic_map', np.zeros((512, 512))).shape
            image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Semantic segmentation
        if 'semantic_map' in results:
            semantic_colored = self._colorize_semantic(results['semantic_map'])
            axes[0, 1].imshow(semantic_colored)
            axes[0, 1].set_title('Semantic Segmentation')
            axes[0, 1].axis('off')
        
        # Instance segmentation
        if 'instances' in results:
            instance_viz = self._visualize_instances(image, results['instances'])
            axes[0, 2].imshow(instance_viz)
            axes[0, 2].set_title('Instance Segmentation')
            axes[0, 2].axis('off')
        
        # Crown diameter map
        if 'crown_map' in results:
            crown_viz = axes[1, 0].imshow(results['crown_map'], cmap='viridis')
            axes[1, 0].set_title('Crown Diameter Map')
            axes[1, 0].axis('off')
            plt.colorbar(crown_viz, ax=axes[1, 0], fraction=0.046)
        
        # Overlay visualization
        overlay = self._create_overlay(image, results)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Combined Overlay')
        axes[1, 1].axis('off')
        
        # Metrics display
        if 'metrics' in results:
            self._display_metrics(axes[1, 2], results['metrics'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def _colorize_semantic(self, semantic_map: np.ndarray) -> np.ndarray:
        """Convert semantic map to colored image."""
        h, w = semantic_map.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.SEMANTIC_COLORS.items():
            mask = semantic_map == class_id
            colored[mask] = color
        
        return colored
    
    def _visualize_instances(self, image: np.ndarray, instances: Dict) -> np.ndarray:
        """Visualize instance segmentation."""
        viz = image.copy()
        
        if not instances or len(instances.get('masks', [])) == 0:
            return viz
        
        masks = instances.get('masks', [])
        labels = instances.get('labels', [])
        boxes = instances.get('boxes', [])
        
        for mask, label, box in zip(masks, labels, boxes):
            if label in self.INSTANCE_COLORS:
                color = self.INSTANCE_COLORS[label]
                
                # Apply mask
                if mask.size > 0:
                    mask_bool = mask > 0.5
                    viz[mask_bool] = viz[mask_bool] * 0.5 + np.array(color) * 0.5
                
                # Draw bounding box
                if box.size == 4:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label_text = 'Cacao' if label == 1 else 'Shade Tree'
                    cv2.putText(viz, label_text, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return viz
    
    def _create_overlay(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """Create combined overlay visualization."""
        overlay = image.copy()
        
        # Add semantic segmentation with transparency
        if 'semantic_map' in results:
            semantic_colored = self._colorize_semantic(results['semantic_map'])
            mask = results['semantic_map'] > 0
            overlay[mask] = overlay[mask] * 0.6 + semantic_colored[mask] * 0.4
        
        # Add instance boundaries
        if 'instances' in results:
            masks = results['instances'].get('masks', [])
            for mask in masks:
                if mask.size > 0:
                    contours, _ = cv2.findContours(
                        (mask > 0.5).astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
        
        return overlay
    
    def _display_metrics(self, ax, metrics: Dict):
        """Display metrics in a subplot."""
        ax.axis('off')
        
        text = "Cabruca System Metrics\n" + "="*30 + "\n\n"
        
        # Tree counts
        text += "Tree Counts:\n"
        text += f"  Cacao Trees: {metrics.get('cacao_count', 0)}\n"
        text += f"  Shade Trees: {metrics.get('shade_tree_count', 0)}\n"
        text += f"  Total Trees: {metrics.get('total_trees', 0)}\n\n"
        
        # Crown statistics
        text += "Crown Diameters:\n"
        text += f"  Average: {metrics.get('avg_crown_diameter', 0):.2f} m\n"
        text += f"  Maximum: {metrics.get('max_crown_diameter', 0):.2f} m\n"
        text += f"  Minimum: {metrics.get('min_crown_diameter', 0):.2f} m\n\n"
        
        # Canopy density
        text += f"Canopy Density: {metrics.get('canopy_density', 0):.2%}\n\n"
        
        # Land cover
        text += "Land Cover Distribution:\n"
        for class_name in ['cacao_tree', 'shade_tree', 'understory', 'bare_soil', 'shadows']:
            coverage = metrics.get(f'{class_name}_coverage', 0)
            text += f"  {class_name.replace('_', ' ').title()}: {coverage:.1%}\n"
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    def export_to_geojson(self, results: Dict, image_path: str, 
                         output_path: str, crs: str = 'EPSG:4326'):
        """
        Export results to GeoJSON format for GIS integration.
        
        Args:
            results: Prediction results
            image_path: Path to georeferenced image
            output_path: Path to save GeoJSON
            crs: Coordinate reference system
        """
        features = []
        
        # Get georeferencing information
        with rasterio.open(image_path) as src:
            transform = src.transform
            img_crs = src.crs
        
        # Convert instances to polygons
        if 'instances' in results:
            masks = results['instances'].get('masks', [])
            labels = results['instances'].get('labels', [])
            
            for i, (mask, label) in enumerate(zip(masks, labels)):
                if mask.size > 0:
                    # Find contours
                    contours, _ = cv2.findContours(
                        (mask > 0.5).astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    for contour in contours:
                        if len(contour) >= 3:
                            # Convert pixel coordinates to geographic
                            coords = []
                            for point in contour.squeeze():
                                x, y = point
                                lon, lat = transform * (x, y)
                                coords.append([lon, lat])
                            
                            if len(coords) >= 3:
                                # Create polygon
                                polygon = Polygon(coords)
                                
                                # Calculate crown diameter
                                crown_diameter = np.sqrt(mask.sum() * 4 / np.pi)
                                
                                # Create feature
                                feature = {
                                    'type': 'Feature',
                                    'geometry': polygon.__geo_interface__,
                                    'properties': {
                                        'id': i,
                                        'class': 'cacao_tree' if label == 1 else 'shade_tree',
                                        'crown_diameter': float(crown_diameter),
                                        'area': float(polygon.area)
                                    }
                                }
                                features.append(feature)
        
        # Create GeoJSON
        geojson = {
            'type': 'FeatureCollection',
            'crs': {'init': crs},
            'features': features
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Exported {len(features)} features to {output_path}")


def run_inference(image_path: str, model_path: str, output_dir: str, config: Dict = None):
    """
    Run inference on an image and save results.
    
    Args:
        image_path: Path to input image
        model_path: Path to model checkpoint
        output_dir: Directory to save results
        config: Model configuration
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize inference pipeline
    inference = CabrucaInference(model_path, config)
    
    # Run prediction
    print(f"Processing {image_path}...")
    results = inference.predict_image(image_path)
    
    # Save visualization
    vis_path = os.path.join(output_dir, 'visualization.png')
    inference.visualize_results(results, save_path=vis_path, show=False)
    print(f"Saved visualization to {vis_path}")
    
    # Export to GeoJSON
    if image_path.endswith(('.tif', '.tiff')):
        geojson_path = os.path.join(output_dir, 'results.geojson')
        inference.export_to_geojson(results, image_path, geojson_path)
        print(f"Saved GeoJSON to {geojson_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results.get('metrics', {}), f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Cabruca Segmentation Inference')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./inference_results',
                       help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Run inference
    results = run_inference(args.image, args.model, args.output, config)
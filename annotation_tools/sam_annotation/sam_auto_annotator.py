"""
SAM-based Semi-Automated Annotation Tool
Leverages Segment Anything Model for zero-shot segmentation to accelerate annotation process
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import supervision as sv
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnnotationConfig:
    """Configuration for SAM-based annotation"""
    model_type: str = "vit_h"  # Can be vit_h, vit_l, or vit_b
    checkpoint_path: str = "models/sam/sam_vit_h_4b8939.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    points_per_side: int = 32  # For automatic mask generation
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    crop_n_layers: int = 1
    crop_n_points_downscale_factor: int = 2
    min_mask_region_area: int = 100
    
    # Class-specific thresholds
    cacao_min_area: int = 50
    cacao_max_area: int = 500
    shade_tree_min_area: int = 500
    shade_tree_max_area: int = 5000

class SAMAutoAnnotator:
    """Semi-automated annotation using SAM"""
    
    def __init__(self, config: AnnotationConfig):
        self.config = config
        self.device = config.device
        
        # Initialize SAM model
        logger.info(f"Loading SAM model: {config.model_type}")
        self.sam = sam_model_registry[config.model_type](checkpoint=config.checkpoint_path)
        self.sam.to(device=self.device)
        
        # Initialize mask generator for automatic segmentation
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=config.points_per_side,
            pred_iou_thresh=config.pred_iou_thresh,
            stability_score_thresh=config.stability_score_thresh,
            crop_n_layers=config.crop_n_layers,
            crop_n_points_downscale_factor=config.crop_n_points_downscale_factor,
            min_mask_region_area=config.min_mask_region_area,
        )
        
        # Initialize predictor for interactive segmentation
        self.predictor = SamPredictor(self.sam)
        
        logger.info("SAM model loaded successfully")
    
    def generate_automatic_masks(self, image: np.ndarray) -> List[Dict]:
        """Generate masks automatically using SAM"""
        logger.info("Generating automatic masks...")
        masks = self.mask_generator.generate(image)
        logger.info(f"Generated {len(masks)} masks")
        return masks
    
    def classify_mask_by_properties(self, mask: Dict, image: np.ndarray) -> int:
        """
        Classify a mask based on its properties
        Returns: 0 (background), 1 (cacao), 2 (shade_tree)
        """
        area = mask['area']
        bbox = mask['bbox']  # x, y, w, h
        
        # Extract region properties
        mask_binary = mask['segmentation']
        
        # Calculate additional features
        aspect_ratio = bbox[2] / bbox[3] if bbox[3] > 0 else 1
        
        # Get mean color in masked region
        masked_region = image[mask_binary]
        if len(masked_region) > 0:
            mean_color = np.mean(masked_region, axis=0)
            # Check for green vegetation (higher green channel)
            is_vegetation = mean_color[1] > mean_color[0] and mean_color[1] > mean_color[2]
        else:
            is_vegetation = False
        
        # Classification rules
        if not is_vegetation:
            return 0  # Background
        
        if self.config.cacao_min_area <= area <= self.config.cacao_max_area:
            # Small, compact objects likely to be cacao trees
            if 0.7 <= aspect_ratio <= 1.3:  # Roughly circular
                return 1  # Cacao
        
        if self.config.shade_tree_min_area <= area <= self.config.shade_tree_max_area:
            # Larger objects likely to be shade trees
            return 2  # Shade tree
        
        return 0  # Default to background
    
    def interactive_segmentation(self, image: np.ndarray, points: np.ndarray, 
                                labels: np.ndarray) -> Dict:
        """
        Perform interactive segmentation with user-provided points
        
        Args:
            image: Input image
            points: Array of points (N, 2)
            labels: Array of labels (N,) - 1 for foreground, 0 for background
        """
        self.predictor.set_image(image)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        return {
            'segmentation': best_mask,
            'score': scores[best_idx],
            'area': np.sum(best_mask)
        }
    
    def refine_annotations(self, image: np.ndarray, initial_masks: List[Dict],
                          user_feedback: Optional[Dict] = None) -> List[Dict]:
        """
        Refine automatic annotations based on user feedback
        
        Args:
            image: Input image
            initial_masks: Initial masks from automatic generation
            user_feedback: Dictionary with corrections
                          {mask_id: {'class': int, 'points': list, 'labels': list}}
        """
        refined_masks = []
        
        for i, mask in enumerate(initial_masks):
            if user_feedback and i in user_feedback:
                # Apply user corrections
                feedback = user_feedback[i]
                
                if 'delete' in feedback and feedback['delete']:
                    continue  # Skip deleted masks
                
                if 'points' in feedback:
                    # Refine mask with user points
                    refined_mask = self.interactive_segmentation(
                        image,
                        np.array(feedback['points']),
                        np.array(feedback['labels'])
                    )
                    mask['segmentation'] = refined_mask['segmentation']
                    mask['score'] = refined_mask['score']
                
                if 'class' in feedback:
                    mask['class_id'] = feedback['class']
            else:
                # Auto-classify if no user feedback
                mask['class_id'] = self.classify_mask_by_properties(mask, image)
            
            refined_masks.append(mask)
        
        return refined_masks
    
    def masks_to_coco_annotations(self, masks: List[Dict], image_id: str,
                                 image_shape: Tuple[int, int]) -> List[Dict]:
        """Convert SAM masks to COCO format annotations"""
        annotations = []
        
        for i, mask in enumerate(masks):
            # Convert mask to polygon
            mask_binary = mask['segmentation'].astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Simplify contour
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 3:  # Valid polygon
                    segmentation = approx.flatten().tolist()
                    
                    # Calculate bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    annotation = {
                        'id': f"{image_id}_{i}",
                        'image_id': image_id,
                        'category_id': mask.get('class_id', 0),
                        'segmentation': [segmentation],
                        'area': float(mask.get('area', cv2.contourArea(contour))),
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'iscrowd': 0,
                        'score': float(mask.get('score', 1.0))
                    }
                    annotations.append(annotation)
        
        return annotations
    
    def annotate_image(self, image_path: str, output_dir: str,
                       interactive: bool = False) -> Dict:
        """
        Annotate a single image
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save annotations
            interactive: Whether to use interactive refinement
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate automatic masks
        masks = self.generate_automatic_masks(image_rgb)
        
        # Classify masks
        for mask in masks:
            mask['class_id'] = self.classify_mask_by_properties(mask, image_rgb)
        
        # Generate image ID
        image_id = hashlib.md5(image_path.encode()).hexdigest()[:8]
        
        # Convert to COCO annotations
        annotations = self.masks_to_coco_annotations(masks, image_id, image.shape[:2])
        
        # Prepare output
        output_data = {
            'image_path': image_path,
            'image_id': image_id,
            'width': image.shape[1],
            'height': image.shape[0],
            'annotations': annotations,
            'metadata': {
                'tool': 'sam_auto_annotator',
                'timestamp': datetime.now().isoformat(),
                'config': asdict(self.config)
            }
        }
        
        # Save annotations
        output_path = Path(output_dir) / f"{Path(image_path).stem}_sam_annotations.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved annotations to {output_path}")
        
        return output_data
    
    def batch_annotate(self, image_dir: str, output_dir: str,
                       image_extensions: List[str] = ['.jpg', '.png', '.tif']) -> List[Dict]:
        """Annotate multiple images in batch"""
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(Path(image_dir).glob(f"*{ext}"))
        
        logger.info(f"Found {len(image_paths)} images to annotate")
        
        all_annotations = []
        for image_path in tqdm(image_paths, desc="Annotating images"):
            try:
                annotations = self.annotate_image(str(image_path), output_dir)
                all_annotations.append(annotations)
            except Exception as e:
                logger.error(f"Error annotating {image_path}: {e}")
                continue
        
        return all_annotations
    
    def visualize_annotations(self, image_path: str, annotations: List[Dict],
                             save_path: Optional[str] = None):
        """Visualize annotations on image"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create color map for classes
        colors = {
            0: (128, 128, 128),  # Background - gray
            1: (0, 255, 0),      # Cacao - green
            2: (255, 0, 0)       # Shade tree - red
        }
        
        # Draw annotations
        for ann in annotations:
            class_id = ann['category_id']
            color = colors.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            x, y, w, h = ann['bbox']
            cv2.rectangle(image_rgb, (int(x), int(y)), 
                         (int(x+w), int(y+h)), color, 2)
            
            # Draw segmentation if available
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(image_rgb, [points], True, color, 2)
        
        # Display or save
        if save_path:
            plt.figure(figsize=(12, 8))
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.title("SAM Auto-Annotations")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.figure(figsize=(12, 8))
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.title("SAM Auto-Annotations")
            plt.show()
        
        return image_rgb


def main():
    """Example usage of SAM auto-annotator"""
    
    # Configure
    config = AnnotationConfig(
        model_type="vit_h",
        checkpoint_path="models/sam/sam_vit_h_4b8939.pth",
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=100,
        cacao_min_area=50,
        cacao_max_area=500,
        shade_tree_min_area=500,
        shade_tree_max_area=5000
    )
    
    # Initialize annotator
    annotator = SAMAutoAnnotator(config)
    
    # Example: Annotate single image
    # image_path = "data/raw/sample_image.jpg"
    # output_dir = "data/annotations/sam"
    # annotations = annotator.annotate_image(image_path, output_dir)
    
    # Example: Batch annotation
    # image_dir = "data/raw"
    # output_dir = "data/annotations/sam"
    # all_annotations = annotator.batch_annotate(image_dir, output_dir)
    
    logger.info("SAM auto-annotation setup complete")


if __name__ == "__main__":
    main()

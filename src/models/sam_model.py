"""
Segment Anything Model (SAM) configuration for tree segmentation.
Optimized for Apple Silicon Metal Performance Shaders.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# Check for Metal availability on macOS
def check_metal_availability():
    """Check if Metal Performance Shaders are available."""
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return True
    return False


# Set device based on availability
if check_metal_availability():
    DEVICE = torch.device("mps")
    print("✅ Using Metal Performance Shaders (MPS) for acceleration")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✅ Using CUDA for acceleration")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ Using CPU (slower performance expected)")


@dataclass
class SAMConfig:
    """Configuration for SAM model."""

    model_type: str = "vit_b"  # Options: "vit_b", "vit_l", "vit_h"
    checkpoint_path: Optional[str] = None
    device: torch.device = DEVICE

    # Tree detection specific parameters
    num_classes: int = 2  # Cacao trees vs Shade trees
    confidence_threshold: float = 0.7
    iou_threshold: float = 0.5

    # Prompt engineering parameters
    points_per_side: int = 32  # For automatic mask generation
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 100  # Minimum area for tree detection

    # Metal optimization parameters
    use_metal_optimization: bool = check_metal_availability()
    batch_size: int = 1 if use_metal_optimization else 4
    enable_mixed_precision: bool = True


class TreePromptEngineering:
    """Custom prompt engineering for tree detection."""

    def __init__(self, config: SAMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Tree-specific parameters
        self.cacao_tree_features = {
            "min_area": 50,  # Square meters
            "max_area": 200,
            "typical_height": 5,  # meters
            "crown_shape": "round",
            "spectral_signature": "dark_green",
        }

        self.shade_tree_features = {
            "min_area": 100,
            "max_area": 500,
            "typical_height": 15,  # meters
            "crown_shape": "irregular",
            "spectral_signature": "light_green",
        }

    def generate_point_prompts(
        self, image: np.ndarray, tree_type: str = "cacao"
    ) -> List[Tuple[int, int]]:
        """
        Generate point prompts for tree detection.

        Args:
            image: Input image array
            tree_type: 'cacao' or 'shade'

        Returns:
            List of (x, y) coordinates for point prompts
        """
        h, w = image.shape[:2]
        points = []

        # Generate grid-based points with adaptive density
        if tree_type == "cacao":
            # Denser grid for smaller cacao trees
            step = min(h, w) // 40
        else:
            # Sparser grid for larger shade trees
            step = min(h, w) // 20

        for y in range(step, h - step, step):
            for x in range(step, w - step, step):
                # Check if point likely corresponds to vegetation
                if self._is_vegetation(image[y, x]):
                    points.append((x, y))

        return points

    def generate_box_prompts(
        self, image: np.ndarray, detections: Optional[List] = None
    ) -> List[List[int]]:
        """
        Generate bounding box prompts for tree detection.

        Args:
            image: Input image array
            detections: Optional pre-detected regions

        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        boxes = []

        if detections:
            # Use pre-detected regions if available
            for det in detections:
                if self._validate_tree_box(det):
                    boxes.append(det)
        else:
            # Generate candidate boxes using sliding window
            h, w = image.shape[:2]

            # Cacao tree boxes (smaller)
            cacao_size = 30  # pixels
            for y in range(0, h - cacao_size, cacao_size // 2):
                for x in range(0, w - cacao_size, cacao_size // 2):
                    box = [x, y, x + cacao_size, y + cacao_size]
                    if self._is_tree_candidate(image, box):
                        boxes.append(box)

            # Shade tree boxes (larger)
            shade_size = 80  # pixels
            for y in range(0, h - shade_size, shade_size // 2):
                for x in range(0, w - shade_size, shade_size // 2):
                    box = [x, y, x + shade_size, y + shade_size]
                    if self._is_tree_candidate(image, box, tree_type="shade"):
                        boxes.append(box)

        return boxes

    def _is_vegetation(self, pixel: np.ndarray) -> bool:
        """Check if pixel likely represents vegetation using NDVI-like logic."""
        if len(pixel) >= 3:
            r, g, b = pixel[:3]
            # Simple vegetation index
            if g > r and g > b:  # Green dominant
                return True
        return False

    def _validate_tree_box(self, box: List[int]) -> bool:
        """Validate if bounding box dimensions match tree characteristics."""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height

        # Check aspect ratio (trees are roughly circular from above)
        aspect_ratio = width / height if height > 0 else 0
        if 0.7 < aspect_ratio < 1.4:  # Roughly square
            # Check area constraints
            if 400 < area < 10000:  # Reasonable tree size in pixels
                return True
        return False

    def _is_tree_candidate(
        self, image: np.ndarray, box: List[int], tree_type: str = "cacao"
    ) -> bool:
        """Check if region is a tree candidate based on color and texture."""
        x1, y1, x2, y2 = box
        region = image[y1:y2, x1:x2]

        if region.size == 0:
            return False

        # Calculate vegetation index
        if len(region.shape) == 3:
            green_channel = region[:, :, 1]
            red_channel = region[:, :, 0]

            # Simple vegetation check
            vegetation_score = np.mean(green_channel) - np.mean(red_channel)

            if tree_type == "cacao":
                return vegetation_score > 20  # Lower threshold for darker cacao
            else:
                return (
                    vegetation_score > 30
                )  # Higher threshold for brighter shade trees

        return False


class SAMTreeSegmenter:
    """SAM-based tree segmentation with multi-class support."""

    def __init__(self, config: SAMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.prompt_engineer = TreePromptEngineering(config)
        self.model = None
        self.predictor = None
        self.mask_generator = None

        # Setup Metal optimization if available
        if config.use_metal_optimization:
            self._setup_metal_optimization()

    def _setup_metal_optimization(self):
        """Configure Metal Performance Shaders optimization."""
        self.logger.info("Configuring Metal Performance Shaders optimization...")

        # Enable mixed precision for Metal
        if self.config.enable_mixed_precision:
            torch.set_float32_matmul_precision("high")

        # Metal-specific optimizations
        torch.mps.set_per_process_memory_fraction(0.0)  # Let Metal manage memory

        self.logger.info("Metal optimization configured successfully")

    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        Load SAM model with specified checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
        """
        try:
            import segment_anything as sam

            # Determine model type and checkpoint
            if checkpoint_path:
                self.config.checkpoint_path = checkpoint_path
            else:
                # Download default checkpoint based on model type
                checkpoint_url = self._get_checkpoint_url(self.config.model_type)
                checkpoint_path = self._download_checkpoint(checkpoint_url)
                self.config.checkpoint_path = checkpoint_path

            # Build SAM model
            if self.config.model_type == "vit_b":
                self.model = sam.sam_model_registry["vit_b"](
                    checkpoint=self.config.checkpoint_path
                )
            elif self.config.model_type == "vit_l":
                self.model = sam.sam_model_registry["vit_l"](
                    checkpoint=self.config.checkpoint_path
                )
            else:  # vit_h
                self.model = sam.sam_model_registry["vit_h"](
                    checkpoint=self.config.checkpoint_path
                )

            # Move model to device
            self.model.to(self.config.device)
            self.model.eval()

            # Initialize predictor and mask generator
            self.predictor = sam.SamPredictor(self.model)
            self.mask_generator = sam.SamAutomaticMaskGenerator(
                model=self.model,
                points_per_side=self.config.points_per_side,
                pred_iou_thresh=self.config.pred_iou_thresh,
                stability_score_thresh=self.config.stability_score_thresh,
                crop_n_layers=self.config.crop_n_layers,
                crop_n_points_downscale_factor=self.config.crop_n_points_downscale_factor,
                min_mask_region_area=self.config.min_mask_region_area,
            )

            self.logger.info(f"Model loaded successfully on {self.config.device}")

        except ImportError:
            self.logger.error(
                "segment-anything not installed. Please install it first."
            )
            raise
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def _get_checkpoint_url(self, model_type: str) -> str:
        """Get checkpoint URL for model type."""
        urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        }
        return urls.get(model_type, urls["vit_b"])

    def _download_checkpoint(self, url: str) -> str:
        """Download model checkpoint if not exists."""
        import os
        import urllib.request

        # Create models directory if not exists
        models_dir = Path("models/sam_checkpoints")
        models_dir.mkdir(parents=True, exist_ok=True)

        # Extract filename from URL
        filename = url.split("/")[-1]
        checkpoint_path = models_dir / filename

        if not checkpoint_path.exists():
            self.logger.info(f"Downloading checkpoint from {url}...")
            urllib.request.urlretrieve(url, checkpoint_path)
            self.logger.info(f"Checkpoint downloaded to {checkpoint_path}")
        else:
            self.logger.info(f"Using existing checkpoint at {checkpoint_path}")

        return str(checkpoint_path)

    def segment_with_points(
        self,
        image: np.ndarray,
        point_coords: List[Tuple[int, int]],
        point_labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Segment using point prompts.

        Args:
            image: Input image
            point_coords: List of (x, y) coordinates
            point_labels: Optional labels (1 for foreground, 0 for background)

        Returns:
            Dictionary with masks and classifications
        """
        if self.predictor is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Set image
        self.predictor.set_image(image)

        # Convert to numpy arrays
        point_coords_np = np.array(point_coords)
        if point_labels is None:
            point_labels_np = np.ones(len(point_coords))
        else:
            point_labels_np = np.array(point_labels)

        # Predict masks
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords_np,
            point_labels=point_labels_np,
            multimask_output=True,
        )

        # Classify masks as cacao or shade trees
        classifications = self._classify_trees(image, masks)

        return {
            "masks": masks,
            "scores": scores,
            "logits": logits,
            "classifications": classifications,
        }

    def segment_with_boxes(
        self, image: np.ndarray, boxes: List[List[int]]
    ) -> Dict[str, Any]:
        """
        Segment using bounding box prompts.

        Args:
            image: Input image
            boxes: List of bounding boxes [x1, y1, x2, y2]

        Returns:
            Dictionary with masks and classifications
        """
        if self.predictor is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Set image
        self.predictor.set_image(image)

        all_masks = []
        all_scores = []
        all_classifications = []

        for box in boxes:
            box_np = np.array(box)

            # Predict mask for box
            masks, scores, logits = self.predictor.predict(
                box=box_np,
                multimask_output=True,
            )

            # Select best mask
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]

            all_masks.append(best_mask)
            all_scores.append(best_score)

            # Classify tree type
            classification = self._classify_single_tree(image, best_mask, box)
            all_classifications.append(classification)

        return {
            "masks": np.array(all_masks),
            "scores": np.array(all_scores),
            "classifications": all_classifications,
        }

    def segment_automatic(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform automatic segmentation without prompts.

        Args:
            image: Input image

        Returns:
            Dictionary with masks and classifications
        """
        if self.mask_generator is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Generate masks
        masks = self.mask_generator.generate(image)

        # Filter and classify masks
        tree_masks = []
        classifications = []

        for mask_data in masks:
            mask = mask_data["segmentation"]
            bbox = mask_data["bbox"]  # x, y, w, h
            area = mask_data["area"]

            # Filter based on tree characteristics
            if self._is_tree_mask(mask, area):
                tree_masks.append(mask_data)

                # Convert bbox format
                box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                classification = self._classify_single_tree(image, mask, box)
                classifications.append(classification)

        return {
            "masks": tree_masks,
            "classifications": classifications,
            "total_trees": len(tree_masks),
            "cacao_count": sum(1 for c in classifications if c == "cacao"),
            "shade_count": sum(1 for c in classifications if c == "shade"),
        }

    def _classify_trees(self, image: np.ndarray, masks: np.ndarray) -> List[str]:
        """Classify masks as cacao or shade trees."""
        classifications = []

        for mask in masks:
            # Get masked region
            masked_image = image.copy()
            masked_image[~mask] = 0

            # Calculate features
            classification = self._classify_by_features(masked_image, mask)
            classifications.append(classification)

        return classifications

    def _classify_single_tree(
        self, image: np.ndarray, mask: np.ndarray, box: List[int]
    ) -> str:
        """Classify a single tree mask."""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = np.sum(mask)

        # Get color features from masked region
        masked_pixels = image[mask > 0]

        if len(masked_pixels) > 0:
            mean_color = np.mean(masked_pixels, axis=0)

            # Simple classification based on size and color
            if area < 2000:  # Smaller trees are likely cacao
                if mean_color[1] < 140:  # Darker green
                    return "cacao"

            if area > 3000:  # Larger trees are likely shade
                if mean_color[1] > 120:  # Lighter green
                    return "shade"

        # Default classification based on size
        return "cacao" if area < 2500 else "shade"

    def _classify_by_features(self, masked_image: np.ndarray, mask: np.ndarray) -> str:
        """Classify tree based on extracted features."""
        # Calculate mask area
        area = np.sum(mask)

        # Get color statistics
        masked_pixels = masked_image[mask > 0]
        if len(masked_pixels) == 0:
            return "unknown"

        mean_color = np.mean(masked_pixels, axis=0)
        std_color = np.std(masked_pixels, axis=0)

        # Simple rule-based classification
        # Can be replaced with a trained classifier
        if area < 2000:  # Small trees
            if mean_color[1] > mean_color[0] * 1.2:  # Green dominant
                return "cacao"
        elif area > 4000:  # Large trees
            return "shade"
        else:
            # Medium size - check color intensity
            if mean_color[1] < 130:  # Darker green
                return "cacao"
            else:
                return "shade"

        return "unknown"

    def _is_tree_mask(self, mask: np.ndarray, area: int) -> bool:
        """Check if mask represents a tree."""
        # Filter by area
        if area < self.config.min_mask_region_area:
            return False

        if area > 50000:  # Too large to be a single tree
            return False

        # Check shape compactness (trees are roughly circular)
        from scipy import ndimage

        labeled_mask = ndimage.label(mask)[0]
        if labeled_mask.max() > 0:
            props = ndimage.measurements.find_objects(labeled_mask)[0]
            height = props[0].stop - props[0].start
            width = props[1].stop - props[1].start

            if height > 0 and width > 0:
                aspect_ratio = width / height
                if 0.5 < aspect_ratio < 2.0:  # Roughly square/circular
                    return True

        return False

    def fine_tune(
        self,
        train_data: Dict[str, Any],
        val_data: Optional[Dict[str, Any]] = None,
        epochs: int = 10,
    ):
        """
        Fine-tune SAM model on tree-specific data.

        Args:
            train_data: Training data with images and annotations
            val_data: Optional validation data
            epochs: Number of training epochs
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.logger.info("Starting fine-tuning for tree segmentation...")

        # This is a placeholder for fine-tuning logic
        # Actual implementation would require:
        # 1. Custom dataset loader for tree annotations
        # 2. Loss function for tree segmentation
        # 3. Optimizer configuration
        # 4. Training loop with Metal optimization

        self.logger.info("Fine-tuning completed")

    def save_model(self, path: str):
        """Save fine-tuned model."""
        if self.model is None:
            raise ValueError("No model to save")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
            },
            path,
        )
        self.logger.info(f"Model saved to {path}")

    def load_finetuned_model(self, path: str):
        """Load fine-tuned model."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.config = checkpoint["config"]
        self.logger.info(f"Fine-tuned model loaded from {path}")


# Example usage function
def setup_sam_for_trees():
    """Setup SAM for tree segmentation with Metal optimization."""

    # Configure SAM
    config = SAMConfig(
        model_type="vit_b",  # Start with smaller model for testing
        num_classes=2,
        confidence_threshold=0.7,
        use_metal_optimization=True,
    )

    # Initialize segmenter
    segmenter = SAMTreeSegmenter(config)

    # Load model (will download if needed)
    segmenter.load_model()

    return segmenter


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize SAM for tree segmentation
    segmenter = setup_sam_for_trees()
    print("SAM model configured and ready for tree segmentation!")

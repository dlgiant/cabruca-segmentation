"""
Multi-class segmentation model for Cabruca agroforestry systems.
Combines instance segmentation for individual trees with semantic segmentation for land cover.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import SamPredictor, sam_model_registry
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.segmentation import deeplabv3_resnet101


class CabrucaSegmentationModel(nn.Module):
    """
    Dual-head model for Cabruca system analysis combining:
    1. Instance segmentation for individual tree detection
    2. Semantic segmentation for land cover classification
    """

    # Class definitions
    INSTANCE_CLASSES = {0: "background", 1: "cacao_tree", 2: "shade_tree"}

    SEMANTIC_CLASSES = {
        0: "background",
        1: "cacao_tree",
        2: "shade_tree",
        3: "understory",
        4: "bare_soil",
        5: "shadows",
    }

    def __init__(
        self,
        num_instance_classes=3,
        num_semantic_classes=6,
        use_sam=False,
        sam_checkpoint=None,
    ):
        """
        Initialize the Cabruca segmentation model.

        Args:
            num_instance_classes: Number of instance classes (trees)
            num_semantic_classes: Number of semantic classes (land cover)
            use_sam: Whether to use SAM for refinement
            sam_checkpoint: Path to SAM checkpoint
        """
        super().__init__()

        self.num_instance_classes = num_instance_classes
        self.num_semantic_classes = num_semantic_classes

        # Instance segmentation head (Mask R-CNN)
        self.instance_head = maskrcnn_resnet50_fpn(
            pretrained=False,
            num_classes=91,  # COCO classes, we'll filter in post-processing
        )
        self.instance_head.roi_heads.box_predictor = self._get_instance_predictor(
            num_instance_classes
        )

        # Semantic segmentation head (DeepLabV3+)
        self.semantic_head = deeplabv3_resnet101(
            pretrained=True, num_classes=num_semantic_classes
        )

        # Feature fusion layers
        self.fusion_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.fusion_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.fusion_bn1 = nn.BatchNorm2d(128)
        self.fusion_bn2 = nn.BatchNorm2d(64)

        # Crown diameter estimation head
        self.crown_estimator = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),  # Output: diameter map
        )

        # Canopy density estimation head
        self.density_estimator = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Output: density value [0, 1]
        )

        # SAM integration for refinement
        self.use_sam = use_sam
        if use_sam and sam_checkpoint:
            self.sam_predictor = self._init_sam(sam_checkpoint)
        else:
            self.sam_predictor = None

    def _get_instance_predictor(self, num_classes):
        """Get custom predictor for instance segmentation."""
        # Box predictor
        in_features = self.instance_head.roi_heads.box_predictor.cls_score.in_features
        box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Mask predictor
        in_features_mask = (
            self.instance_head.roi_heads.mask_predictor.conv5_mask.in_channels
        )
        hidden_layer = 256
        mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        self.instance_head.roi_heads.mask_predictor = mask_predictor
        return box_predictor

    def _init_sam(self, checkpoint_path):
        """Initialize SAM model for mask refinement."""
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam.eval()
        if torch.cuda.is_available():
            sam = sam.cuda()
        return SamPredictor(sam)

    def forward(self, images: torch.Tensor, targets=None):
        """
        Forward pass through the model.

        Args:
            images: Input tensor of shape (B, C, H, W)
            targets: Training targets (optional)

        Returns:
            Dictionary containing predictions for all tasks
        """
        # Instance segmentation
        if self.training and targets is not None:
            instance_outputs = self.instance_head(images, targets)
        else:
            instance_outputs = self.instance_head(images)

        # Semantic segmentation
        semantic_outputs = self.semantic_head(images)["out"]

        # Extract features for fusion
        fusion_features = self._extract_fusion_features(images)

        # Apply fusion layers
        fusion_features = F.relu(self.fusion_bn1(self.fusion_conv1(fusion_features)))
        fusion_features = F.relu(self.fusion_bn2(self.fusion_conv2(fusion_features)))

        # Crown diameter estimation
        crown_diameters = self.crown_estimator(fusion_features)

        # Canopy density calculation
        canopy_density = self.density_estimator(fusion_features)

        outputs = {
            "instances": instance_outputs,
            "semantic": semantic_outputs,
            "crown_diameters": crown_diameters,
            "canopy_density": canopy_density,
        }

        # SAM refinement (inference only)
        if not self.training and self.sam_predictor:
            outputs = self._refine_with_sam(images, outputs)

        return outputs

    def _extract_fusion_features(self, images):
        """Extract intermediate features for fusion."""
        # This is a simplified version - in practice, you'd extract from backbone
        features = F.interpolate(
            images, size=(images.shape[2] // 4, images.shape[3] // 4)
        )
        features = F.conv2d(features, torch.randn(256, 3, 3, 3).to(images.device) * 0.1)
        return features

    def _refine_with_sam(self, images, outputs):
        """Refine masks using SAM."""
        refined_outputs = outputs.copy()

        if self.sam_predictor and "instances" in outputs:
            for i, instance in enumerate(outputs["instances"]):
                if "masks" in instance:
                    # Convert to numpy for SAM
                    image_np = images[i].cpu().numpy().transpose(1, 2, 0)
                    self.sam_predictor.set_image(image_np)

                    # Refine each mask
                    refined_masks = []
                    for mask in instance["masks"]:
                        # Use mask as prompt for SAM
                        mask_np = mask.cpu().numpy()
                        refined_mask = self._sam_refine_mask(mask_np)
                        refined_masks.append(torch.from_numpy(refined_mask))

                    instance["masks"] = torch.stack(refined_masks)

        return refined_outputs

    def _sam_refine_mask(self, mask):
        """Refine a single mask using SAM."""
        # Find bounding box from mask
        y_indices, x_indices = np.where(mask > 0.5)
        if len(x_indices) == 0:
            return mask

        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # Use box as prompt
        box = np.array([x_min, y_min, x_max, y_max])
        masks, _, _ = self.sam_predictor.predict(box=box, multimask_output=False)

        return masks[0]

    def calculate_tree_metrics(self, outputs: Dict) -> Dict:
        """
        Calculate tree-specific metrics from model outputs.

        Args:
            outputs: Model predictions

        Returns:
            Dictionary with tree metrics
        """
        metrics = {}

        # Extract instance masks
        if "instances" in outputs:
            instances = outputs["instances"]

            # Count trees by type
            if isinstance(instances, list):
                for inst in instances:
                    if "labels" in inst:
                        labels = inst["labels"].cpu().numpy()
                        metrics["cacao_count"] = np.sum(labels == 1)
                        metrics["shade_tree_count"] = np.sum(labels == 2)

            # Calculate crown areas
            if "crown_diameters" in outputs:
                crown_map = outputs["crown_diameters"].squeeze()
                metrics["avg_crown_diameter"] = crown_map[crown_map > 0].mean().item()
                metrics["total_crown_area"] = (crown_map > 0).sum().item()

        # Canopy density
        if "canopy_density" in outputs:
            metrics["canopy_density"] = outputs["canopy_density"].item()

        # Semantic class distribution
        if "semantic" in outputs:
            semantic_pred = torch.argmax(outputs["semantic"], dim=1)
            total_pixels = semantic_pred.numel()

            for class_id, class_name in self.SEMANTIC_CLASSES.items():
                class_pixels = (semantic_pred == class_id).sum().item()
                metrics[f"{class_name}_coverage"] = class_pixels / total_pixels

        return metrics


class CabrucaLoss(nn.Module):
    """
    Combined loss function for multi-task learning in Cabruca segmentation.
    """

    def __init__(
        self,
        instance_weight=1.0,
        semantic_weight=1.0,
        crown_weight=0.5,
        density_weight=0.5,
    ):
        super().__init__()
        self.instance_weight = instance_weight
        self.semantic_weight = semantic_weight
        self.crown_weight = crown_weight
        self.density_weight = density_weight

        self.semantic_loss = nn.CrossEntropyLoss()
        self.crown_loss = nn.MSELoss()
        self.density_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        """
        Calculate combined loss.

        Args:
            outputs: Model predictions
            targets: Ground truth targets

        Returns:
            Total loss and individual components
        """
        total_loss = 0
        losses = {}

        # Instance segmentation loss (handled by Mask R-CNN internally during training)
        if "instances" in outputs and isinstance(outputs["instances"], dict):
            instance_loss = sum(loss for loss in outputs["instances"].values())
            total_loss += self.instance_weight * instance_loss
            losses["instance"] = instance_loss

        # Semantic segmentation loss
        if "semantic" in outputs and "semantic_masks" in targets:
            semantic_loss = self.semantic_loss(
                outputs["semantic"], targets["semantic_masks"]
            )
            total_loss += self.semantic_weight * semantic_loss
            losses["semantic"] = semantic_loss

        # Crown diameter loss
        if "crown_diameters" in outputs and "crown_gt" in targets:
            crown_loss = self.crown_loss(
                outputs["crown_diameters"], targets["crown_gt"]
            )
            total_loss += self.crown_weight * crown_loss
            losses["crown"] = crown_loss

        # Canopy density loss
        if "canopy_density" in outputs and "density_gt" in targets:
            density_loss = self.density_loss(
                outputs["canopy_density"], targets["density_gt"]
            )
            total_loss += self.density_weight * density_loss
            losses["density"] = density_loss

        losses["total"] = total_loss
        return total_loss, losses


def create_cabruca_model(config: Dict) -> CabrucaSegmentationModel:
    """
    Factory function to create Cabruca segmentation model.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized model
    """
    model = CabrucaSegmentationModel(
        num_instance_classes=config.get("num_instance_classes", 3),
        num_semantic_classes=config.get("num_semantic_classes", 6),
        use_sam=config.get("use_sam", False),
        sam_checkpoint=config.get("sam_checkpoint", None),
    )

    if config.get("pretrained_weights"):
        model.load_state_dict(torch.load(config["pretrained_weights"]))

    return model

"""
Dataset and data loading pipeline for Cabruca multi-class segmentation.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from rasterio.windows import Window
from shapely.geometry import box
from torch.utils.data import DataLoader, Dataset


class CabrucaDataset(Dataset):
    """
    Dataset for Cabruca agroforestry system segmentation.
    Supports multiple annotation formats and multi-task learning.
    """

    def __init__(
        self,
        data_dir: str,
        annotation_file: str = None,
        mode: str = "train",
        transform=None,
        use_augmentation: bool = True,
        tile_size: int = 512,
        overlap: int = 64,
    ):
        """
        Initialize Cabruca dataset.

        Args:
            data_dir: Root directory containing images
            annotation_file: Path to COCO-format annotations
            mode: 'train', 'val', or 'test'
            transform: Additional transformations
            use_augmentation: Whether to apply augmentations
            tile_size: Size of image tiles
            overlap: Overlap between tiles
        """
        self.data_dir = data_dir
        self.mode = mode
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform

        # Load annotations
        if annotation_file and os.path.exists(annotation_file):
            self.coco = COCO(annotation_file)
            self.image_ids = list(self.coco.imgs.keys())
        else:
            # Load images without annotations (for inference)
            self.image_files = self._load_image_files()
            self.image_ids = list(range(len(self.image_files)))
            self.coco = None

        # Define augmentations
        if use_augmentation and mode == "train":
            self.augmentation = self._get_augmentation_pipeline()
        else:
            self.augmentation = self._get_validation_pipeline()

        # Class mappings
        self.instance_classes = {"cacao_tree": 1, "shade_tree": 2}

        self.semantic_classes = {
            "background": 0,
            "cacao_tree": 1,
            "shade_tree": 2,
            "understory": 3,
            "bare_soil": 4,
            "shadows": 5,
        }

    def _load_image_files(self):
        """Load image files from directory."""
        image_extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
        image_files = []

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))

        return sorted(image_files)

    def _get_augmentation_pipeline(self):
        """Define augmentation pipeline for training."""
        return A.Compose(
            [
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
                ),
                # Atmospheric effects
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.1),
                # Geometric augmentations
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3
                ),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                # Normalize and convert
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["keypoint_labels"]
            ),
        )

    def _get_validation_pipeline(self):
        """Define preprocessing pipeline for validation/test."""
        return A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Get item from dataset.

        Returns:
            Dictionary containing:
                - image: Tensor of shape (C, H, W)
                - instances: Instance segmentation targets
                - semantic_mask: Semantic segmentation mask
                - crown_map: Crown diameter ground truth
                - metadata: Additional information
        """
        image_id = self.image_ids[idx]

        # Load image
        if self.coco:
            img_info = self.coco.imgs[image_id]
            image_path = os.path.join(self.data_dir, img_info["file_name"])
        else:
            image_path = self.image_files[idx]

        image = self._load_image(image_path)

        # Initialize targets
        targets = {"image_id": image_id, "image_path": image_path}

        # Load annotations if available
        if self.coco:
            targets.update(self._load_annotations(image_id, image.shape[:2]))
        else:
            # Create empty targets for inference
            h, w = image.shape[:2]
            targets["boxes"] = np.array([[0, 0, 1, 1]], dtype=np.float32)
            targets["labels"] = np.array([0], dtype=np.int64)
            targets["masks"] = np.zeros((1, h, w), dtype=np.uint8)
            targets["semantic_mask"] = np.zeros((h, w), dtype=np.int64)
            targets["crown_map"] = np.zeros((h, w), dtype=np.float32)

        # Apply augmentations
        if self.augmentation:
            augmented = self.augmentation(
                image=image,
                mask=targets["semantic_mask"],
                masks=targets["masks"],
                bboxes=targets["boxes"],
                labels=targets["labels"],
            )

            image = augmented["image"]
            targets["semantic_mask"] = augmented["mask"]
            if "masks" in augmented:
                targets["masks"] = np.array(augmented["masks"])
            if "bboxes" in augmented:
                targets["boxes"] = np.array(augmented["bboxes"])

        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        targets["boxes"] = torch.from_numpy(targets["boxes"]).float()
        targets["labels"] = torch.from_numpy(targets["labels"]).long()
        targets["masks"] = torch.from_numpy(targets["masks"]).float()
        targets["semantic_mask"] = torch.from_numpy(targets["semantic_mask"]).long()
        targets["crown_map"] = torch.from_numpy(targets["crown_map"]).float()

        # Apply additional transforms
        if self.transform:
            image = self.transform(image)

        return image, targets

    def _load_image(self, image_path):
        """Load image from file."""
        if image_path.endswith((".tif", ".tiff")):
            # Load GeoTIFF
            with rasterio.open(image_path) as src:
                image = src.read()
                if image.shape[0] > 3:
                    # Take first 3 bands if more than RGB
                    image = image[:3]
                image = np.transpose(image, (1, 2, 0))
        else:
            # Load regular image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image / image.max() * 255).astype(np.uint8)

        return image

    def _load_annotations(self, image_id, image_shape):
        """Load annotations for an image."""
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        h, w = image_shape

        # Initialize arrays
        boxes = []
        labels = []
        masks = []
        semantic_mask = np.zeros((h, w), dtype=np.int64)
        crown_map = np.zeros((h, w), dtype=np.float32)

        for ann in annotations:
            # Get category
            cat_id = ann["category_id"]
            cat_info = self.coco.cats[cat_id]
            cat_name = cat_info["name"]

            # Instance segmentation
            if cat_name in self.instance_classes:
                label = self.instance_classes[cat_name]
                labels.append(label)

                # Bounding box
                x, y, w_box, h_box = ann["bbox"]
                boxes.append([x, y, x + w_box, y + h_box])

                # Mask
                if "segmentation" in ann:
                    mask = self.coco.annToMask(ann)
                    masks.append(mask)

                    # Update semantic mask
                    semantic_mask[mask > 0] = label

                    # Estimate crown diameter
                    if "crown_diameter" in ann:
                        crown_diameter = ann["crown_diameter"]
                    else:
                        # Estimate from mask
                        crown_diameter = np.sqrt(mask.sum() * 4 / np.pi)

                    crown_map[mask > 0] = crown_diameter

            # Semantic segmentation classes
            elif cat_name in self.semantic_classes:
                label = self.semantic_classes[cat_name]
                if "segmentation" in ann:
                    mask = self.coco.annToMask(ann)
                    semantic_mask[mask > 0] = label

        # Convert to arrays
        if boxes:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            masks = np.array(masks, dtype=np.uint8)
        else:
            # Empty annotations
            boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
            labels = np.array([0], dtype=np.int64)
            masks = np.zeros((1, h, w), dtype=np.uint8)

        return {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "semantic_mask": semantic_mask,
            "crown_map": crown_map,
        }

    def create_tiles(self, image_path: str) -> List[Dict]:
        """
        Create tiles from large image for processing.

        Args:
            image_path: Path to large image

        Returns:
            List of tile dictionaries
        """
        tiles = []

        with rasterio.open(image_path) as src:
            height, width = src.shape

            # Calculate tile positions
            for y in range(0, height - self.overlap, self.tile_size - self.overlap):
                for x in range(0, width - self.overlap, self.tile_size - self.overlap):
                    # Define window
                    window = Window(
                        col_off=x,
                        row_off=y,
                        width=min(self.tile_size, width - x),
                        height=min(self.tile_size, height - y),
                    )

                    # Read tile
                    tile_data = src.read(window=window)

                    # Get transform for georeferencing
                    tile_transform = src.window_transform(window)

                    tiles.append(
                        {
                            "data": tile_data,
                            "window": window,
                            "transform": tile_transform,
                            "bounds": box(x, y, x + window.width, y + window.height),
                        }
                    )

        return tiles


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = CabrucaDataset(
        data_dir=config["train_dir"],
        annotation_file=config.get("train_annotations"),
        mode="train",
        use_augmentation=True,
        tile_size=config.get("tile_size", 512),
    )

    val_dataset = CabrucaDataset(
        data_dir=config["val_dir"],
        annotation_file=config.get("val_annotations"),
        mode="val",
        use_augmentation=False,
        tile_size=config.get("tile_size", 512),
    )

    test_dataset = CabrucaDataset(
        data_dir=config["test_dir"],
        annotation_file=config.get("test_annotations"),
        mode="test",
        use_augmentation=False,
        tile_size=config.get("tile_size", 512),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """Custom collate function for batching."""
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    images = torch.stack(images, dim=0)

    return images, targets

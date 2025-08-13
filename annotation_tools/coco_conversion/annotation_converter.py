"""
Annotation Format Converter
Converts annotations from various formats (LabelMe, custom JSON, etc.) to COCO format
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COCOConverter:
    """Convert various annotation formats to COCO format"""

    def __init__(self):
        self.coco_dataset = {
            "info": {
                "description": "Cacao Tree Detection Dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Cabruca Segmentation Project",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "background", "supercategory": "none"},
                {"id": 1, "name": "cacao", "supercategory": "tree"},
                {"id": 2, "name": "shade_tree", "supercategory": "tree"},
            ],
        }
        self.annotation_id_counter = 1
        self.image_id_counter = 1
        self.image_id_map = {}

    def generate_image_id(self, image_path: str) -> int:
        """Generate consistent image ID from path"""
        if image_path not in self.image_id_map:
            self.image_id_map[image_path] = self.image_id_counter
            self.image_id_counter += 1
        return self.image_id_map[image_path]

    def add_image(self, image_path: str, width: int, height: int) -> int:
        """Add image to COCO dataset"""
        image_id = self.generate_image_id(image_path)

        # Check if image already exists
        existing = [img for img in self.coco_dataset["images"] if img["id"] == image_id]
        if not existing:
            image_info = {
                "id": image_id,
                "file_name": Path(image_path).name,
                "width": width,
                "height": height,
                "date_captured": datetime.now().isoformat(),
                "license": 0,
                "coco_url": "",
                "flickr_url": "",
            }
            self.coco_dataset["images"].append(image_info)

        return image_id

    def convert_labelme_to_coco(self, labelme_json_path: str) -> None:
        """Convert LabelMe format annotations to COCO format"""
        with open(labelme_json_path, "r") as f:
            labelme_data = json.load(f)

        # Get image info
        image_path = Path(labelme_json_path).parent / labelme_data["imagePath"]

        # Get image dimensions
        if "imageHeight" in labelme_data and "imageWidth" in labelme_data:
            height = labelme_data["imageHeight"]
            width = labelme_data["imageWidth"]
        else:
            # Load image to get dimensions
            img = Image.open(image_path)
            width, height = img.size

        # Add image to dataset
        image_id = self.add_image(str(image_path), width, height)

        # Convert shapes to annotations
        for shape in labelme_data.get("shapes", []):
            label = shape["label"]
            points = shape["points"]
            shape_type = shape.get("shape_type", "polygon")

            # Map label to category ID
            category_map = {
                "background": 0,
                "cacao": 1,
                "cacao_tree": 1,
                "shade_tree": 2,
                "shade": 2,
            }
            category_id = category_map.get(label.lower(), 0)

            if shape_type == "polygon":
                # Flatten points for segmentation
                segmentation = [coord for point in points for coord in point]

                # Calculate bounding box
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                # Calculate area
                area = self.calculate_polygon_area(points)

                annotation = {
                    "id": self.annotation_id_counter,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }

                self.coco_dataset["annotations"].append(annotation)
                self.annotation_id_counter += 1

            elif shape_type == "rectangle":
                # Convert rectangle to polygon
                x1, y1 = points[0]
                x2, y2 = points[1]
                segmentation = [x1, y1, x2, y1, x2, y2, x1, y2]
                bbox = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
                area = abs(x2 - x1) * abs(y2 - y1)

                annotation = {
                    "id": self.annotation_id_counter,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }

                self.coco_dataset["annotations"].append(annotation)
                self.annotation_id_counter += 1

    def convert_custom_json_to_coco(self, json_path: str) -> None:
        """Convert custom JSON format (from Streamlit app or SAM) to COCO"""
        with open(json_path, "r") as f:
            data = json.load(f)

        # Get image info
        image_path = data.get("image_path", "")
        width = data.get("width", 0)
        height = data.get("height", 0)

        # If dimensions not provided, try to load image
        if width == 0 or height == 0:
            try:
                img = Image.open(image_path)
                width, height = img.size
            except:
                logger.warning(f"Could not load image {image_path} to get dimensions")
                return

        # Add image to dataset
        image_id = self.add_image(image_path, width, height)

        # Convert annotations
        annotations = data.get("annotations", [])
        for ann in annotations:
            # Handle different formats
            if "category_id" in ann:
                # Already in COCO-like format
                annotation = {
                    "id": self.annotation_id_counter,
                    "image_id": image_id,
                    "category_id": ann["category_id"],
                    "segmentation": ann.get("segmentation", []),
                    "area": ann.get("area", 0),
                    "bbox": ann.get("bbox", [0, 0, 0, 0]),
                    "iscrowd": ann.get("iscrowd", 0),
                }
            elif "class_id" in ann:
                # Custom format with class_id
                points = ann.get("points", [])
                if points:
                    # Calculate bounding box
                    if isinstance(points[0], (list, tuple)):
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                    else:
                        # Flat list of coordinates
                        x_coords = points[::2]
                        y_coords = points[1::2]

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                    # Flatten points for segmentation
                    if isinstance(points[0], (list, tuple)):
                        segmentation = [coord for point in points for coord in point]
                    else:
                        segmentation = points

                    area = ann.get("area", (x_max - x_min) * (y_max - y_min))

                    annotation = {
                        "id": self.annotation_id_counter,
                        "image_id": image_id,
                        "category_id": ann["class_id"],
                        "segmentation": [segmentation],
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0,
                    }
                else:
                    continue
            else:
                logger.warning(f"Unknown annotation format in {json_path}")
                continue

            self.coco_dataset["annotations"].append(annotation)
            self.annotation_id_counter += 1

    def calculate_polygon_area(self, points: List[Tuple[float, float]]) -> float:
        """Calculate area of polygon using shoelace formula"""
        n = len(points)
        if n < 3:
            return 0

        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]

        return abs(area) / 2.0

    def batch_convert(self, annotation_dir: str, format_type: str = "auto") -> None:
        """
        Batch convert all annotations in a directory

        Args:
            annotation_dir: Directory containing annotation files
            format_type: "labelme", "custom", or "auto" (auto-detect)
        """
        annotation_dir = Path(annotation_dir)
        json_files = list(annotation_dir.glob("*.json"))

        logger.info(f"Found {len(json_files)} annotation files")

        for json_file in tqdm(json_files, desc="Converting annotations"):
            try:
                if format_type == "auto":
                    # Try to detect format
                    with open(json_file, "r") as f:
                        data = json.load(f)

                    if "shapes" in data and "imagePath" in data:
                        # LabelMe format
                        self.convert_labelme_to_coco(str(json_file))
                    elif "annotations" in data or "class_id" in data:
                        # Custom format
                        self.convert_custom_json_to_coco(str(json_file))
                    else:
                        logger.warning(f"Unknown format for {json_file}")

                elif format_type == "labelme":
                    self.convert_labelme_to_coco(str(json_file))

                elif format_type == "custom":
                    self.convert_custom_json_to_coco(str(json_file))

            except Exception as e:
                logger.error(f"Error converting {json_file}: {e}")
                continue

    def save_coco_dataset(self, output_path: str) -> None:
        """Save COCO dataset to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.coco_dataset, f, indent=2)

        logger.info(f"Saved COCO dataset to {output_path}")
        logger.info(
            f"Dataset contains {len(self.coco_dataset['images'])} images and "
            f"{len(self.coco_dataset['annotations'])} annotations"
        )

    def validate_coco_dataset(self) -> Dict[str, Any]:
        """Validate COCO dataset structure and return statistics"""
        stats = {
            "num_images": len(self.coco_dataset["images"]),
            "num_annotations": len(self.coco_dataset["annotations"]),
            "num_categories": len(self.coco_dataset["categories"]),
            "annotations_per_category": {},
            "images_per_category": {},
            "avg_annotations_per_image": 0,
            "images_without_annotations": 0,
        }

        # Count annotations per category
        for cat in self.coco_dataset["categories"]:
            cat_id = cat["id"]
            cat_name = cat["name"]
            ann_count = sum(
                1
                for ann in self.coco_dataset["annotations"]
                if ann["category_id"] == cat_id
            )
            stats["annotations_per_category"][cat_name] = ann_count

            # Count images with this category
            image_ids = set(
                ann["image_id"]
                for ann in self.coco_dataset["annotations"]
                if ann["category_id"] == cat_id
            )
            stats["images_per_category"][cat_name] = len(image_ids)

        # Calculate average annotations per image
        if stats["num_images"] > 0:
            stats["avg_annotations_per_image"] = (
                stats["num_annotations"] / stats["num_images"]
            )

        # Find images without annotations
        annotated_images = set(
            ann["image_id"] for ann in self.coco_dataset["annotations"]
        )
        all_images = set(img["id"] for img in self.coco_dataset["images"])
        stats["images_without_annotations"] = len(all_images - annotated_images)

        return stats

    def merge_coco_datasets(self, dataset_paths: List[str]) -> None:
        """Merge multiple COCO datasets into one"""
        for dataset_path in dataset_paths:
            with open(dataset_path, "r") as f:
                dataset = json.load(f)

            # Merge images
            for image in dataset.get("images", []):
                # Generate new ID to avoid conflicts
                old_id = image["id"]
                new_id = self.generate_image_id(image["file_name"])
                image["id"] = new_id

                # Update annotations with new image ID
                for ann in dataset.get("annotations", []):
                    if ann["image_id"] == old_id:
                        ann["image_id"] = new_id
                        ann["id"] = self.annotation_id_counter
                        self.annotation_id_counter += 1
                        self.coco_dataset["annotations"].append(ann)

                self.coco_dataset["images"].append(image)


def main():
    """Example usage of COCO converter"""

    # Initialize converter
    converter = COCOConverter()

    # Example: Convert LabelMe annotations
    # converter.batch_convert("data/annotations/labelme", format_type="labelme")

    # Example: Convert custom annotations
    # converter.batch_convert("data/annotations/streamlit", format_type="custom")
    # converter.batch_convert("data/annotations/sam", format_type="custom")

    # Example: Auto-detect and convert all
    # converter.batch_convert("data/annotations", format_type="auto")

    # Validate dataset
    # stats = converter.validate_coco_dataset()
    # print("Dataset Statistics:")
    # for key, value in stats.items():
    #     print(f"  {key}: {value}")

    # Save COCO dataset
    # converter.save_coco_dataset("data/processed/annotations_coco.json")

    logger.info("COCO conversion setup complete")


if __name__ == "__main__":
    main()

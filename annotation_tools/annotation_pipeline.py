"""
Main Annotation and Data Preparation Pipeline
Orchestrates the entire workflow from annotation to COCO dataset creation
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from annotation_tools.coco_conversion.annotation_converter import COCOConverter
from annotation_tools.data_splitting.dataset_splitter import DatasetSplitter
from annotation_tools.sam_annotation.sam_auto_annotator import (
    AnnotationConfig,
    SAMAutoAnnotator,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AnnotationPipeline:
    """Main pipeline for annotation and data preparation"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.annotation_dir = self.data_dir / "annotations"
        self.processed_dir = self.data_dir / "processed"

        # Create directories if they don't exist
        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized pipeline with project root: {self.project_root}")

    def run_sam_annotation(
        self, image_dir: str, output_dir: Optional[str] = None
    ) -> str:
        """
        Run SAM-based automatic annotation on images

        Args:
            image_dir: Directory containing images to annotate
            output_dir: Output directory for annotations (optional)

        Returns:
            Path to output directory
        """
        if output_dir is None:
            output_dir = self.annotation_dir / "sam"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting SAM-based annotation...")

        # Configure SAM
        config = AnnotationConfig(
            model_type="vit_h",
            checkpoint_path=str(self.project_root / "models/sam/sam_vit_h_4b8939.pth"),
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=100,
            cacao_min_area=50,
            cacao_max_area=500,
            shade_tree_min_area=500,
            shade_tree_max_area=5000,
        )

        # Check if checkpoint exists
        if not Path(config.checkpoint_path).exists():
            logger.warning(f"SAM checkpoint not found at {config.checkpoint_path}")
            logger.info("Please download the SAM model checkpoint first")
            return str(output_dir)

        # Initialize annotator
        annotator = SAMAutoAnnotator(config)

        # Run batch annotation
        annotations = annotator.batch_annotate(image_dir, str(output_dir))

        logger.info(
            f"Completed SAM annotation. Generated {len(annotations)} annotation files"
        )

        return str(output_dir)

    def convert_to_coco(
        self,
        annotation_dirs: list,
        output_path: Optional[str] = None,
        format_types: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Convert annotations from various formats to COCO format

        Args:
            annotation_dirs: List of directories containing annotations
            output_path: Path to save COCO dataset (optional)
            format_types: Dictionary mapping directory to format type (optional)

        Returns:
            Path to COCO dataset file
        """
        if output_path is None:
            output_path = self.processed_dir / "annotations_coco.json"
        else:
            output_path = Path(output_path)

        logger.info("Converting annotations to COCO format...")

        converter = COCOConverter()

        for ann_dir in annotation_dirs:
            ann_dir = Path(ann_dir)
            if not ann_dir.exists():
                logger.warning(f"Annotation directory not found: {ann_dir}")
                continue

            # Determine format type
            if format_types and str(ann_dir) in format_types:
                format_type = format_types[str(ann_dir)]
            else:
                # Auto-detect based on directory name
                if "labelme" in ann_dir.name.lower():
                    format_type = "labelme"
                elif "sam" in ann_dir.name.lower():
                    format_type = "custom"
                elif "streamlit" in ann_dir.name.lower():
                    format_type = "custom"
                else:
                    format_type = "auto"

            logger.info(
                f"Converting annotations from {ann_dir} (format: {format_type})"
            )
            converter.batch_convert(str(ann_dir), format_type=format_type)

        # Validate dataset
        stats = converter.validate_coco_dataset()
        logger.info(f"Dataset statistics: {stats}")

        # Save COCO dataset
        converter.save_coco_dataset(str(output_path))

        return str(output_path)

    def split_dataset(
        self,
        coco_path: str,
        output_dir: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        balance_strategy: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Split COCO dataset into train/val/test sets

        Args:
            coco_path: Path to COCO dataset JSON
            output_dir: Directory to save splits (optional)
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            balance_strategy: Class balancing strategy (optional)

        Returns:
            Dictionary with paths to split datasets
        """
        if output_dir is None:
            output_dir = self.processed_dir / "splits"
        else:
            output_dir = Path(output_dir)

        logger.info("Splitting dataset...")

        splitter = DatasetSplitter(coco_path)

        # Analyze class distribution
        stats = splitter.analyze_class_distribution()
        logger.info(f"Original class distribution: {stats['class_counts']}")
        logger.info(f"Class imbalance ratio: {stats['imbalance_ratio']:.2f}")

        # Create visualization
        viz_path = output_dir / "class_distribution.png"
        splitter.visualize_class_distribution(str(viz_path))

        # Create splits
        output_paths = splitter.create_split_datasets(
            output_dir=str(output_dir),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            balance_strategy=balance_strategy,
        )

        # Create balanced subset for initial training
        balanced_path = output_dir / "balanced_subset.json"
        splitter.create_balanced_subset(
            max_samples_per_class=100, output_path=str(balanced_path)
        )
        output_paths["balanced_subset"] = str(balanced_path)

        logger.info(f"Dataset splits created: {list(output_paths.keys())}")

        return output_paths

    def run_full_pipeline(
        self, image_dir: str, balance_strategy: str = "mixed"
    ) -> Dict:
        """
        Run the complete annotation and data preparation pipeline

        Args:
            image_dir: Directory containing raw images
            balance_strategy: Strategy for class balancing

        Returns:
            Dictionary with all output paths
        """
        logger.info("=" * 50)
        logger.info("Starting Full Annotation Pipeline")
        logger.info("=" * 50)

        results = {}

        # Step 1: Run SAM annotation (if needed)
        sam_output = self.annotation_dir / "sam"
        if not sam_output.exists() or not any(sam_output.glob("*.json")):
            logger.info("Step 1: Running SAM-based annotation...")
            results["sam_annotations"] = self.run_sam_annotation(image_dir)
        else:
            logger.info("Step 1: SAM annotations already exist, skipping...")
            results["sam_annotations"] = str(sam_output)

        # Step 2: Convert to COCO format
        logger.info("Step 2: Converting annotations to COCO format...")

        # Collect all annotation directories
        annotation_dirs = []
        for subdir in self.annotation_dir.iterdir():
            if subdir.is_dir() and any(subdir.glob("*.json")):
                annotation_dirs.append(str(subdir))

        if annotation_dirs:
            results["coco_dataset"] = self.convert_to_coco(annotation_dirs)
        else:
            logger.warning("No annotation directories found")
            return results

        # Step 3: Split dataset
        logger.info("Step 3: Splitting dataset...")
        results["splits"] = self.split_dataset(
            results["coco_dataset"], balance_strategy=balance_strategy
        )

        # Step 4: Generate summary report
        logger.info("Step 4: Generating summary report...")
        report_path = self.processed_dir / "pipeline_report.json"

        report = {
            "pipeline_results": results,
            "configuration": {
                "balance_strategy": balance_strategy,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
            },
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        results["report"] = str(report_path)

        logger.info("=" * 50)
        logger.info("Pipeline Complete!")
        logger.info("=" * 50)
        logger.info(f"Results saved to: {report_path}")

        return results


def main():
    """Main entry point for the annotation pipeline"""

    parser = argparse.ArgumentParser(
        description="Annotation and Data Preparation Pipeline"
    )
    parser.add_argument(
        "--image-dir", type=str, help="Directory containing images to annotate"
    )
    parser.add_argument(
        "--annotation-dirs", nargs="+", help="Directories with existing annotations"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for processed data"
    )
    parser.add_argument(
        "--balance-strategy",
        type=str,
        default="mixed",
        choices=["oversample", "undersample", "mixed", "none"],
        help="Class balancing strategy",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation set ratio"
    )
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--run-sam", action="store_true", help="Run SAM annotation")
    parser.add_argument(
        "--convert-only", action="store_true", help="Only convert annotations to COCO"
    )
    parser.add_argument(
        "--split-only", action="store_true", help="Only split existing COCO dataset"
    )
    parser.add_argument("--coco-path", type=str, help="Path to existing COCO dataset")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = AnnotationPipeline()

    if args.run_sam and args.image_dir:
        # Run SAM annotation only
        output = pipeline.run_sam_annotation(args.image_dir)
        print(f"SAM annotations saved to: {output}")

    elif args.convert_only and args.annotation_dirs:
        # Convert annotations only
        output = pipeline.convert_to_coco(args.annotation_dirs)
        print(f"COCO dataset saved to: {output}")

    elif args.split_only and args.coco_path:
        # Split dataset only
        outputs = pipeline.split_dataset(
            args.coco_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            balance_strategy=(
                args.balance_strategy if args.balance_strategy != "none" else None
            ),
        )
        print(f"Dataset splits saved to: {outputs}")

    elif args.image_dir:
        # Run full pipeline
        results = pipeline.run_full_pipeline(
            args.image_dir,
            balance_strategy=(
                args.balance_strategy if args.balance_strategy != "none" else None
            ),
        )
        print(f"Pipeline complete. Results: {results}")

    else:
        print("Please provide required arguments. Use --help for more information.")

        # Show example commands
        print("\nExample commands:")
        print("  # Run full pipeline:")
        print("  python annotation_pipeline.py --image-dir data/raw")
        print("\n  # Run SAM annotation only:")
        print("  python annotation_pipeline.py --run-sam --image-dir data/raw")
        print("\n  # Convert annotations to COCO:")
        print(
            "  python annotation_pipeline.py --convert-only --annotation-dirs data/annotations/sam data/annotations/labelme"
        )
        print("\n  # Split existing COCO dataset:")
        print(
            "  python annotation_pipeline.py --split-only --coco-path data/processed/annotations_coco.json"
        )


if __name__ == "__main__":
    main()

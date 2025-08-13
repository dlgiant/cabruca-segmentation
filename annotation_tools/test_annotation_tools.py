"""
Test script for annotation tools
Verifies that all components are properly installed and configured
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing imports...")

    try:
        # Test annotation tools imports
        from coco_conversion.annotation_converter import COCOConverter

        logger.info("✓ COCO converter imported successfully")

        from data_splitting.dataset_splitter import DatasetSplitter

        logger.info("✓ Dataset splitter imported successfully")

        # Test dependencies
        import streamlit

        logger.info("✓ Streamlit imported successfully")

        import plotly

        logger.info("✓ Plotly imported successfully")

        import cv2

        logger.info("✓ OpenCV imported successfully")

        from sklearn.model_selection import train_test_split

        logger.info("✓ Scikit-learn imported successfully")

        from imblearn.over_sampling import RandomOverSampler

        logger.info("✓ Imbalanced-learn imported successfully")

        import tqdm

        logger.info("✓ TQDM imported successfully")

        return True

    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_coco_converter():
    """Test COCO converter functionality"""
    logger.info("\nTesting COCO converter...")

    try:
        from coco_conversion.annotation_converter import COCOConverter

        # Create converter
        converter = COCOConverter()

        # Test adding an image
        image_id = converter.add_image("test_image.jpg", 1024, 768)
        assert image_id == 1, "Image ID should be 1"

        # Test dataset structure
        assert "images" in converter.coco_dataset
        assert "annotations" in converter.coco_dataset
        assert "categories" in converter.coco_dataset
        assert len(converter.coco_dataset["categories"]) == 3

        logger.info("✓ COCO converter test passed")
        return True

    except Exception as e:
        logger.error(f"✗ COCO converter test failed: {e}")
        return False


def test_dataset_splitter():
    """Test dataset splitter functionality"""
    logger.info("\nTesting dataset splitter...")

    try:
        from data_splitting.dataset_splitter import DatasetSplitter

        # Create temporary COCO dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_data = {
                "images": [
                    {
                        "id": i,
                        "file_name": f"image_{i}.jpg",
                        "width": 640,
                        "height": 480,
                    }
                    for i in range(1, 11)
                ],
                "annotations": [
                    {
                        "id": i,
                        "image_id": (i % 10) + 1,
                        "category_id": i % 3,
                        "bbox": [10, 10, 50, 50],
                        "area": 2500,
                        "segmentation": [[10, 10, 60, 10, 60, 60, 10, 60]],
                        "iscrowd": 0,
                    }
                    for i in range(1, 31)
                ],
                "categories": [
                    {"id": 0, "name": "background"},
                    {"id": 1, "name": "cacao"},
                    {"id": 2, "name": "shade_tree"},
                ],
            }
            json.dump(test_data, f)
            temp_path = f.name

        # Test splitter
        splitter = DatasetSplitter(temp_path)

        # Test class distribution analysis
        stats = splitter.analyze_class_distribution()
        assert "class_counts" in stats
        assert "imbalance_ratio" in stats

        # Test stratified split
        train, val, test = splitter.stratified_split(0.6, 0.2, 0.2)
        assert len(train) == 6
        assert len(val) == 2
        assert len(test) == 2

        # Clean up
        Path(temp_path).unlink()

        logger.info("✓ Dataset splitter test passed")
        return True

    except Exception as e:
        logger.error(f"✗ Dataset splitter test failed: {e}")
        return False


def test_sam_annotator():
    """Test SAM annotator setup (without running actual annotation)"""
    logger.info("\nTesting SAM annotator setup...")

    try:
        from sam_annotation.sam_auto_annotator import AnnotationConfig

        # Test configuration
        config = AnnotationConfig(
            model_type="vit_h",
            checkpoint_path="models/sam/sam_vit_h_4b8939.pth",
            cacao_min_area=50,
            cacao_max_area=500,
        )

        assert config.model_type == "vit_h"
        assert config.cacao_min_area == 50
        assert config.cacao_max_area == 500

        logger.info("✓ SAM annotator configuration test passed")

        # Check if SAM can be imported
        try:
            from segment_anything import sam_model_registry

            logger.info("✓ SAM (Segment Anything) imported successfully")
        except ImportError:
            logger.warning(
                "⚠ SAM not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        return True

    except Exception as e:
        logger.error(f"✗ SAM annotator test failed: {e}")
        return False


def test_pipeline():
    """Test the main annotation pipeline"""
    logger.info("\nTesting annotation pipeline...")

    try:
        from annotation_pipeline import AnnotationPipeline

        # Create pipeline
        pipeline = AnnotationPipeline()

        # Test directory creation
        assert pipeline.data_dir.exists()
        assert pipeline.annotation_dir.exists()
        assert pipeline.processed_dir.exists()

        logger.info("✓ Annotation pipeline test passed")
        return True

    except Exception as e:
        logger.error(f"✗ Annotation pipeline test failed: {e}")
        return False


def create_test_image():
    """Create a test image for annotation"""
    logger.info("\nCreating test image...")

    try:
        # Create test image directory
        test_dir = Path("data/test_images")
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create a simple test image
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Add some "tree-like" features
        # Green circles for cacao trees
        for _ in range(5):
            x, y = np.random.randint(50, 450, 2)
            cv2.circle(img_array, (x, y), 20, (0, 255, 0), -1)

        # Red circles for shade trees
        for _ in range(3):
            x, y = np.random.randint(50, 450, 2)
            cv2.circle(img_array, (x, y), 40, (255, 0, 0), -1)

        # Save image
        img = Image.fromarray(img_array)
        img_path = test_dir / "test_annotation_image.jpg"
        img.save(img_path)

        logger.info(f"✓ Test image created at: {img_path}")
        return str(img_path)

    except Exception as e:
        logger.error(f"✗ Failed to create test image: {e}")
        return None


def main():
    """Run all tests"""
    logger.info("=" * 50)
    logger.info("Annotation Tools Test Suite")
    logger.info("=" * 50)

    results = []

    # Run tests
    results.append(("Import Test", test_imports()))
    results.append(("COCO Converter", test_coco_converter()))
    results.append(("Dataset Splitter", test_dataset_splitter()))
    results.append(("SAM Annotator", test_sam_annotator()))
    results.append(("Pipeline", test_pipeline()))

    # Create test image
    test_image = create_test_image()
    if test_image:
        results.append(("Test Image Creation", True))
    else:
        results.append(("Test Image Creation", False))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Test Summary")
    logger.info("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 All tests passed! Annotation tools are ready to use.")
        logger.info("\nNext steps:")
        logger.info(
            "1. Launch Streamlit app: streamlit run annotation_tools/streamlit_app/annotation_app.py"
        )
        logger.info(
            "2. Run annotation pipeline: python annotation_tools/annotation_pipeline.py --help"
        )
        logger.info("3. Place your images in data/raw/ directory")
    else:
        logger.warning("\n⚠ Some tests failed. Please check the error messages above.")
        logger.info("\nCommon fixes:")
        logger.info("1. Install missing dependencies: pip install -r requirements.txt")
        logger.info(
            "2. Install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )
        logger.info("3. Download SAM checkpoint if using SAM annotation")

    return passed == total


if __name__ == "__main__":
    import cv2  # Import here to test if OpenCV is available

    success = main()
    sys.exit(0 if success else 1)

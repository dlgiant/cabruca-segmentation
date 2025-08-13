#!/usr/bin/env python3
"""
Setup script for Segment Anything Model (SAM) with tree segmentation capabilities.
Downloads model checkpoints and configures for macOS Metal optimization.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def check_dependencies():
    """Check and install required dependencies."""
    print("Checking dependencies...")

    required_packages = [
        "torch",
        "torchvision",
        "opencv-python",
        "matplotlib",
        "scipy",
        "segment-anything",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == "segment-anything":
                __import__("segment_anything")
            elif package == "opencv-python":
                __import__("cv2")
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        install = input("Do you want to install missing packages? (y/n): ")

        if install.lower() == "y":
            for package in missing_packages:
                if package == "segment-anything":
                    print("Installing segment-anything from GitHub...")
                    subprocess.check_call(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "git+https://github.com/facebookresearch/segment-anything.git",
                        ]
                    )
                else:
                    print(f"Installing {package}...")
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", package]
                    )
        else:
            print("Please install missing packages manually.")
            sys.exit(1)

    print("✅ All dependencies are installed.")


def download_sam_checkpoint(model_type="vit_b", force_download=False):
    """
    Download SAM model checkpoint.

    Args:
        model_type: Type of model ('vit_b', 'vit_l', or 'vit_h')
        force_download: Force re-download even if file exists

    Returns:
        Path to downloaded checkpoint
    """
    urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }

    if model_type not in urls:
        raise ValueError(f"Invalid model type: {model_type}")

    # Create checkpoint directory
    checkpoint_dir = Path("models/sam_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get checkpoint URL and filename
    url = urls[model_type]
    filename = url.split("/")[-1]
    checkpoint_path = checkpoint_dir / filename

    # Download if needed
    if force_download or not checkpoint_path.exists():
        print(f"Downloading SAM {model_type} checkpoint...")
        print(f"URL: {url}")
        print(f"Destination: {checkpoint_path}")

        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            mb_downloaded = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            sys.stdout.write(
                f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
            )
            sys.stdout.flush()

        urllib.request.urlretrieve(url, checkpoint_path, download_progress)
        print(f"\n✅ Checkpoint downloaded successfully!")
    else:
        print(f"✅ Checkpoint already exists at {checkpoint_path}")

    return checkpoint_path


def test_metal_support():
    """Test Metal Performance Shaders support."""
    print("\nTesting Metal Performance Shaders support...")

    try:
        import torch

        if torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                print("✅ Metal Performance Shaders (MPS) is available!")

                # Test with a simple operation
                device = torch.device("mps")
                x = torch.randn(100, 100).to(device)
                y = torch.randn(100, 100).to(device)
                z = torch.matmul(x, y)

                print("✅ MPS test successful - matrix multiplication works!")
                return True
            else:
                print(
                    "⚠️ MPS is available but not built. PyTorch needs to be built with MPS support."
                )
        else:
            print("⚠️ MPS is not available on this system.")
    except Exception as e:
        print(f"❌ Error testing MPS: {e}")

    return False


def create_config_file(model_type="vit_b", checkpoint_path=None):
    """Create configuration file for SAM."""
    config = {
        "model": {
            "type": model_type,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "device": "auto",  # Will auto-detect best device
        },
        "tree_detection": {
            "num_classes": 2,
            "class_names": ["cacao", "shade"],
            "confidence_threshold": 0.7,
            "iou_threshold": 0.5,
            "min_tree_area": 100,
            "max_tree_area": 50000,
        },
        "prompt_engineering": {
            "points_per_side": 32,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "crop_n_layers": 0,
            "min_mask_region_area": 100,
        },
        "optimization": {
            "use_metal": "auto",
            "enable_mixed_precision": True,
            "batch_size": 1,
        },
    }

    # Save config
    config_path = Path("configs/sam_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Configuration saved to {config_path}")
    return config_path


def test_sam_model(checkpoint_path, model_type="vit_b"):
    """Test SAM model loading and basic functionality."""
    print("\nTesting SAM model...")

    try:
        # Import our custom SAM module
        from src.models.sam_model import SAMConfig, SAMTreeSegmenter

        # Create config
        config = SAMConfig(model_type=model_type, checkpoint_path=str(checkpoint_path))

        # Initialize segmenter
        segmenter = SAMTreeSegmenter(config)

        # Load model
        print("Loading model...")
        segmenter.load_model(checkpoint_path=str(checkpoint_path))

        print("✅ SAM model loaded successfully!")

        # Create a dummy image for testing
        import numpy as np

        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Test point prompt generation
        points = segmenter.prompt_engineer.generate_point_prompts(dummy_image)
        print(f"✅ Generated {len(points)} point prompts")

        # Test box prompt generation
        boxes = segmenter.prompt_engineer.generate_box_prompts(dummy_image)
        print(f"✅ Generated {len(boxes)} box prompts")

        return True

    except Exception as e:
        print(f"❌ Error testing SAM model: {e}")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup SAM for tree segmentation")
    parser.add_argument(
        "--model",
        choices=["vit_b", "vit_l", "vit_h"],
        default="vit_b",
        help="SAM model variant to download",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of model checkpoint",
    )
    parser.add_argument("--skip-test", action="store_true", help="Skip model testing")

    args = parser.parse_args()

    print("=" * 60)
    print("SAM Setup for Tree Segmentation")
    print("=" * 60)

    # Check dependencies
    check_dependencies()

    # Test Metal support
    has_metal = test_metal_support()

    # Download checkpoint
    checkpoint_path = download_sam_checkpoint(
        model_type=args.model, force_download=args.force_download
    )

    # Create config file
    config_path = create_config_file(
        model_type=args.model, checkpoint_path=checkpoint_path
    )

    # Test model
    if not args.skip_test:
        success = test_sam_model(checkpoint_path, args.model)
        if not success:
            print("\n⚠️ Model test failed, but setup is complete.")
            print("You may need to debug the model loading.")

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\n✅ Model checkpoint: {checkpoint_path}")
    print(f"✅ Configuration file: {config_path}")
    if has_metal:
        print("✅ Metal Performance Shaders: Enabled")
    else:
        print("⚠️ Metal Performance Shaders: Not available (using CPU/CUDA)")

    print("\nNext steps:")
    print("1. Run example segmentation: python run_sam_segmentation.py")
    print("2. Fine-tune on your data: python train_sam.py")
    print("3. Evaluate performance: python evaluate_sam.py")


if __name__ == "__main__":
    main()

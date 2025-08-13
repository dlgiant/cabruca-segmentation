#!/usr/bin/env python3
"""
Quick test script to verify SAM setup and Metal optimization.
"""

import sys
from pathlib import Path

import numpy as np
import torch


def test_metal_support():
    """Test if Metal Performance Shaders are available."""
    print("=" * 60)
    print("Testing Metal Performance Shaders Support")
    print("=" * 60)

    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("✅ Metal Performance Shaders (MPS) is available!")

            # Test basic operations
            try:
                device = torch.device("mps")
                x = torch.randn(1000, 1000).to(device)
                y = torch.randn(1000, 1000).to(device)
                z = torch.matmul(x, y)
                print("✅ MPS matrix multiplication test passed!")

                # Test memory
                print(f"✅ MPS device: {device}")
                return True
            except Exception as e:
                print(f"❌ MPS test failed: {e}")
                return False
        else:
            print("⚠️ MPS is available but PyTorch wasn't built with MPS support")
            return False
    else:
        print("⚠️ MPS is not available on this system")
        return False


def test_sam_import():
    """Test if SAM module can be imported."""
    print("\n" + "=" * 60)
    print("Testing SAM Module Import")
    print("=" * 60)

    try:
        from src.models.sam_model import (
            SAMConfig,
            SAMTreeSegmenter,
            TreePromptEngineering,
        )

        print("✅ SAM module imported successfully!")

        # Test configuration
        config = SAMConfig()
        print(f"✅ Default configuration created:")
        print(f"   - Model type: {config.model_type}")
        print(f"   - Device: {config.device}")
        print(f"   - Num classes: {config.num_classes}")
        print(f"   - Metal optimization: {config.use_metal_optimization}")

        return True
    except ImportError as e:
        print(f"❌ Failed to import SAM module: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing SAM module: {e}")
        return False


def test_prompt_engineering():
    """Test prompt engineering functionality."""
    print("\n" + "=" * 60)
    print("Testing Prompt Engineering")
    print("=" * 60)

    try:
        from src.models.sam_model import SAMConfig, TreePromptEngineering

        config = SAMConfig()
        prompt_engineer = TreePromptEngineering(config)

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        # Make some pixels "green" to simulate vegetation
        dummy_image[:, :, 1] = dummy_image[:, :, 1] + 50  # Boost green channel

        # Test point generation
        cacao_points = prompt_engineer.generate_point_prompts(
            dummy_image, tree_type="cacao"
        )
        shade_points = prompt_engineer.generate_point_prompts(
            dummy_image, tree_type="shade"
        )

        print(f"✅ Generated {len(cacao_points)} cacao tree point prompts")
        print(f"✅ Generated {len(shade_points)} shade tree point prompts")

        # Test box generation
        boxes = prompt_engineer.generate_box_prompts(dummy_image)
        print(f"✅ Generated {len(boxes)} bounding box prompts")

        return True
    except Exception as e:
        print(f"❌ Error testing prompt engineering: {e}")
        return False


def test_dependencies():
    """Test if all required dependencies are installed."""
    print("\n" + "=" * 60)
    print("Testing Dependencies")
    print("=" * 60)

    dependencies = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "matplotlib": "Matplotlib",
    }

    all_installed = True

    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name} is installed")
        except ImportError:
            print(f"❌ {name} is NOT installed")
            all_installed = False

    # Check for segment-anything
    try:
        import segment_anything

        print("✅ segment-anything is installed")
    except ImportError:
        print("❌ segment-anything is NOT installed")
        print(
            "   Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )
        all_installed = False

    return all_installed


def test_checkpoint_exists():
    """Test if SAM checkpoint exists."""
    print("\n" + "=" * 60)
    print("Testing Model Checkpoint")
    print("=" * 60)

    checkpoint_dir = Path("models/sam_checkpoints")

    if not checkpoint_dir.exists():
        print("⚠️ Checkpoint directory doesn't exist")
        print("   Run: python setup_sam.py --model vit_b")
        return False

    checkpoints = list(checkpoint_dir.glob("*.pth"))

    if checkpoints:
        print(f"✅ Found {len(checkpoints)} checkpoint(s):")
        for ckpt in checkpoints:
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"   - {ckpt.name} ({size_mb:.1f} MB)")
        return True
    else:
        print("⚠️ No checkpoints found")
        print("   Run: python setup_sam.py --model vit_b")
        return False


def main():
    """Run all tests."""
    print("\n" + "🌳" * 30)
    print("SAM Tree Segmentation Setup Test")
    print("🌳" * 30)

    results = {
        "Metal Support": test_metal_support(),
        "Dependencies": test_dependencies(),
        "SAM Module": test_sam_import(),
        "Prompt Engineering": test_prompt_engineering(),
        "Model Checkpoint": test_checkpoint_exists(),
    }

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All tests passed! SAM is ready for tree segmentation.")
        print("\nNext steps:")
        print("1. If no checkpoint exists, run: python setup_sam.py --model vit_b")
        print(
            "2. Test segmentation: python run_sam_segmentation.py --image <your_image.jpg>"
        )
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")
        if not results["Dependencies"]:
            print("\nInstall missing dependencies:")
            print("  pip install -r requirements.txt")
            print(
                "  pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        if not results["Model Checkpoint"]:
            print("\nDownload model checkpoint:")
            print("  python setup_sam.py --model vit_b")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

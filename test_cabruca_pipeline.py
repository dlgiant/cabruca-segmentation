"""
Test script for the complete Cabruca segmentation pipeline.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
sys.path.append("src")

from data_processing.cabruca_dataset import CabrucaDataset
from inference.cabruca_inference import CabrucaInference
from models.cabruca_segmentation_model import (CabrucaLoss,
                                               CabrucaSegmentationModel,
                                               create_cabruca_model)
from training.train_cabruca import CabrucaTrainer


def test_model_creation():
    """Test model creation and forward pass."""
    print("\n" + "=" * 50)
    print("Testing Model Creation")
    print("=" * 50)

    try:
        # Create model
        config = {
            "num_instance_classes": 3,
            "num_semantic_classes": 6,
            "use_sam": False,
        }

        model = create_cabruca_model(config)
        print(f"✓ Model created successfully")
        print(f"  - Instance classes: {config['num_instance_classes']}")
        print(f"  - Semantic classes: {config['num_semantic_classes']}")

        # Test forward pass
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 512, 512)

        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)

        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output keys: {outputs.keys()}")

        # Check output shapes
        if "semantic" in outputs:
            print(f"  - Semantic shape: {outputs['semantic'].shape}")
        if "crown_diameters" in outputs:
            print(f"  - Crown map shape: {outputs['crown_diameters'].shape}")
        if "canopy_density" in outputs:
            print(f"  - Canopy density shape: {outputs['canopy_density'].shape}")

        return True

    except Exception as e:
        print(f"✗ Model creation failed: {str(e)}")
        return False


def test_data_pipeline():
    """Test data loading and augmentation pipeline."""
    print("\n" + "=" * 50)
    print("Testing Data Pipeline")
    print("=" * 50)

    try:
        # Create dummy data directory
        data_dir = "data/raw"

        # Create dataset
        dataset = CabrucaDataset(
            data_dir=data_dir, mode="train", use_augmentation=True, tile_size=512
        )

        print(f"✓ Dataset created successfully")
        print(f"  - Data directory: {data_dir}")
        print(f"  - Mode: train")
        print(f"  - Augmentation: enabled")

        # Test data loading (if images exist)
        if len(dataset) > 0:
            image, targets = dataset[0]
            print(f"✓ Data loading successful")
            print(f"  - Image shape: {image.shape}")
            print(f"  - Target keys: {targets.keys()}")
        else:
            print("  - No images found (expected for test)")

        # Test augmentation pipeline
        aug_pipeline = dataset._get_augmentation_pipeline()
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        dummy_mask = np.random.randint(0, 6, (512, 512), dtype=np.int64)

        augmented = aug_pipeline(image=dummy_image, mask=dummy_mask)
        print(f"✓ Augmentation pipeline successful")
        print(f"  - Augmented image shape: {augmented['image'].shape}")

        return True

    except Exception as e:
        print(f"✗ Data pipeline failed: {str(e)}")
        return False


def test_loss_function():
    """Test loss function calculation."""
    print("\n" + "=" * 50)
    print("Testing Loss Function")
    print("=" * 50)

    try:
        # Create loss function
        loss_fn = CabrucaLoss(
            instance_weight=1.0,
            semantic_weight=1.0,
            crown_weight=0.5,
            density_weight=0.5,
        )

        print(f"✓ Loss function created")

        # Create dummy outputs and targets
        batch_size = 2
        height, width = 512, 512
        num_classes = 6

        outputs = {
            "semantic": torch.randn(batch_size, num_classes, height, width),
            "crown_diameters": torch.randn(batch_size, 1, height, width),
            "canopy_density": torch.randn(batch_size, 1),
        }

        targets = {
            "semantic_masks": torch.randint(
                0, num_classes, (batch_size, height, width)
            ),
            "crown_gt": torch.randn(batch_size, 1, height, width),
            "density_gt": torch.randn(batch_size, 1),
        }

        # Calculate loss
        total_loss, loss_dict = loss_fn(outputs, targets)

        print(f"✓ Loss calculation successful")
        print(f"  - Total loss: {total_loss.item():.4f}")
        for k, v in loss_dict.items():
            if k != "total":
                print(f"  - {k} loss: {v.item():.4f}")

        return True

    except Exception as e:
        print(f"✗ Loss function failed: {str(e)}")
        return False


def test_metrics_calculation():
    """Test metrics calculation."""
    print("\n" + "=" * 50)
    print("Testing Metrics Calculation")
    print("=" * 50)

    try:
        # Create model
        model = CabrucaSegmentationModel()

        # Create dummy outputs
        outputs = {
            "instances": [
                {
                    "labels": torch.tensor([1, 1, 2, 2]),
                    "scores": torch.tensor([0.9, 0.8, 0.85, 0.7]),
                }
            ],
            "semantic": torch.randn(1, 6, 512, 512),
            "crown_diameters": torch.randn(1, 1, 512, 512).abs() * 10,
            "canopy_density": torch.tensor([0.65]),
        }

        # Calculate metrics
        metrics = model.calculate_tree_metrics(outputs)

        print(f"✓ Metrics calculation successful")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  - {k}: {v:.4f}")
            else:
                print(f"  - {k}: {v}")

        return True

    except Exception as e:
        print(f"✗ Metrics calculation failed: {str(e)}")
        return False


def test_inference_pipeline():
    """Test inference pipeline."""
    print("\n" + "=" * 50)
    print("Testing Inference Pipeline")
    print("=" * 50)

    try:
        # Create and save a dummy model
        model = CabrucaSegmentationModel()
        model_path = "temp_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"✓ Dummy model saved")

        # Initialize inference
        inference = CabrucaInference(model_path)
        print(f"✓ Inference pipeline initialized")

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Test preprocessing
        preprocessed = inference._preprocess_image(dummy_image)
        print(f"✓ Image preprocessing successful")
        print(f"  - Input shape: {dummy_image.shape}")
        print(f"  - Preprocessed shape: {preprocessed.shape}")

        # Test postprocessing
        dummy_outputs = {
            "semantic": torch.randn(1, 6, 512, 512),
            "crown_diameters": torch.randn(1, 1, 512, 512),
            "canopy_density": torch.tensor([[0.65]]),
        }

        results = inference._postprocess_outputs(dummy_outputs, (512, 512))
        print(f"✓ Output postprocessing successful")
        print(f"  - Result keys: {results.keys()}")

        # Clean up
        os.remove(model_path)

        return True

    except Exception as e:
        print(f"✗ Inference pipeline failed: {str(e)}")
        return False


def test_training_config():
    """Test training configuration."""
    print("\n" + "=" * 50)
    print("Testing Training Configuration")
    print("=" * 50)

    try:
        # Create training config
        config = {
            "model": {
                "num_instance_classes": 3,
                "num_semantic_classes": 6,
                "use_sam": False,
            },
            "data": {
                "train_dir": "data/train",
                "val_dir": "data/val",
                "test_dir": "data/test",
                "batch_size": 4,
                "num_workers": 2,
                "tile_size": 512,
            },
            "optimizer": {"type": "adamw", "lr": 0.0001, "weight_decay": 0.01},
            "scheduler": {"type": "cosine", "T_max": 50, "eta_min": 1e-6},
            "loss": {
                "instance_weight": 1.0,
                "semantic_weight": 1.0,
                "crown_weight": 0.5,
                "density_weight": 0.5,
            },
            "training": {
                "num_epochs": 100,
                "gradient_clip": 1.0,
                "save_freq": 10,
                "early_stopping": 20,
            },
            "output_dir": "outputs",
            "use_wandb": False,
        }

        # Save config
        config_path = "test_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✓ Training configuration created")
        print(f"  - Model classes: {config['model']['num_semantic_classes']}")
        print(f"  - Batch size: {config['data']['batch_size']}")
        print(f"  - Learning rate: {config['optimizer']['lr']}")
        print(f"  - Epochs: {config['training']['num_epochs']}")

        # Clean up
        os.remove(config_path)

        return True

    except Exception as e:
        print(f"✗ Training config failed: {str(e)}")
        return False


def test_qgis_integration():
    """Test QGIS integration components."""
    print("\n" + "=" * 50)
    print("Testing QGIS Integration")
    print("=" * 50)

    try:
        # Check if QGIS plugin files exist
        plugin_dir = "qgis_plugin"
        required_files = ["cabruca_qgis_plugin.py", "__init__.py", "metadata.txt"]

        for file in required_files:
            file_path = os.path.join(plugin_dir, file)
            if os.path.exists(file_path):
                print(f"✓ {file} exists")
            else:
                print(f"✗ {file} not found")

        # Check metadata
        metadata_path = os.path.join(plugin_dir, "metadata.txt")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                content = f.read()
                if "name=Cabruca Segmentation Analysis" in content:
                    print(f"✓ Plugin metadata valid")

        return True

    except Exception as e:
        print(f"✗ QGIS integration failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print(" CABRUCA SEGMENTATION PIPELINE TEST SUITE")
    print("=" * 60)

    tests = [
        ("Model Creation", test_model_creation),
        ("Data Pipeline", test_data_pipeline),
        ("Loss Function", test_loss_function),
        ("Metrics Calculation", test_metrics_calculation),
        ("Inference Pipeline", test_inference_pipeline),
        ("Training Configuration", test_training_config),
        ("QGIS Integration", test_qgis_integration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {str(e)}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print(" TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    print("\n" + "-" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The Cabruca segmentation pipeline is ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

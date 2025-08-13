#!/usr/bin/env python3
"""
Test script for aerial imagery augmentation pipeline

This script demonstrates the usage of the comprehensive aerial augmentation pipeline
with various augmentation types and configurations.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing.aerial_augmentation import (
    AerialAugmentationPipeline,
    AtmosphericAugmentations,
    AugmentationConfig,
    AugmentationType,
    GeometricAugmentations,
    MultispectralAugmentations,
    RadiometricAugmentations,
    SensorAugmentations,
    TemporalAugmentations,
    create_preset_config,
)
from src.data_processing.augmentation_visualizer import AugmentationVisualizer


def create_synthetic_aerial_image(
    width: int = 512, height: int = 512, num_bands: int = 3
) -> np.ndarray:
    """
    Create a synthetic aerial image for testing

    Args:
        width: Image width
        height: Image height
        num_bands: Number of spectral bands

    Returns:
        Synthetic aerial image
    """
    image = np.zeros((height, width, num_bands), dtype=np.uint8)

    # Create base pattern
    for i in range(height):
        for j in range(width):
            # Simulate agricultural fields
            if (i // 64) % 2 == (j // 64) % 2:
                base_color = [120, 180, 100]  # Green vegetation
            else:
                base_color = [160, 140, 120]  # Soil

            # Add some variation
            noise = np.random.randint(-20, 20, size=num_bands)
            color = np.clip(np.array(base_color[:num_bands]) + noise, 0, 255)
            image[i, j] = color

    # Add some features
    # Roads
    cv2.line(image, (width // 4, 0), (width // 4, height), (128, 128, 128), 5)
    cv2.line(image, (0, height // 2), (width, height // 2), (128, 128, 128), 5)

    # Buildings/structures
    for _ in range(5):
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)
        size = np.random.randint(10, 30)
        color = tuple(np.random.randint(100, 200, size=num_bands).tolist())
        cv2.rectangle(image, (x, y), (x + size, y + size), color, -1)

    # Water body
    center = (3 * width // 4, 3 * height // 4)
    radius = min(width, height) // 8
    color = tuple([100, 150, 200][:num_bands])
    cv2.circle(image, center, radius, color, -1)

    return image


def create_synthetic_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a synthetic segmentation mask

    Args:
        image: Input image

    Returns:
        Segmentation mask
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # Segment based on simple color thresholding
    # Vegetation
    if len(image.shape) == 3:
        vegetation = image[:, :, 1] > 150  # Green channel
        mask[vegetation] = 1

        # Water
        water = (image[:, :, 2] > 180) & (image[:, :, 1] > 140)  # Blue-ish
        mask[water] = 2

        # Built-up
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        built = (gray > 120) & (gray < 140)
        mask[built] = 3

    return mask


def test_individual_augmentations(
    image: np.ndarray, mask: Optional[np.ndarray] = None
) -> None:
    """
    Test individual augmentation types

    Args:
        image: Input image
        mask: Optional segmentation mask
    """
    print("\n" + "=" * 50)
    print("Testing Individual Augmentations")
    print("=" * 50)

    visualizer = AugmentationVisualizer(figsize=(12, 8))

    # Test Geometric Augmentations
    print("\n1. Geometric Augmentations:")
    geometric = GeometricAugmentations()

    # Rotation
    rotated, rotated_mask = geometric.rotate_nadir(image, 45, mask)
    print("   - Rotation: Applied 45-degree rotation")

    # Perspective
    perspective, perspective_mask = geometric.perspective_transform(image, 0.002, mask)
    print("   - Perspective: Applied perspective transformation")

    # Orthorectification
    ortho, ortho_mask = geometric.orthorectification_distortion(image, 30, mask)
    print("   - Orthorectification: Applied terrain distortion")

    # Elastic
    elastic, elastic_mask = geometric.elastic_transform(image, 50, 5, mask)
    print("   - Elastic: Applied elastic deformation")

    # Visualize geometric augmentations
    geometric_results = [
        (rotated, "Rotation (45°)"),
        (perspective, "Perspective"),
        (ortho, "Orthorectification"),
        (elastic, "Elastic"),
    ]
    visualizer.visualize_augmentation_sequence(image, geometric_results)

    # Test Radiometric Augmentations
    print("\n2. Radiometric Augmentations:")
    radiometric = RadiometricAugmentations()

    # Brightness/Contrast
    bright = radiometric.adjust_brightness_contrast(image, 1.2, 1.1)
    print("   - Brightness/Contrast: Adjusted")

    # Gamma
    gamma = radiometric.gamma_correction(image, 0.8)
    print("   - Gamma: Applied correction (0.8)")

    # Shadows
    shadows = radiometric.simulate_shadows(image, 2)
    print("   - Shadows: Added cloud shadows")

    # Histogram
    histogram = radiometric.histogram_matching(image)
    print("   - Histogram: Equalized")

    # Visualize radiometric augmentations
    radiometric_results = [
        (bright, "Brightness/Contrast"),
        (gamma, "Gamma Correction"),
        (shadows, "Shadows"),
        (histogram, "Histogram Equalization"),
    ]
    visualizer.visualize_augmentation_sequence(image, radiometric_results)

    # Test Atmospheric Augmentations
    print("\n3. Atmospheric Augmentations:")
    atmospheric = AtmosphericAugmentations()

    # Haze
    haze = atmospheric.add_haze(image, 0.3)
    print("   - Haze: Added atmospheric haze")

    # Clouds
    clouds = atmospheric.add_clouds(image, 0.3, 0.6)
    print("   - Clouds: Added cloud cover")

    # Scattering
    scattering = atmospheric.atmospheric_scattering(image, 0.2)
    print("   - Scattering: Applied Rayleigh scattering")

    # Visualize atmospheric augmentations
    atmospheric_results = [
        (haze, "Haze"),
        (clouds, "Clouds"),
        (scattering, "Atmospheric Scattering"),
    ]
    visualizer.visualize_augmentation_sequence(image, atmospheric_results)

    # Test Sensor Augmentations
    print("\n4. Sensor Augmentations:")
    sensor = SensorAugmentations()

    # Noise
    noise = sensor.add_sensor_noise(image, "gaussian", 0.01)
    print("   - Noise: Added Gaussian noise")

    # Motion blur
    motion = sensor.motion_blur(image, 45, 7)
    print("   - Motion Blur: Applied platform motion")

    # Chromatic aberration
    chromatic = sensor.chromatic_aberration(image, 3)
    print("   - Chromatic: Added aberration")

    # Vignetting
    vignette = sensor.vignetting(image, 0.4)
    print("   - Vignetting: Added lens effect")

    # Visualize sensor augmentations
    sensor_results = [
        (noise, "Sensor Noise"),
        (motion, "Motion Blur"),
        (chromatic, "Chromatic Aberration"),
        (vignette, "Vignetting"),
    ]
    visualizer.visualize_augmentation_sequence(image, sensor_results)

    # Test Temporal Augmentations
    print("\n5. Temporal Augmentations:")
    temporal = TemporalAugmentations()

    # Seasonal variations
    seasons = []
    for season in ["spring", "summer", "autumn", "winter"]:
        seasonal = temporal.seasonal_color_shift(image, season)
        seasons.append((seasonal, season.capitalize()))
        print(f"   - {season.capitalize()}: Applied seasonal variation")

    visualizer.visualize_augmentation_sequence(image, seasons)

    # Sun angle
    sun_low = temporal.sun_angle_illumination(image, 20, 90)
    sun_high = temporal.sun_angle_illumination(image, 70, 180)
    print("   - Sun Angle: Applied different illuminations")

    sun_results = [(sun_low, "Low Sun (20°)"), (sun_high, "High Sun (70°)")]
    visualizer.visualize_augmentation_sequence(image, sun_results)


def test_pipeline_with_presets(
    image: np.ndarray, mask: Optional[np.ndarray] = None
) -> None:
    """
    Test augmentation pipeline with different presets

    Args:
        image: Input image
        mask: Optional segmentation mask
    """
    print("\n" + "=" * 50)
    print("Testing Pipeline with Presets")
    print("=" * 50)

    visualizer = AugmentationVisualizer()

    presets = [
        "light",
        "moderate",
        "heavy",
        "geometric_only",
        "atmospheric_only",
        "sensor_artifacts",
    ]

    augmented_images = {}

    for preset_name in presets:
        print(f"\nTesting preset: {preset_name}")

        # Create pipeline with preset
        config = create_preset_config(preset_name)
        pipeline = AerialAugmentationPipeline(config)

        # Apply augmentation
        aug_image, aug_mask, params = pipeline.augment(image, mask, return_params=True)

        augmented_images[preset_name] = [aug_image]

        # Print applied augmentations
        if params:
            print(f"  Applied augmentations:")
            for aug_name, aug_params in params.items():
                print(f"    - {aug_name}: {aug_params}")
        else:
            print("  No augmentations applied (probability check)")

    # Visualize all presets
    visualizer.visualize_augmentation_grid(image, augmented_images)


def test_custom_pipeline(image: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
    """
    Test custom augmentation pipeline

    Args:
        image: Input image
        mask: Optional segmentation mask
    """
    print("\n" + "=" * 50)
    print("Testing Custom Pipeline Configuration")
    print("=" * 50)

    # Create custom configuration
    config = AugmentationConfig(
        enabled_types=[
            AugmentationType.GEOMETRIC,
            AugmentationType.ATMOSPHERIC,
            AugmentationType.RADIOMETRIC,
        ],
        probability=0.8,
        rotation_range=(-90, 90),
        brightness_range=(0.7, 1.3),
        haze_intensity_range=(0.1, 0.4),
        cloud_coverage_range=(0.2, 0.5),
        shadow_intensity_range=(0.3, 0.6),
    )

    pipeline = AerialAugmentationPipeline(config)
    visualizer = AugmentationVisualizer()

    # Generate multiple augmentations
    augmented_samples = []

    for i in range(6):
        aug_image, aug_mask, params = pipeline.augment(image, mask, return_params=True)
        augmented_samples.append((aug_image, f"Sample {i+1}"))

        print(f"\nSample {i+1} augmentations:")
        for aug_name, aug_params in params.items():
            print(f"  - {aug_name}")

    # Visualize samples
    visualizer.visualize_augmentation_sequence(image, augmented_samples)

    # Test with mask if available
    if mask is not None:
        print("\n\nTesting with segmentation mask:")
        aug_image, aug_mask, _ = pipeline.augment(image, mask)
        visualizer.visualize_mask_augmentation(image, mask, aug_image, aug_mask)


def test_multispectral_augmentations(width: int = 256, height: int = 256) -> None:
    """
    Test multispectral-specific augmentations

    Args:
        width: Image width
        height: Image height
    """
    print("\n" + "=" * 50)
    print("Testing Multispectral Augmentations")
    print("=" * 50)

    # Create multispectral image (6 bands)
    multispectral = create_synthetic_aerial_image(width, height, num_bands=6)

    multi_aug = MultispectralAugmentations()
    visualizer = AugmentationVisualizer()

    # Band misalignment
    misaligned = multi_aug.band_misalignment(multispectral, 3)
    print("  - Band Misalignment: Applied registration errors")

    # Spectral noise
    spectral_noise = multi_aug.spectral_noise(multispectral, band_specific=True)
    print("  - Spectral Noise: Added band-specific noise")

    # Atmospheric absorption
    absorption = multi_aug.atmospheric_absorption(multispectral, [2, 4], 0.3)
    print("  - Atmospheric Absorption: Simulated absorption bands")

    # Visualize RGB channels only for display
    rgb_original = multispectral[:, :, :3]
    rgb_misaligned = misaligned[:, :, :3]
    rgb_noise = spectral_noise[:, :, :3]
    rgb_absorption = absorption[:, :, :3]

    results = [
        (rgb_misaligned, "Band Misalignment"),
        (rgb_noise, "Spectral Noise"),
        (rgb_absorption, "Atmospheric Absorption"),
    ]

    visualizer.visualize_augmentation_sequence(rgb_original, results)


def main():
    """Main function to run all tests"""
    parser = argparse.ArgumentParser(
        description="Test aerial imagery augmentation pipeline"
    )
    parser.add_argument("--image-path", type=str, help="Path to test image (optional)")
    parser.add_argument(
        "--mask-path", type=str, help="Path to segmentation mask (optional)"
    )
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["all", "individual", "presets", "custom", "multispectral"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument("--save-dir", type=str, help="Directory to save results")

    args = parser.parse_args()

    # Load or create test image
    if args.image_path:
        image = cv2.imread(args.image_path)
        if image is None:
            print(f"Error: Could not load image from {args.image_path}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Loaded image: {image.shape}")
    else:
        print("Creating synthetic aerial image for testing...")
        image = create_synthetic_aerial_image(512, 512)

    # Load or create mask
    mask = None
    if args.mask_path:
        mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not load mask from {args.mask_path}")
    else:
        print("Creating synthetic segmentation mask...")
        mask = create_synthetic_mask(image)

    # Run tests
    print("\n" + "=" * 50)
    print("AERIAL IMAGERY AUGMENTATION PIPELINE TEST")
    print("=" * 50)

    if args.test_type == "all" or args.test_type == "individual":
        test_individual_augmentations(image, mask)

    if args.test_type == "all" or args.test_type == "presets":
        test_pipeline_with_presets(image, mask)

    if args.test_type == "all" or args.test_type == "custom":
        test_custom_pipeline(image, mask)

    if args.test_type == "all" or args.test_type == "multispectral":
        test_multispectral_augmentations()

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)

    # Show final message
    print("\nThe aerial imagery augmentation pipeline includes:")
    print("  - Geometric: rotation, perspective, orthorectification, elastic")
    print("  - Radiometric: brightness, contrast, gamma, shadows, histogram")
    print("  - Atmospheric: haze, clouds, scattering")
    print("  - Sensor: noise, motion blur, chromatic aberration, vignetting")
    print("  - Temporal: seasonal variations, sun angle illumination")
    print("  - Multispectral: band misalignment, spectral noise, absorption")
    print("\nUse the pipeline with different configurations for your specific needs!")


if __name__ == "__main__":
    main()

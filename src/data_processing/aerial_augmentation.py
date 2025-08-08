"""
Comprehensive Aerial Imagery Augmentation Pipeline

This module provides specialized augmentation techniques for overhead/satellite imagery,
including geometric, radiometric, atmospheric, sensor-specific, and temporal augmentations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import random
import math
from enum import Enum
import cv2
from scipy import ndimage
from scipy.interpolate import griddata
import warnings


class AugmentationType(Enum):
    """Types of augmentations available"""
    GEOMETRIC = "geometric"
    RADIOMETRIC = "radiometric"
    ATMOSPHERIC = "atmospheric"
    SENSOR = "sensor"
    TEMPORAL = "temporal"
    MULTISPECTRAL = "multispectral"


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline"""
    enabled_types: List[AugmentationType] = field(default_factory=lambda: list(AugmentationType))
    probability: float = 0.5
    seed: Optional[int] = None
    preserve_dtype: bool = True
    clip_values: bool = True
    
    # Geometric parameters
    rotation_range: Tuple[float, float] = (-180, 180)
    scale_range: Tuple[float, float] = (0.8, 1.2)
    shear_range: Tuple[float, float] = (-10, 10)
    perspective_strength: float = 0.001
    elastic_alpha: float = 50
    elastic_sigma: float = 5
    
    # Radiometric parameters
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    gamma_range: Tuple[float, float] = (0.8, 1.2)
    shadow_intensity_range: Tuple[float, float] = (0.3, 0.7)
    shadow_area_range: Tuple[float, float] = (0.1, 0.3)
    
    # Atmospheric parameters
    haze_intensity_range: Tuple[float, float] = (0.1, 0.3)
    cloud_opacity_range: Tuple[float, float] = (0.3, 0.8)
    cloud_coverage_range: Tuple[float, float] = (0.1, 0.4)
    atmospheric_scattering_strength: float = 0.1
    
    # Sensor parameters
    noise_variance_range: Tuple[float, float] = (0.001, 0.01)
    blur_kernel_range: Tuple[int, int] = (3, 7)
    chromatic_aberration_strength: float = 2.0
    vignetting_strength: float = 0.3
    dead_pixel_probability: float = 0.0001
    
    # Temporal parameters
    seasonal_variation_strength: float = 0.2
    illumination_angle_range: Tuple[float, float] = (20, 70)
    
    # Multispectral parameters
    band_shift_range: Tuple[float, float] = (-5, 5)
    band_noise_variance: float = 0.005
    atmospheric_absorption_bands: List[int] = field(default_factory=list)


class GeometricAugmentations:
    """Geometric transformations specific to aerial imagery"""
    
    @staticmethod
    def rotate_nadir(image: np.ndarray, angle: float, 
                     mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Rotate image around nadir point (center)
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            mask: Optional segmentation mask
        
        Returns:
            Rotated image and mask
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated_image = cv2.warpAffine(image, M, (w, h), 
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
        
        rotated_mask = None
        if mask is not None:
            rotated_mask = cv2.warpAffine(mask, M, (w, h),
                                         flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_REFLECT)
        
        return rotated_image, rotated_mask
    
    @staticmethod
    def perspective_transform(image: np.ndarray, strength: float = 0.001,
                            mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply perspective transformation to simulate viewing angle changes
        
        Args:
            image: Input image
            strength: Strength of perspective effect
            mask: Optional segmentation mask
        
        Returns:
            Transformed image and mask
        """
        h, w = image.shape[:2]
        
        # Define source points (corners of image)
        src_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        
        # Add random perspective distortion
        dst_points = src_points.copy()
        for i in range(4):
            dst_points[i][0] += random.uniform(-strength * w, strength * w)
            dst_points[i][1] += random.uniform(-strength * h, strength * h)
        
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        warped_image = cv2.warpPerspective(image, M, (w, h),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT)
        
        warped_mask = None
        if mask is not None:
            warped_mask = cv2.warpPerspective(mask, M, (w, h),
                                             flags=cv2.INTER_NEAREST,
                                             borderMode=cv2.BORDER_REFLECT)
        
        return warped_image, warped_mask
    
    @staticmethod
    def orthorectification_distortion(image: np.ndarray, 
                                     elevation_variation: float = 50.0,
                                     mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Simulate orthorectification artifacts from terrain elevation
        
        Args:
            image: Input image
            elevation_variation: Maximum elevation variation in pixels
            mask: Optional segmentation mask
        
        Returns:
            Distorted image and mask
        """
        h, w = image.shape[:2]
        
        # Create synthetic elevation map
        x = np.linspace(0, 4*np.pi, w)
        y = np.linspace(0, 4*np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        # Generate smooth elevation variations
        elevation = elevation_variation * (
            np.sin(X * 0.5) * np.cos(Y * 0.3) +
            np.sin(X * 0.2) * np.sin(Y * 0.7) * 0.5
        )
        
        # Create displacement maps
        flow_x = ndimage.gaussian_filter(elevation, sigma=3) * 0.1
        flow_y = ndimage.gaussian_filter(elevation.T, sigma=3).T * 0.1
        
        # Create coordinate grids
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        map_x = (grid_x + flow_x).astype(np.float32)
        map_y = (grid_y + flow_y).astype(np.float32)
        
        distorted_image = cv2.remap(image, map_x, map_y, 
                                   cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
        
        distorted_mask = None
        if mask is not None:
            distorted_mask = cv2.remap(mask, map_x, map_y,
                                      cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_REFLECT)
        
        return distorted_image, distorted_mask
    
    @staticmethod
    def elastic_transform(image: np.ndarray, alpha: float = 50, sigma: float = 5,
                         mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply elastic deformation to simulate geometric distortions
        
        Args:
            image: Input image
            alpha: Strength of deformation
            sigma: Smoothness of deformation
            mask: Optional segmentation mask
        
        Returns:
            Deformed image and mask
        """
        h, w = image.shape[:2]
        
        # Random displacement fields
        dx = np.random.randn(h, w) * alpha
        dy = np.random.randn(h, w) * alpha
        
        # Smooth the displacement fields
        dx = ndimage.gaussian_filter(dx, sigma)
        dy = ndimage.gaussian_filter(dy, sigma)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        deformed_image = cv2.remap(image, map_x, map_y,
                                  cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
        
        deformed_mask = None
        if mask is not None:
            deformed_mask = cv2.remap(mask, map_x, map_y,
                                     cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_REFLECT)
        
        return deformed_image, deformed_mask


class RadiometricAugmentations:
    """Radiometric and color transformations for aerial imagery"""
    
    @staticmethod
    def adjust_brightness_contrast(image: np.ndarray, 
                                  brightness: float = 1.0,
                                  contrast: float = 1.0) -> np.ndarray:
        """
        Adjust brightness and contrast
        
        Args:
            image: Input image
            brightness: Brightness factor
            contrast: Contrast factor
        
        Returns:
            Adjusted image
        """
        result = image.astype(np.float32)
        
        # Apply contrast
        mean = np.mean(result, axis=(0, 1), keepdims=True)
        result = mean + contrast * (result - mean)
        
        # Apply brightness
        result = result * brightness
        
        return np.clip(result, 0, 255).astype(image.dtype)
    
    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply gamma correction
        
        Args:
            image: Input image
            gamma: Gamma value
        
        Returns:
            Corrected image
        """
        normalized = image.astype(np.float32) / 255.0
        corrected = np.power(normalized, gamma)
        return (corrected * 255).astype(image.dtype)
    
    @staticmethod
    def simulate_shadows(image: np.ndarray, 
                        num_shadows: int = 3,
                        intensity_range: Tuple[float, float] = (0.3, 0.7),
                        area_range: Tuple[float, float] = (0.1, 0.3)) -> np.ndarray:
        """
        Simulate shadows from clouds or tall structures
        
        Args:
            image: Input image
            num_shadows: Number of shadow regions
            intensity_range: Shadow darkness range
            area_range: Shadow area range as fraction of image
        
        Returns:
            Image with shadows
        """
        h, w = image.shape[:2]
        shadow_mask = np.ones((h, w), dtype=np.float32)
        
        for _ in range(num_shadows):
            # Random shadow parameters
            intensity = random.uniform(*intensity_range)
            area = random.uniform(*area_range)
            
            # Create irregular shadow shape
            num_vertices = random.randint(4, 8)
            vertices = []
            center_x = random.randint(w//4, 3*w//4)
            center_y = random.randint(h//4, 3*h//4)
            
            for i in range(num_vertices):
                angle = 2 * np.pi * i / num_vertices
                r = random.uniform(0.5, 1.0) * min(w, h) * np.sqrt(area) / 2
                x = center_x + r * np.cos(angle)
                y = center_y + r * np.sin(angle)
                vertices.append([int(x), int(y)])
            
            vertices = np.array(vertices, dtype=np.int32)
            
            # Create shadow mask
            temp_mask = np.ones((h, w), dtype=np.float32)
            cv2.fillPoly(temp_mask, [vertices], intensity)
            
            # Blur shadow edges
            temp_mask = cv2.GaussianBlur(temp_mask, (21, 21), 10)
            
            # Combine with existing shadows
            shadow_mask = np.minimum(shadow_mask, temp_mask)
        
        # Apply shadows to image
        result = image.astype(np.float32)
        if len(image.shape) == 3:
            shadow_mask = shadow_mask[:, :, np.newaxis]
        
        result = result * shadow_mask
        
        return result.astype(image.dtype)
    
    @staticmethod
    def histogram_matching(image: np.ndarray, 
                          reference_histogram: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Match histogram to reference or equalize
        
        Args:
            image: Input image
            reference_histogram: Optional reference histogram
        
        Returns:
            Histogram-matched image
        """
        if len(image.shape) == 3:
            # Process each channel separately
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = cv2.equalizeHist(image[:, :, i])
            return result
        else:
            return cv2.equalizeHist(image)


class AtmosphericAugmentations:
    """Atmospheric effects simulation for aerial imagery"""
    
    @staticmethod
    def add_haze(image: np.ndarray, intensity: float = 0.2) -> np.ndarray:
        """
        Add atmospheric haze effect
        
        Args:
            image: Input image
            intensity: Haze intensity
        
        Returns:
            Image with haze
        """
        h, w = image.shape[:2]
        
        # Create haze layer with gradient
        haze = np.ones((h, w), dtype=np.float32)
        
        # Vertical gradient for atmospheric perspective
        for i in range(h):
            haze[i, :] = 1 - (i / h) * intensity * 0.5
        
        # Add some noise to haze
        noise = np.random.randn(h, w) * 0.02
        haze += noise
        
        # Smooth the haze
        haze = cv2.GaussianBlur(haze, (51, 51), 25)
        haze = np.clip(haze, 1 - intensity, 1)
        
        # Apply haze
        result = image.astype(np.float32)
        if len(image.shape) == 3:
            haze = haze[:, :, np.newaxis]
        
        # Mix with white/gray for haze effect
        haze_color = np.array([200, 200, 210], dtype=np.float32)
        if len(image.shape) == 3:
            result = result * haze + haze_color * (1 - haze)
        else:
            result = result * haze + 200 * (1 - haze)
        
        return np.clip(result, 0, 255).astype(image.dtype)
    
    @staticmethod
    def add_clouds(image: np.ndarray, 
                  coverage: float = 0.3,
                  opacity: float = 0.7) -> np.ndarray:
        """
        Add cloud cover to image
        
        Args:
            image: Input image
            coverage: Fraction of image covered by clouds
            opacity: Cloud opacity
        
        Returns:
            Image with clouds
        """
        h, w = image.shape[:2]
        
        # Generate cloud pattern using Perlin-like noise
        cloud_mask = np.zeros((h, w), dtype=np.float32)
        
        # Multiple octaves for realistic clouds
        octaves = 4
        for octave in range(octaves):
            scale = 2 ** octave
            noise = np.random.randn(h // scale + 1, w // scale + 1)
            
            # Upscale noise
            noise_upscaled = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Weight by octave
            weight = 1.0 / (2 ** octave)
            cloud_mask += noise_upscaled * weight
        
        # Normalize and threshold for coverage
        cloud_mask = (cloud_mask - cloud_mask.min()) / (cloud_mask.max() - cloud_mask.min())
        threshold = np.percentile(cloud_mask, (1 - coverage) * 100)
        cloud_mask = np.where(cloud_mask > threshold, cloud_mask, 0)
        
        # Smooth cloud edges
        cloud_mask = cv2.GaussianBlur(cloud_mask, (31, 31), 15)
        cloud_mask = np.clip(cloud_mask * opacity, 0, opacity)
        
        # Apply clouds
        result = image.astype(np.float32)
        if len(image.shape) == 3:
            cloud_mask = cloud_mask[:, :, np.newaxis]
            cloud_color = np.array([240, 240, 245], dtype=np.float32)
            result = result * (1 - cloud_mask) + cloud_color * cloud_mask
        else:
            result = result * (1 - cloud_mask) + 240 * cloud_mask
        
        return np.clip(result, 0, 255).astype(image.dtype)
    
    @staticmethod
    def atmospheric_scattering(image: np.ndarray, 
                              wavelength_shift: float = 0.1) -> np.ndarray:
        """
        Simulate Rayleigh scattering effects
        
        Args:
            image: Input image
            wavelength_shift: Strength of scattering effect
        
        Returns:
            Image with atmospheric scattering
        """
        if len(image.shape) != 3:
            return image
        
        result = image.astype(np.float32)
        
        # Rayleigh scattering is stronger for shorter wavelengths (blue)
        # Approximate with channel-specific attenuation
        scattering_factors = np.array([
            1.0 + wavelength_shift * 0.3,  # Blue scattered more
            1.0 + wavelength_shift * 0.1,  # Green
            1.0 - wavelength_shift * 0.1   # Red scattered less
        ])
        
        for i in range(3):
            if i < result.shape[2]:
                result[:, :, i] *= scattering_factors[i]
        
        return np.clip(result, 0, 255).astype(image.dtype)


class SensorAugmentations:
    """Sensor-specific artifacts and characteristics"""
    
    @staticmethod
    def add_sensor_noise(image: np.ndarray, 
                        noise_type: str = 'gaussian',
                        variance: float = 0.01) -> np.ndarray:
        """
        Add sensor noise
        
        Args:
            image: Input image
            noise_type: Type of noise ('gaussian', 'poisson', 'salt_pepper')
            variance: Noise variance
        
        Returns:
            Noisy image
        """
        result = image.astype(np.float32) / 255.0
        
        if noise_type == 'gaussian':
            noise = np.random.randn(*image.shape) * np.sqrt(variance)
            result += noise
        
        elif noise_type == 'poisson':
            result = np.random.poisson(result / variance) * variance
        
        elif noise_type == 'salt_pepper':
            mask = np.random.random(image.shape) < variance
            result[mask] = np.random.choice([0, 1], size=np.sum(mask))
        
        result = np.clip(result * 255, 0, 255)
        return result.astype(image.dtype)
    
    @staticmethod
    def motion_blur(image: np.ndarray, 
                   angle: float = 0,
                   length: int = 10) -> np.ndarray:
        """
        Simulate motion blur from platform movement
        
        Args:
            image: Input image
            angle: Blur angle in degrees
            length: Blur length in pixels
        
        Returns:
            Blurred image
        """
        # Create motion blur kernel
        kernel = np.zeros((length, length))
        
        # Calculate kernel center
        center = length // 2
        
        # Create line in kernel
        angle_rad = np.deg2rad(angle)
        for i in range(length):
            x = int(center + (i - center) * np.cos(angle_rad))
            y = int(center + (i - center) * np.sin(angle_rad))
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        
        # Apply blur
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = cv2.filter2D(image[:, :, i], -1, kernel)
            return result
        else:
            return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def chromatic_aberration(image: np.ndarray, 
                            shift: float = 2.0) -> np.ndarray:
        """
        Simulate chromatic aberration
        
        Args:
            image: Input image
            shift: Maximum pixel shift
        
        Returns:
            Image with chromatic aberration
        """
        if len(image.shape) != 3:
            return image
        
        h, w = image.shape[:2]
        result = np.zeros_like(image)
        
        # Different shifts for each channel
        shifts = [
            (shift, 0),      # Red channel
            (0, 0),          # Green channel (reference)
            (-shift, 0)      # Blue channel
        ]
        
        for i, (dx, dy) in enumerate(shifts):
            if i < image.shape[2]:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                result[:, :, i] = cv2.warpAffine(image[:, :, i], M, (w, h),
                                                borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    @staticmethod
    def vignetting(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        Add lens vignetting effect
        
        Args:
            image: Input image
            strength: Vignetting strength
        
        Returns:
            Image with vignetting
        """
        h, w = image.shape[:2]
        
        # Create radial gradient
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distance from center
        radius = np.sqrt(X**2 + Y**2)
        radius = np.clip(radius, 0, 1)
        
        # Create vignette mask
        vignette = 1 - strength * radius**2
        
        # Apply vignetting
        result = image.astype(np.float32)
        if len(image.shape) == 3:
            vignette = vignette[:, :, np.newaxis]
        
        result = result * vignette
        
        return np.clip(result, 0, 255).astype(image.dtype)
    
    @staticmethod
    def dead_pixels(image: np.ndarray, 
                   probability: float = 0.0001) -> np.ndarray:
        """
        Simulate dead/hot pixels
        
        Args:
            image: Input image
            probability: Probability of dead pixels
        
        Returns:
            Image with dead pixels
        """
        mask = np.random.random(image.shape[:2]) < probability
        result = image.copy()
        
        # Dead pixels (black) or hot pixels (white)
        dead_value = np.random.choice([0, 255], size=np.sum(mask))
        
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                result[:, :, i][mask] = dead_value
        else:
            result[mask] = dead_value
        
        return result


class TemporalAugmentations:
    """Temporal and seasonal variations"""
    
    @staticmethod
    def seasonal_color_shift(image: np.ndarray, 
                            season: str = 'summer') -> np.ndarray:
        """
        Apply seasonal color variations
        
        Args:
            image: Input image
            season: Season type ('spring', 'summer', 'autumn', 'winter')
        
        Returns:
            Seasonally adjusted image
        """
        if len(image.shape) != 3:
            return image
        
        result = image.astype(np.float32)
        
        season_adjustments = {
            'spring': [1.0, 1.1, 0.9],   # More green
            'summer': [1.05, 1.05, 0.95],  # Warmer, brighter
            'autumn': [1.1, 0.95, 0.85],   # More red/orange
            'winter': [0.95, 0.95, 1.05]   # Cooler, more blue
        }
        
        if season in season_adjustments:
            factors = season_adjustments[season]
            for i in range(min(3, image.shape[2])):
                result[:, :, i] *= factors[i]
        
        return np.clip(result, 0, 255).astype(image.dtype)
    
    @staticmethod
    def sun_angle_illumination(image: np.ndarray, 
                              sun_elevation: float = 45,
                              sun_azimuth: float = 180) -> np.ndarray:
        """
        Simulate different sun angles and illumination
        
        Args:
            image: Input image
            sun_elevation: Sun elevation angle in degrees
            sun_azimuth: Sun azimuth angle in degrees
        
        Returns:
            Image with adjusted illumination
        """
        h, w = image.shape[:2]
        
        # Create synthetic elevation map for shadows
        x = np.linspace(0, 2*np.pi, w)
        y = np.linspace(0, 2*np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        # Simple elevation model
        elevation = np.sin(X) * np.cos(Y) * 20
        
        # Calculate illumination based on sun angle
        sun_elevation_rad = np.deg2rad(sun_elevation)
        sun_azimuth_rad = np.deg2rad(sun_azimuth)
        
        # Surface normal (simplified)
        dx = np.gradient(elevation, axis=1)
        dy = np.gradient(elevation, axis=0)
        
        # Illumination factor
        illumination = np.cos(sun_elevation_rad) - \
                      dx * np.sin(sun_elevation_rad) * np.cos(sun_azimuth_rad) - \
                      dy * np.sin(sun_elevation_rad) * np.sin(sun_azimuth_rad)
        
        illumination = np.clip(illumination, 0.3, 1.0)
        illumination = cv2.GaussianBlur(illumination.astype(np.float32), (21, 21), 10)
        
        # Apply illumination
        result = image.astype(np.float32)
        if len(image.shape) == 3:
            illumination = illumination[:, :, np.newaxis]
        
        result = result * illumination
        
        return np.clip(result, 0, 255).astype(image.dtype)


class MultispectralAugmentations:
    """Augmentations specific to multispectral imagery"""
    
    @staticmethod
    def band_misalignment(image: np.ndarray, 
                         max_shift: float = 5) -> np.ndarray:
        """
        Simulate band misalignment/registration errors
        
        Args:
            image: Multispectral image
            max_shift: Maximum pixel shift
        
        Returns:
            Image with band misalignment
        """
        if len(image.shape) != 3:
            return image
        
        result = np.zeros_like(image)
        h, w = image.shape[:2]
        
        for i in range(image.shape[2]):
            # Random shift for each band
            dx = random.uniform(-max_shift, max_shift)
            dy = random.uniform(-max_shift, max_shift)
            
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            result[:, :, i] = cv2.warpAffine(image[:, :, i], M, (w, h),
                                            borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    @staticmethod
    def spectral_noise(image: np.ndarray, 
                      band_specific: bool = True,
                      noise_levels: Optional[List[float]] = None) -> np.ndarray:
        """
        Add band-specific noise patterns
        
        Args:
            image: Multispectral image
            band_specific: Whether to use different noise for each band
            noise_levels: Specific noise levels for each band
        
        Returns:
            Noisy image
        """
        result = image.astype(np.float32)
        
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                if noise_levels and i < len(noise_levels):
                    variance = noise_levels[i]
                else:
                    # Different bands have different noise characteristics
                    variance = 0.01 * (1 + i * 0.2) if band_specific else 0.01
                
                noise = np.random.randn(*image.shape[:2]) * np.sqrt(variance) * 255
                result[:, :, i] += noise
        else:
            noise = np.random.randn(*image.shape) * 0.01 * 255
            result += noise
        
        return np.clip(result, 0, 255).astype(image.dtype)
    
    @staticmethod
    def atmospheric_absorption(image: np.ndarray,
                              absorption_bands: List[int] = [3, 5],
                              absorption_strength: float = 0.3) -> np.ndarray:
        """
        Simulate atmospheric absorption at specific wavelengths
        
        Args:
            image: Multispectral image
            absorption_bands: Band indices with absorption
            absorption_strength: Strength of absorption
        
        Returns:
            Image with atmospheric absorption
        """
        if len(image.shape) != 3:
            return image
        
        result = image.copy()
        
        for band_idx in absorption_bands:
            if band_idx < image.shape[2]:
                result[:, :, band_idx] = (
                    result[:, :, band_idx] * (1 - absorption_strength)
                ).astype(image.dtype)
        
        return result


class AerialAugmentationPipeline:
    """Main augmentation pipeline for aerial imagery"""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmentation pipeline
        
        Args:
            config: Configuration object
        """
        self.config = config or AugmentationConfig()
        
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
        
        # Initialize augmentation classes
        self.geometric = GeometricAugmentations()
        self.radiometric = RadiometricAugmentations()
        self.atmospheric = AtmosphericAugmentations()
        self.sensor = SensorAugmentations()
        self.temporal = TemporalAugmentations()
        self.multispectral = MultispectralAugmentations()
        
        # Build augmentation registry
        self._build_augmentation_registry()
    
    def _build_augmentation_registry(self):
        """Build registry of available augmentations"""
        self.augmentation_registry = {
            AugmentationType.GEOMETRIC: [
                ('rotate_nadir', self._apply_rotation),
                ('perspective', self._apply_perspective),
                ('orthorectification', self._apply_orthorectification),
                ('elastic', self._apply_elastic)
            ],
            AugmentationType.RADIOMETRIC: [
                ('brightness_contrast', self._apply_brightness_contrast),
                ('gamma', self._apply_gamma),
                ('shadows', self._apply_shadows),
                ('histogram', self._apply_histogram)
            ],
            AugmentationType.ATMOSPHERIC: [
                ('haze', self._apply_haze),
                ('clouds', self._apply_clouds),
                ('scattering', self._apply_scattering)
            ],
            AugmentationType.SENSOR: [
                ('noise', self._apply_noise),
                ('motion_blur', self._apply_motion_blur),
                ('chromatic', self._apply_chromatic),
                ('vignetting', self._apply_vignetting),
                ('dead_pixels', self._apply_dead_pixels)
            ],
            AugmentationType.TEMPORAL: [
                ('seasonal', self._apply_seasonal),
                ('sun_angle', self._apply_sun_angle)
            ],
            AugmentationType.MULTISPECTRAL: [
                ('band_misalignment', self._apply_band_misalignment),
                ('spectral_noise', self._apply_spectral_noise),
                ('absorption', self._apply_absorption)
            ]
        }
    
    def augment(self, image: np.ndarray, 
               mask: Optional[np.ndarray] = None,
               augmentation_types: Optional[List[AugmentationType]] = None,
               return_params: bool = False) -> Union[
                   Tuple[np.ndarray, Optional[np.ndarray]],
                   Tuple[np.ndarray, Optional[np.ndarray], Dict]
               ]:
        """
        Apply augmentation pipeline to image
        
        Args:
            image: Input image
            mask: Optional segmentation mask
            augmentation_types: Specific augmentation types to apply
            return_params: Whether to return augmentation parameters
        
        Returns:
            Augmented image, mask, and optionally parameters
        """
        # Determine which augmentations to apply
        if augmentation_types is None:
            augmentation_types = self.config.enabled_types
        
        # Store original dtype
        original_dtype = image.dtype
        
        # Track applied augmentations
        applied_params = {}
        
        # Apply augmentations by type
        for aug_type in augmentation_types:
            if aug_type not in self.augmentation_registry:
                continue
            
            # Get augmentations for this type
            augmentations = self.augmentation_registry[aug_type]
            
            # Apply each augmentation with probability
            for aug_name, aug_func in augmentations:
                if random.random() < self.config.probability:
                    image, mask, params = aug_func(image, mask)
                    applied_params[f"{aug_type.value}_{aug_name}"] = params
        
        # Restore original dtype if requested
        if self.config.preserve_dtype:
            image = image.astype(original_dtype)
            if mask is not None:
                mask = mask.astype(np.uint8)
        
        # Clip values if requested
        if self.config.clip_values:
            image = np.clip(image, 0, 255)
        
        if return_params:
            return image, mask, applied_params
        else:
            return image, mask
    
    # Geometric augmentation wrappers
    def _apply_rotation(self, image: np.ndarray, mask: Optional[np.ndarray]):
        angle = random.uniform(*self.config.rotation_range)
        image, mask = self.geometric.rotate_nadir(image, angle, mask)
        return image, mask, {'angle': angle}
    
    def _apply_perspective(self, image: np.ndarray, mask: Optional[np.ndarray]):
        strength = self.config.perspective_strength
        image, mask = self.geometric.perspective_transform(image, strength, mask)
        return image, mask, {'strength': strength}
    
    def _apply_orthorectification(self, image: np.ndarray, mask: Optional[np.ndarray]):
        elevation = random.uniform(20, 100)
        image, mask = self.geometric.orthorectification_distortion(image, elevation, mask)
        return image, mask, {'elevation_variation': elevation}
    
    def _apply_elastic(self, image: np.ndarray, mask: Optional[np.ndarray]):
        image, mask = self.geometric.elastic_transform(
            image, self.config.elastic_alpha, self.config.elastic_sigma, mask
        )
        return image, mask, {'alpha': self.config.elastic_alpha, 'sigma': self.config.elastic_sigma}
    
    # Radiometric augmentation wrappers
    def _apply_brightness_contrast(self, image: np.ndarray, mask: Optional[np.ndarray]):
        brightness = random.uniform(*self.config.brightness_range)
        contrast = random.uniform(*self.config.contrast_range)
        image = self.radiometric.adjust_brightness_contrast(image, brightness, contrast)
        return image, mask, {'brightness': brightness, 'contrast': contrast}
    
    def _apply_gamma(self, image: np.ndarray, mask: Optional[np.ndarray]):
        gamma = random.uniform(*self.config.gamma_range)
        image = self.radiometric.gamma_correction(image, gamma)
        return image, mask, {'gamma': gamma}
    
    def _apply_shadows(self, image: np.ndarray, mask: Optional[np.ndarray]):
        num_shadows = random.randint(1, 5)
        image = self.radiometric.simulate_shadows(
            image, num_shadows, 
            self.config.shadow_intensity_range,
            self.config.shadow_area_range
        )
        return image, mask, {'num_shadows': num_shadows}
    
    def _apply_histogram(self, image: np.ndarray, mask: Optional[np.ndarray]):
        image = self.radiometric.histogram_matching(image)
        return image, mask, {}
    
    # Atmospheric augmentation wrappers
    def _apply_haze(self, image: np.ndarray, mask: Optional[np.ndarray]):
        intensity = random.uniform(*self.config.haze_intensity_range)
        image = self.atmospheric.add_haze(image, intensity)
        return image, mask, {'intensity': intensity}
    
    def _apply_clouds(self, image: np.ndarray, mask: Optional[np.ndarray]):
        coverage = random.uniform(*self.config.cloud_coverage_range)
        opacity = random.uniform(*self.config.cloud_opacity_range)
        image = self.atmospheric.add_clouds(image, coverage, opacity)
        return image, mask, {'coverage': coverage, 'opacity': opacity}
    
    def _apply_scattering(self, image: np.ndarray, mask: Optional[np.ndarray]):
        strength = self.config.atmospheric_scattering_strength
        image = self.atmospheric.atmospheric_scattering(image, strength)
        return image, mask, {'strength': strength}
    
    # Sensor augmentation wrappers
    def _apply_noise(self, image: np.ndarray, mask: Optional[np.ndarray]):
        noise_type = random.choice(['gaussian', 'poisson', 'salt_pepper'])
        variance = random.uniform(*self.config.noise_variance_range)
        image = self.sensor.add_sensor_noise(image, noise_type, variance)
        return image, mask, {'type': noise_type, 'variance': variance}
    
    def _apply_motion_blur(self, image: np.ndarray, mask: Optional[np.ndarray]):
        angle = random.uniform(0, 360)
        length = random.randint(*self.config.blur_kernel_range)
        image = self.sensor.motion_blur(image, angle, length)
        return image, mask, {'angle': angle, 'length': length}
    
    def _apply_chromatic(self, image: np.ndarray, mask: Optional[np.ndarray]):
        shift = self.config.chromatic_aberration_strength
        image = self.sensor.chromatic_aberration(image, shift)
        return image, mask, {'shift': shift}
    
    def _apply_vignetting(self, image: np.ndarray, mask: Optional[np.ndarray]):
        strength = self.config.vignetting_strength
        image = self.sensor.vignetting(image, strength)
        return image, mask, {'strength': strength}
    
    def _apply_dead_pixels(self, image: np.ndarray, mask: Optional[np.ndarray]):
        probability = self.config.dead_pixel_probability
        image = self.sensor.dead_pixels(image, probability)
        return image, mask, {'probability': probability}
    
    # Temporal augmentation wrappers
    def _apply_seasonal(self, image: np.ndarray, mask: Optional[np.ndarray]):
        season = random.choice(['spring', 'summer', 'autumn', 'winter'])
        image = self.temporal.seasonal_color_shift(image, season)
        return image, mask, {'season': season}
    
    def _apply_sun_angle(self, image: np.ndarray, mask: Optional[np.ndarray]):
        elevation = random.uniform(*self.config.illumination_angle_range)
        azimuth = random.uniform(0, 360)
        image = self.temporal.sun_angle_illumination(image, elevation, azimuth)
        return image, mask, {'elevation': elevation, 'azimuth': azimuth}
    
    # Multispectral augmentation wrappers
    def _apply_band_misalignment(self, image: np.ndarray, mask: Optional[np.ndarray]):
        max_shift = random.uniform(*self.config.band_shift_range)
        image = self.multispectral.band_misalignment(image, max_shift)
        return image, mask, {'max_shift': max_shift}
    
    def _apply_spectral_noise(self, image: np.ndarray, mask: Optional[np.ndarray]):
        image = self.multispectral.spectral_noise(image, band_specific=True)
        return image, mask, {}
    
    def _apply_absorption(self, image: np.ndarray, mask: Optional[np.ndarray]):
        absorption_bands = self.config.atmospheric_absorption_bands or [3, 5]
        strength = random.uniform(0.1, 0.4)
        image = self.multispectral.atmospheric_absorption(image, absorption_bands, strength)
        return image, mask, {'bands': absorption_bands, 'strength': strength}
    
    def get_random_augmentation_sequence(self, 
                                        num_augmentations: int = 3) -> List[Callable]:
        """
        Get a random sequence of augmentations
        
        Args:
            num_augmentations: Number of augmentations to select
        
        Returns:
            List of augmentation functions
        """
        all_augmentations = []
        for aug_type in self.augmentation_registry:
            all_augmentations.extend(self.augmentation_registry[aug_type])
        
        selected = random.sample(all_augmentations, 
                               min(num_augmentations, len(all_augmentations)))
        
        return [aug_func for _, aug_func in selected]


def create_preset_config(preset: str) -> AugmentationConfig:
    """
    Create configuration from preset
    
    Args:
        preset: Preset name
    
    Returns:
        Configuration object
    """
    presets = {
        'light': AugmentationConfig(
            enabled_types=[
                AugmentationType.GEOMETRIC,
                AugmentationType.RADIOMETRIC
            ],
            probability=0.3,
            rotation_range=(-30, 30),
            brightness_range=(0.9, 1.1),
            contrast_range=(0.9, 1.1)
        ),
        'moderate': AugmentationConfig(
            enabled_types=[
                AugmentationType.GEOMETRIC,
                AugmentationType.RADIOMETRIC,
                AugmentationType.ATMOSPHERIC
            ],
            probability=0.5,
            rotation_range=(-90, 90),
            brightness_range=(0.8, 1.2),
            haze_intensity_range=(0.1, 0.2)
        ),
        'heavy': AugmentationConfig(
            enabled_types=list(AugmentationType),
            probability=0.7,
            rotation_range=(-180, 180),
            brightness_range=(0.7, 1.3),
            haze_intensity_range=(0.1, 0.4),
            cloud_coverage_range=(0.2, 0.5)
        ),
        'geometric_only': AugmentationConfig(
            enabled_types=[AugmentationType.GEOMETRIC],
            probability=0.8
        ),
        'atmospheric_only': AugmentationConfig(
            enabled_types=[AugmentationType.ATMOSPHERIC],
            probability=0.8
        ),
        'sensor_artifacts': AugmentationConfig(
            enabled_types=[AugmentationType.SENSOR],
            probability=0.6,
            noise_variance_range=(0.005, 0.02),
            dead_pixel_probability=0.001
        )
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    return presets[preset]
"""
Visualization utilities for aerial imagery augmentation pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Optional, Dict, Any
import cv2
from pathlib import Path


class AugmentationVisualizer:
    """Utility class for visualizing augmented aerial imagery"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
    
    def visualize_single_augmentation(self, 
                                     original: np.ndarray,
                                     augmented: np.ndarray,
                                     title: str = "Augmentation Result",
                                     save_path: Optional[str] = None) -> None:
        """
        Visualize original vs augmented image
        
        Args:
            original: Original image
            augmented: Augmented image
            title: Figure title
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Original image
        if len(original.shape) == 3:
            axes[0].imshow(original.astype(np.uint8))
        else:
            axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Augmented image
        if len(augmented.shape) == 3:
            axes[1].imshow(augmented.astype(np.uint8))
        else:
            axes[1].imshow(augmented, cmap='gray')
        axes[1].set_title('Augmented')
        axes[1].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def visualize_augmentation_sequence(self,
                                       original: np.ndarray,
                                       augmented_images: List[Tuple[np.ndarray, str]],
                                       save_path: Optional[str] = None) -> None:
        """
        Visualize a sequence of augmentations
        
        Args:
            original: Original image
            augmented_images: List of (image, name) tuples
            save_path: Optional path to save figure
        """
        n_augmentations = len(augmented_images)
        n_cols = min(3, n_augmentations + 1)
        n_rows = (n_augmentations + 1 + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Show original
        if len(original.shape) == 3:
            axes_flat[0].imshow(original.astype(np.uint8))
        else:
            axes_flat[0].imshow(original, cmap='gray')
        axes_flat[0].set_title('Original', fontweight='bold')
        axes_flat[0].axis('off')
        
        # Show augmented images
        for i, (aug_img, aug_name) in enumerate(augmented_images):
            if len(aug_img.shape) == 3:
                axes_flat[i + 1].imshow(aug_img.astype(np.uint8))
            else:
                axes_flat[i + 1].imshow(aug_img, cmap='gray')
            axes_flat[i + 1].set_title(aug_name)
            axes_flat[i + 1].axis('off')
        
        # Hide unused subplots
        for i in range(n_augmentations + 1, len(axes_flat)):
            axes_flat[i].axis('off')
        
        plt.suptitle('Augmentation Sequence', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def visualize_augmentation_grid(self,
                                   original: np.ndarray,
                                   augmentation_types: Dict[str, List[np.ndarray]],
                                   save_path: Optional[str] = None) -> None:
        """
        Visualize different types of augmentations in a grid
        
        Args:
            original: Original image
            augmentation_types: Dictionary mapping augmentation type to list of augmented images
            save_path: Optional path to save figure
        """
        n_types = len(augmentation_types)
        max_samples = max(len(samples) for samples in augmentation_types.values())
        
        fig = plt.figure(figsize=(3 * (max_samples + 1), 3 * n_types))
        gs = gridspec.GridSpec(n_types, max_samples + 1, figure=fig)
        
        for row, (aug_type, samples) in enumerate(augmentation_types.items()):
            # Show original in first column
            ax = fig.add_subplot(gs[row, 0])
            if len(original.shape) == 3:
                ax.imshow(original.astype(np.uint8))
            else:
                ax.imshow(original, cmap='gray')
            
            if row == 0:
                ax.set_title('Original', fontweight='bold')
            ax.set_ylabel(aug_type, fontweight='bold', rotation=0, ha='right', va='center')
            ax.axis('off')
            
            # Show augmented samples
            for col, sample in enumerate(samples):
                ax = fig.add_subplot(gs[row, col + 1])
                if len(sample.shape) == 3:
                    ax.imshow(sample.astype(np.uint8))
                else:
                    ax.imshow(sample, cmap='gray')
                
                if row == 0:
                    ax.set_title(f'Sample {col + 1}')
                ax.axis('off')
        
        plt.suptitle('Augmentation Types Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def visualize_mask_augmentation(self,
                                   original_image: np.ndarray,
                                   original_mask: np.ndarray,
                                   augmented_image: np.ndarray,
                                   augmented_mask: np.ndarray,
                                   alpha: float = 0.5,
                                   save_path: Optional[str] = None) -> None:
        """
        Visualize image and mask augmentation with overlay
        
        Args:
            original_image: Original image
            original_mask: Original segmentation mask
            augmented_image: Augmented image
            augmented_mask: Augmented segmentation mask
            alpha: Overlay transparency
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        if len(original_image.shape) == 3:
            axes[0, 0].imshow(original_image.astype(np.uint8))
        else:
            axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Original mask
        axes[0, 1].imshow(original_mask, cmap='tab20')
        axes[0, 1].set_title('Original Mask')
        axes[0, 1].axis('off')
        
        # Original overlay
        overlay_orig = self._create_overlay(original_image, original_mask, alpha)
        axes[0, 2].imshow(overlay_orig)
        axes[0, 2].set_title('Original Overlay')
        axes[0, 2].axis('off')
        
        # Augmented image
        if len(augmented_image.shape) == 3:
            axes[1, 0].imshow(augmented_image.astype(np.uint8))
        else:
            axes[1, 0].imshow(augmented_image, cmap='gray')
        axes[1, 0].set_title('Augmented Image')
        axes[1, 0].axis('off')
        
        # Augmented mask
        axes[1, 1].imshow(augmented_mask, cmap='tab20')
        axes[1, 1].set_title('Augmented Mask')
        axes[1, 1].axis('off')
        
        # Augmented overlay
        overlay_aug = self._create_overlay(augmented_image, augmented_mask, alpha)
        axes[1, 2].imshow(overlay_aug)
        axes[1, 2].set_title('Augmented Overlay')
        axes[1, 2].axis('off')
        
        plt.suptitle('Image and Mask Augmentation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def visualize_histogram_comparison(self,
                                      original: np.ndarray,
                                      augmented: np.ndarray,
                                      save_path: Optional[str] = None) -> None:
        """
        Compare histograms of original and augmented images
        
        Args:
            original: Original image
            augmented: Augmented image
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Images
        if len(original.shape) == 3:
            axes[0, 0].imshow(original.astype(np.uint8))
            axes[1, 0].imshow(augmented.astype(np.uint8))
        else:
            axes[0, 0].imshow(original, cmap='gray')
            axes[1, 0].imshow(augmented, cmap='gray')
        
        axes[0, 0].set_title('Original Image')
        axes[1, 0].set_title('Augmented Image')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Histograms
        if len(original.shape) == 3:
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                if i < original.shape[2]:
                    # Original histogram
                    hist_orig = cv2.calcHist([original], [i], None, [256], [0, 256])
                    axes[0, 1].plot(hist_orig, color=color, alpha=0.7, label=color.capitalize())
                    
                    # Augmented histogram
                    hist_aug = cv2.calcHist([augmented], [i], None, [256], [0, 256])
                    axes[1, 1].plot(hist_aug, color=color, alpha=0.7, label=color.capitalize())
        else:
            # Grayscale histograms
            hist_orig = cv2.calcHist([original], [0], None, [256], [0, 256])
            axes[0, 1].plot(hist_orig, color='gray')
            
            hist_aug = cv2.calcHist([augmented], [0], None, [256], [0, 256])
            axes[1, 1].plot(hist_aug, color='gray')
        
        axes[0, 1].set_title('Original Histogram')
        axes[1, 1].set_title('Augmented Histogram')
        axes[0, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[1, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[1, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Statistics
        self._plot_statistics(original, axes[0, 2], 'Original Statistics')
        self._plot_statistics(augmented, axes[1, 2], 'Augmented Statistics')
        
        plt.suptitle('Histogram and Statistics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def create_augmentation_report(self,
                                   original: np.ndarray,
                                   augmentations: Dict[str, np.ndarray],
                                   params: Dict[str, Any],
                                   save_dir: Optional[str] = None) -> None:
        """
        Create a comprehensive augmentation report
        
        Args:
            original: Original image
            augmentations: Dictionary of augmented images
            params: Augmentation parameters
            save_dir: Directory to save report
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots for all augmentations
        n_augmentations = len(augmentations)
        n_cols = 4
        n_rows = (n_augmentations + n_cols) // n_cols
        
        fig = plt.figure(figsize=(20, 5 * n_rows))
        
        # Add original image
        ax = plt.subplot(n_rows, n_cols, 1)
        if len(original.shape) == 3:
            ax.imshow(original.astype(np.uint8))
        else:
            ax.imshow(original, cmap='gray')
        ax.set_title('Original', fontweight='bold', fontsize=10)
        ax.axis('off')
        
        # Add augmented images
        for idx, (aug_name, aug_img) in enumerate(augmentations.items()):
            ax = plt.subplot(n_rows, n_cols, idx + 2)
            if len(aug_img.shape) == 3:
                ax.imshow(aug_img.astype(np.uint8))
            else:
                ax.imshow(aug_img, cmap='gray')
            
            # Add parameters to title if available
            title = aug_name
            if aug_name in params:
                param_str = self._format_params(params[aug_name])
                title += f'\n{param_str}'
            
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        
        plt.suptitle('Augmentation Report', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / 'augmentation_report.png', dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
        """
        Create overlay of mask on image
        
        Args:
            image: Input image
            mask: Segmentation mask
            alpha: Overlay transparency
        
        Returns:
            Overlay image
        """
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Create colored mask
        mask_colored = np.zeros_like(image)
        unique_values = np.unique(mask)
        
        # Generate colors for each class
        np.random.seed(42)  # For consistent colors
        colors = np.random.randint(0, 255, (len(unique_values), 3))
        
        for i, value in enumerate(unique_values):
            if value > 0:  # Skip background
                mask_colored[mask == value] = colors[i]
        
        # Create overlay
        overlay = cv2.addWeighted(image.astype(np.uint8), 1 - alpha, 
                                 mask_colored.astype(np.uint8), alpha, 0)
        
        return overlay
    
    def _plot_statistics(self, image: np.ndarray, ax: plt.Axes, title: str) -> None:
        """
        Plot image statistics
        
        Args:
            image: Input image
            ax: Matplotlib axis
            title: Plot title
        """
        stats_text = []
        
        if len(image.shape) == 3:
            channels = ['Red', 'Green', 'Blue']
            for i, channel in enumerate(channels):
                if i < image.shape[2]:
                    channel_data = image[:, :, i]
                    stats_text.append(f'{channel}:')
                    stats_text.append(f'  Mean: {np.mean(channel_data):.2f}')
                    stats_text.append(f'  Std: {np.std(channel_data):.2f}')
                    stats_text.append(f'  Min: {np.min(channel_data):.0f}')
                    stats_text.append(f'  Max: {np.max(channel_data):.0f}')
                    stats_text.append('')
        else:
            stats_text.append(f'Mean: {np.mean(image):.2f}')
            stats_text.append(f'Std: {np.std(image):.2f}')
            stats_text.append(f'Min: {np.min(image):.0f}')
            stats_text.append(f'Max: {np.max(image):.0f}')
        
        ax.text(0.1, 0.5, '\n'.join(stats_text), 
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='center',
               fontfamily='monospace')
        ax.set_title(title)
        ax.axis('off')
    
    def _format_params(self, params: Dict[str, Any]) -> str:
        """
        Format parameters for display
        
        Args:
            params: Parameter dictionary
        
        Returns:
            Formatted string
        """
        param_strs = []
        for key, value in params.items():
            if isinstance(value, float):
                param_strs.append(f'{key}={value:.2f}')
            else:
                param_strs.append(f'{key}={value}')
        
        return ', '.join(param_strs)
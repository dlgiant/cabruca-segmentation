#!/usr/bin/env python3
"""
Example script for running SAM-based tree segmentation on satellite/drone imagery.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import logging

from src.models.sam_model import SAMConfig, SAMTreeSegmenter


def load_config(config_path: str = "configs/sam_config.json") -> Dict:
    """Load SAM configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def visualize_masks(image: np.ndarray, masks: List[Dict], 
                   classifications: List[str], 
                   output_path: Optional[str] = None):
    """
    Visualize segmentation masks with classifications.
    
    Args:
        image: Original image
        masks: List of mask dictionaries
        classifications: List of tree classifications
        output_path: Optional path to save visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # All masks overlay
    overlay = image.copy()
    mask_overlay = np.zeros_like(image)
    
    for i, (mask_data, classification) in enumerate(zip(masks, classifications)):
        if isinstance(mask_data, dict):
            mask = mask_data.get('segmentation', mask_data)
        else:
            mask = mask_data
        
        # Color based on classification
        if classification == 'cacao':
            color = [255, 0, 0]  # Red for cacao
        elif classification == 'shade':
            color = [0, 255, 0]  # Green for shade
        else:
            color = [128, 128, 128]  # Gray for unknown
        
        mask_overlay[mask > 0] = color
    
    # Blend with original
    alpha = 0.5
    overlay = cv2.addWeighted(image, 1-alpha, mask_overlay, alpha, 0)
    
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title("All Trees Segmented")
    axes[0, 1].axis('off')
    
    # Cacao trees only
    cacao_overlay = image.copy()
    cacao_mask = np.zeros_like(image)
    cacao_count = 0
    
    for mask_data, classification in zip(masks, classifications):
        if classification == 'cacao':
            if isinstance(mask_data, dict):
                mask = mask_data.get('segmentation', mask_data)
            else:
                mask = mask_data
            cacao_mask[mask > 0] = [255, 0, 0]
            cacao_count += 1
    
    cacao_overlay = cv2.addWeighted(image, 1-alpha, cacao_mask, alpha, 0)
    axes[1, 0].imshow(cacao_overlay)
    axes[1, 0].set_title(f"Cacao Trees ({cacao_count})")
    axes[1, 0].axis('off')
    
    # Shade trees only
    shade_overlay = image.copy()
    shade_mask = np.zeros_like(image)
    shade_count = 0
    
    for mask_data, classification in zip(masks, classifications):
        if classification == 'shade':
            if isinstance(mask_data, dict):
                mask = mask_data.get('segmentation', mask_data)
            else:
                mask = mask_data
            shade_mask[mask > 0] = [0, 255, 0]
            shade_count += 1
    
    shade_overlay = cv2.addWeighted(image, 1-alpha, shade_mask, alpha, 0)
    axes[1, 1].imshow(shade_overlay)
    axes[1, 1].set_title(f"Shade Trees ({shade_count})")
    axes[1, 1].axis('off')
    
    plt.suptitle(f"Tree Segmentation Results - Total: {len(masks)} trees")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()


def segment_image_with_points(segmenter: SAMTreeSegmenter, 
                             image: np.ndarray,
                             interactive: bool = False) -> Dict[str, Any]:
    """
    Segment image using point prompts.
    
    Args:
        segmenter: SAM tree segmenter instance
        image: Input image
        interactive: Whether to allow interactive point selection
    
    Returns:
        Segmentation results
    """
    if interactive:
        print("Click on the image to select tree centers. Press 'q' to finish.")
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(param['display'], (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Select Trees", param['display'])
        
        display_image = image.copy()
        cv2.namedWindow("Select Trees")
        cv2.setMouseCallback("Select Trees", mouse_callback, {'display': display_image})
        cv2.imshow("Select Trees", display_image)
        
        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        if not points:
            print("No points selected, using automatic generation.")
            points = segmenter.prompt_engineer.generate_point_prompts(image)
    else:
        # Automatic point generation
        points = segmenter.prompt_engineer.generate_point_prompts(image, tree_type='cacao')
        points.extend(segmenter.prompt_engineer.generate_point_prompts(image, tree_type='shade'))
    
    print(f"Using {len(points)} point prompts")
    
    # Perform segmentation
    results = segmenter.segment_with_points(image, points)
    return results


def segment_image_with_boxes(segmenter: SAMTreeSegmenter,
                            image: np.ndarray) -> Dict[str, Any]:
    """
    Segment image using bounding box prompts.
    
    Args:
        segmenter: SAM tree segmenter instance
        image: Input image
    
    Returns:
        Segmentation results
    """
    # Generate box prompts
    boxes = segmenter.prompt_engineer.generate_box_prompts(image)
    
    print(f"Generated {len(boxes)} bounding box prompts")
    
    if boxes:
        # Perform segmentation
        results = segmenter.segment_with_boxes(image, boxes)
        return results
    else:
        print("No valid bounding boxes generated")
        return {'masks': [], 'classifications': []}


def segment_image_automatic(segmenter: SAMTreeSegmenter,
                           image: np.ndarray) -> Dict[str, Any]:
    """
    Perform automatic segmentation without prompts.
    
    Args:
        segmenter: SAM tree segmenter instance
        image: Input image
    
    Returns:
        Segmentation results
    """
    print("Performing automatic segmentation...")
    results = segmenter.segment_automatic(image)
    
    print(f"Detected {results['total_trees']} trees:")
    print(f"  - Cacao trees: {results['cacao_count']}")
    print(f"  - Shade trees: {results['shade_count']}")
    
    return results


def process_directory(segmenter: SAMTreeSegmenter,
                     input_dir: Path,
                     output_dir: Path,
                     mode: str = 'automatic'):
    """
    Process all images in a directory.
    
    Args:
        segmenter: SAM tree segmenter instance
        input_dir: Input directory with images
        output_dir: Output directory for results
        mode: Segmentation mode ('automatic', 'points', 'boxes')
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images to process")
    
    results_summary = []
    
    for image_path in image_files:
        print(f"\nProcessing {image_path.name}...")
        
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform segmentation based on mode
        if mode == 'automatic':
            results = segment_image_automatic(segmenter, image)
        elif mode == 'points':
            results = segment_image_with_points(segmenter, image, interactive=False)
        elif mode == 'boxes':
            results = segment_image_with_boxes(segmenter, image)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Save results
        output_path = output_dir / f"{image_path.stem}_segmented.png"
        
        if 'masks' in results and results['masks']:
            visualize_masks(
                image, 
                results['masks'], 
                results.get('classifications', []),
                output_path=str(output_path)
            )
        
        # Save metadata
        metadata = {
            'image': image_path.name,
            'total_trees': len(results.get('masks', [])),
            'cacao_count': results.get('cacao_count', 0),
            'shade_count': results.get('shade_count', 0)
        }
        results_summary.append(metadata)
    
    # Save summary
    summary_path = output_dir / 'segmentation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nProcessing complete! Results saved to {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run SAM tree segmentation')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--input-dir', type=str, help='Path to input directory')
    parser.add_argument('--output-dir', type=str, default='output/sam_results',
                       help='Path to output directory')
    parser.add_argument('--mode', choices=['automatic', 'points', 'boxes'],
                       default='automatic', help='Segmentation mode')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive point selection')
    parser.add_argument('--config', type=str, default='configs/sam_config.json',
                       help='Path to configuration file')
    parser.add_argument('--model-type', choices=['vit_b', 'vit_l', 'vit_h'],
                       default='vit_b', help='SAM model variant')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    if Path(args.config).exists():
        config_dict = load_config(args.config)
        model_type = config_dict['model']['type']
        checkpoint_path = config_dict['model']['checkpoint']
    else:
        model_type = args.model_type
        checkpoint_path = args.checkpoint
    
    # Create SAM configuration
    config = SAMConfig(
        model_type=model_type,
        checkpoint_path=checkpoint_path
    )
    
    # Initialize segmenter
    print("Initializing SAM tree segmenter...")
    segmenter = SAMTreeSegmenter(config)
    
    # Load model
    print("Loading model...")
    segmenter.load_model(checkpoint_path=checkpoint_path)
    
    # Process based on input
    if args.image:
        # Process single image
        print(f"Processing image: {args.image}")
        
        image = cv2.imread(args.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform segmentation
        if args.mode == 'automatic':
            results = segment_image_automatic(segmenter, image)
        elif args.mode == 'points':
            results = segment_image_with_points(segmenter, image, args.interactive)
        elif args.mode == 'boxes':
            results = segment_image_with_boxes(segmenter, image)
        
        # Visualize results
        if 'masks' in results and results['masks']:
            output_path = Path(args.output_dir) / "segmentation_result.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            visualize_masks(
                image,
                results['masks'],
                results.get('classifications', []),
                output_path=str(output_path)
            )
    
    elif args.input_dir:
        # Process directory
        process_directory(
            segmenter,
            Path(args.input_dir),
            Path(args.output_dir),
            mode=args.mode
        )
    
    else:
        print("Please provide either --image or --input-dir")
        return
    
    print("\nSegmentation complete!")


if __name__ == "__main__":
    main()

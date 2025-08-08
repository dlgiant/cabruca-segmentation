"""
Dataset Splitter with Class Balancing
Splits COCO dataset into train/validation/test sets with stratification and handles class imbalance
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import defaultdict, Counter
import shutil
import logging
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetSplitter:
    """Split COCO dataset with stratification and class balancing"""
    
    def __init__(self, coco_json_path: str, seed: int = 42):
        """
        Initialize dataset splitter
        
        Args:
            coco_json_path: Path to COCO format JSON file
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Load COCO dataset
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = self.coco_data['categories']
        
        # Create mappings
        self.image_id_to_annotations = defaultdict(list)
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.categories}
        
        # Group annotations by image
        for ann in self.annotations:
            self.image_id_to_annotations[ann['image_id']].append(ann)
        
        logger.info(f"Loaded dataset with {len(self.images)} images and {len(self.annotations)} annotations")
    
    def analyze_class_distribution(self) -> Dict[str, Any]:
        """Analyze class distribution in the dataset"""
        # Count annotations per class
        class_counts = Counter()
        image_class_counts = defaultdict(lambda: defaultdict(int))
        
        for ann in self.annotations:
            class_name = self.category_id_to_name[ann['category_id']]
            class_counts[class_name] += 1
            image_class_counts[ann['image_id']][class_name] += 1
        
        # Calculate statistics
        total_annotations = sum(class_counts.values())
        class_percentages = {
            cls: (count / total_annotations * 100) if total_annotations > 0 else 0
            for cls, count in class_counts.items()
        }
        
        # Images per class
        images_per_class = defaultdict(int)
        for image_id, class_dict in image_class_counts.items():
            for class_name in class_dict.keys():
                images_per_class[class_name] += 1
        
        # Calculate imbalance ratio
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        else:
            imbalance_ratio = 0
        
        stats = {
            'class_counts': dict(class_counts),
            'class_percentages': class_percentages,
            'images_per_class': dict(images_per_class),
            'total_annotations': total_annotations,
            'total_images': len(self.images),
            'imbalance_ratio': imbalance_ratio,
            'avg_annotations_per_image': total_annotations / len(self.images) if self.images else 0
        }
        
        return stats
    
    def stratified_split(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                        test_ratio: float = 0.15) -> Tuple[List, List, List]:
        """
        Perform stratified split based on dominant class per image
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
        
        Returns:
            Tuple of (train_images, val_images, test_images)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"
        
        # Determine dominant class for each image
        image_dominant_class = {}
        for img in self.images:
            img_id = img['id']
            if img_id in self.image_id_to_annotations:
                class_counts = Counter()
                for ann in self.image_id_to_annotations[img_id]:
                    class_counts[ann['category_id']] += 1
                
                if class_counts:
                    dominant_class = class_counts.most_common(1)[0][0]
                    image_dominant_class[img_id] = dominant_class
                else:
                    image_dominant_class[img_id] = 0  # Default to background
            else:
                image_dominant_class[img_id] = 0  # No annotations, treat as background
        
        # Prepare data for stratified split
        image_ids = list(image_dominant_class.keys())
        labels = list(image_dominant_class.values())
        
        # First split: train and temp (val + test)
        train_ids, temp_ids, train_labels, temp_labels = train_test_split(
            image_ids, labels, 
            test_size=(val_ratio + test_ratio),
            stratify=labels,
            random_state=self.seed
        )
        
        # Second split: val and test
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=val_test_ratio,
            stratify=temp_labels,
            random_state=self.seed
        )
        
        # Get full image objects
        id_to_image = {img['id']: img for img in self.images}
        train_images = [id_to_image[img_id] for img_id in train_ids]
        val_images = [id_to_image[img_id] for img_id in val_ids]
        test_images = [id_to_image[img_id] for img_id in test_ids]
        
        logger.info(f"Split: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
        
        return train_images, val_images, test_images
    
    def balance_classes(self, images: List[Dict], strategy: str = 'oversample',
                       target_ratio: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        Balance classes in the dataset
        
        Args:
            images: List of image dictionaries
            strategy: 'oversample', 'undersample', 'mixed', or 'augment'
            target_ratio: Target ratio for each class (optional)
        
        Returns:
            Balanced list of images with adjusted annotations
        """
        if strategy not in ['oversample', 'undersample', 'mixed', 'augment']:
            raise ValueError(f"Unknown balancing strategy: {strategy}")
        
        # Get current class distribution
        class_counts = defaultdict(int)
        image_annotations = []
        
        for img in images:
            img_id = img['id']
            anns = self.image_id_to_annotations[img_id]
            
            for ann in anns:
                class_counts[ann['category_id']] += 1
            
            if anns:
                # Store image with its dominant class
                dominant_class = Counter(ann['category_id'] for ann in anns).most_common(1)[0][0]
                image_annotations.append((img_id, dominant_class))
        
        if not image_annotations:
            return images
        
        # Prepare data for resampling
        X = np.array([item[0] for item in image_annotations]).reshape(-1, 1)
        y = np.array([item[1] for item in image_annotations])
        
        # Apply balancing strategy
        if strategy == 'oversample':
            sampler = RandomOverSampler(random_state=self.seed)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
        elif strategy == 'undersample':
            sampler = RandomUnderSampler(random_state=self.seed)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
        elif strategy == 'mixed':
            # First oversample minority classes to median
            class_counts_list = list(class_counts.values())
            median_count = np.median(class_counts_list)
            
            # Calculate sampling strategy
            sampling_strategy = {}
            for class_id, count in class_counts.items():
                if count < median_count:
                    sampling_strategy[class_id] = int(median_count)
            
            if sampling_strategy:
                oversampler = RandomOverSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=self.seed
                )
                X_temp, y_temp = oversampler.fit_resample(X, y)
            else:
                X_temp, y_temp = X, y
            
            # Then undersample majority classes
            undersampler = RandomUnderSampler(random_state=self.seed)
            X_resampled, y_resampled = undersampler.fit_resample(X_temp, y_temp)
            
        elif strategy == 'augment':
            # This would require actual data augmentation
            # For now, we'll use oversampling as a placeholder
            logger.warning("Augmentation strategy not fully implemented, using oversampling")
            sampler = RandomOverSampler(random_state=self.seed)
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Convert back to image list
        id_to_image = {img['id']: img for img in images}
        balanced_images = []
        
        for img_id in X_resampled.flatten():
            if img_id in id_to_image:
                balanced_images.append(id_to_image[img_id])
        
        logger.info(f"Balanced dataset from {len(images)} to {len(balanced_images)} images")
        
        return balanced_images
    
    def create_split_datasets(self, output_dir: str, 
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15,
                             balance_strategy: Optional[str] = None) -> Dict[str, str]:
        """
        Create train/val/test splits and save as separate COCO files
        
        Args:
            output_dir: Directory to save split datasets
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            balance_strategy: Strategy for class balancing (optional)
        
        Returns:
            Dictionary with paths to created files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Perform stratified split
        train_images, val_images, test_images = self.stratified_split(
            train_ratio, val_ratio, test_ratio
        )
        
        # Apply class balancing if requested
        if balance_strategy:
            logger.info(f"Applying {balance_strategy} balancing to training set")
            train_images = self.balance_classes(train_images, strategy=balance_strategy)
        
        # Create split datasets
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        output_paths = {}
        
        for split_name, split_images in splits.items():
            # Get image IDs for this split
            image_ids = set(img['id'] for img in split_images)
            
            # Filter annotations for this split
            split_annotations = [
                ann for ann in self.annotations 
                if ann['image_id'] in image_ids
            ]
            
            # Create COCO dataset for this split
            split_dataset = {
                'info': self.coco_data.get('info', {}),
                'licenses': self.coco_data.get('licenses', []),
                'categories': self.categories,
                'images': split_images,
                'annotations': split_annotations
            }
            
            # Update annotation IDs to be sequential
            for i, ann in enumerate(split_dataset['annotations']):
                ann['id'] = i + 1
            
            # Save to file
            output_path = output_dir / f'{split_name}_coco.json'
            with open(output_path, 'w') as f:
                json.dump(split_dataset, f, indent=2)
            
            output_paths[split_name] = str(output_path)
            
            # Log statistics
            split_stats = self._calculate_split_stats(split_dataset)
            logger.info(f"{split_name.upper()} split: {split_stats}")
        
        return output_paths
    
    def _calculate_split_stats(self, dataset: Dict) -> Dict:
        """Calculate statistics for a split dataset"""
        class_counts = Counter()
        for ann in dataset['annotations']:
            class_name = self.category_id_to_name[ann['category_id']]
            class_counts[class_name] += 1
        
        return {
            'images': len(dataset['images']),
            'annotations': len(dataset['annotations']),
            'class_distribution': dict(class_counts)
        }
    
    def visualize_class_distribution(self, save_path: Optional[str] = None):
        """Visualize class distribution across splits"""
        stats = self.analyze_class_distribution()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Overall class distribution
        ax = axes[0, 0]
        classes = list(stats['class_counts'].keys())
        counts = list(stats['class_counts'].values())
        colors = ['#808080', '#00FF00', '#FF0000']  # Gray, Green, Red for bg, cacao, shade
        
        ax.bar(classes, counts, color=colors[:len(classes)])
        ax.set_title('Overall Class Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Annotations')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. Class percentage pie chart
        ax = axes[0, 1]
        percentages = list(stats['class_percentages'].values())
        ax.pie(percentages, labels=classes, colors=colors[:len(classes)], 
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Class Distribution (%)')
        
        # 3. Images per class
        ax = axes[1, 0]
        image_counts = list(stats['images_per_class'].values())
        ax.bar(classes, image_counts, color=colors[:len(classes)])
        ax.set_title('Images per Class')
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Images')
        ax.tick_params(axis='x', rotation=45)
        
        # 4. Statistics text
        ax = axes[1, 1]
        ax.axis('off')
        stats_text = f"""Dataset Statistics:
        
Total Images: {stats['total_images']}
Total Annotations: {stats['total_annotations']}
Avg Annotations/Image: {stats['avg_annotations_per_image']:.2f}
Class Imbalance Ratio: {stats['imbalance_ratio']:.2f}

Class Counts:
"""
        for cls, count in stats['class_counts'].items():
            stats_text += f"  {cls}: {count} ({stats['class_percentages'][cls]:.1f}%)\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Dataset Class Distribution Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def create_balanced_subset(self, max_samples_per_class: int = 100,
                              output_path: Optional[str] = None) -> Dict:
        """
        Create a balanced subset of the dataset for initial training
        
        Args:
            max_samples_per_class: Maximum annotations per class
            output_path: Path to save the subset (optional)
        
        Returns:
            Balanced subset as COCO dictionary
        """
        # Group annotations by class
        class_annotations = defaultdict(list)
        for ann in self.annotations:
            class_annotations[ann['category_id']].append(ann)
        
        # Sample annotations
        balanced_annotations = []
        for class_id, anns in class_annotations.items():
            sampled = random.sample(anns, min(len(anns), max_samples_per_class))
            balanced_annotations.extend(sampled)
        
        # Get unique image IDs
        image_ids = set(ann['image_id'] for ann in balanced_annotations)
        
        # Filter images
        balanced_images = [img for img in self.images if img['id'] in image_ids]
        
        # Create balanced dataset
        balanced_dataset = {
            'info': self.coco_data.get('info', {}),
            'licenses': self.coco_data.get('licenses', []),
            'categories': self.categories,
            'images': balanced_images,
            'annotations': balanced_annotations
        }
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(balanced_dataset, f, indent=2)
            logger.info(f"Saved balanced subset to {output_path}")
        
        return balanced_dataset


def main():
    """Example usage of dataset splitter"""
    
    # Example usage
    # splitter = DatasetSplitter("data/processed/annotations_coco.json")
    
    # Analyze class distribution
    # stats = splitter.analyze_class_distribution()
    # print("Class Distribution:")
    # for key, value in stats.items():
    #     print(f"  {key}: {value}")
    
    # Visualize distribution
    # splitter.visualize_class_distribution("data/processed/class_distribution.png")
    
    # Create splits with balancing
    # output_paths = splitter.create_split_datasets(
    #     output_dir="data/processed/splits",
    #     train_ratio=0.7,
    #     val_ratio=0.15,
    #     test_ratio=0.15,
    #     balance_strategy="mixed"  # or "oversample", "undersample", None
    # )
    
    # Create balanced subset for initial training
    # balanced_subset = splitter.create_balanced_subset(
    #     max_samples_per_class=100,
    #     output_path="data/processed/balanced_subset.json"
    # )
    
    logger.info("Dataset splitting setup complete")


if __name__ == "__main__":
    main()

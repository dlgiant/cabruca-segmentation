"""
Advanced training pipeline for Cabruca segmentation with experiment tracking.
Supports MLflow, Weights & Biases, and memory-efficient training on macOS.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, 
    CosineAnnealingWarmRestarts, ExponentialLR
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict
import time
import psutil
import platform
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Experiment tracking
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Install with: pip install mlflow")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. Install with: pip install wandb")

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cabruca_segmentation_model import CabrucaSegmentationModel, CabrucaLoss, create_cabruca_model
from data_processing.cabruca_dataset import create_data_loaders
from evaluation.agroforestry_metrics import AgroforestryMetrics


class ExperimentTracker:
    """
    Unified experiment tracking interface for MLflow, W&B, and TensorBoard.
    """
    
    def __init__(self, config: Dict, experiment_name: str = None):
        """
        Initialize experiment tracking.
        
        Args:
            config: Training configuration
            experiment_name: Name for the experiment
        """
        self.config = config
        self.use_mlflow = config.get('tracking', {}).get('mlflow', False) and MLFLOW_AVAILABLE
        self.use_wandb = config.get('tracking', {}).get('wandb', False) and WANDB_AVAILABLE
        self.use_tensorboard = config.get('tracking', {}).get('tensorboard', True)
        
        # Generate experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"cabruca_{timestamp}"
        self.experiment_name = experiment_name
        
        # Initialize trackers
        self._init_mlflow()
        self._init_wandb()
        self._init_tensorboard()
    
    def _init_mlflow(self):
        """Initialize MLflow tracking."""
        if self.use_mlflow:
            mlflow.set_tracking_uri(self.config.get('tracking', {}).get('mlflow_uri', 'file:./mlruns'))
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run()
            
            # Log parameters
            mlflow.log_params(self._flatten_dict(self.config))
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        if self.use_wandb:
            wandb.init(
                project=self.config.get('tracking', {}).get('wandb_project', 'cabruca-segmentation'),
                name=self.experiment_name,
                config=self.config,
                resume='allow'
            )
    
    def _init_tensorboard(self):
        """Initialize TensorBoard logging."""
        if self.use_tensorboard:
            log_dir = Path(self.config.get('output_dir', 'outputs')) / self.experiment_name / 'tensorboard'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir)
        else:
            self.tb_writer = None
    
    def log_metrics(self, metrics: Dict, step: int = None):
        """Log metrics to all active trackers."""
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
    
    def log_model(self, model, optimizer=None):
        """Log model to trackers."""
        if self.use_mlflow:
            mlflow.pytorch.log_model(model, "model")
        
        if self.use_wandb:
            wandb.save('*.pth')
    
    def log_artifact(self, file_path: str, artifact_type: str = None):
        """Log artifact file."""
        if self.use_mlflow:
            mlflow.log_artifact(file_path)
        
        if self.use_wandb:
            wandb.save(file_path)
    
    def finish(self):
        """Close all tracking sessions."""
        if self.use_mlflow:
            mlflow.end_run()
        
        if self.use_wandb:
            wandb.finish()
        
        if self.tb_writer:
            self.tb_writer.close()
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary for MLflow parameters."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class MemoryEfficientTrainer:
    """
    Memory-efficient trainer with support for macOS MPS and gradient accumulation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = self._setup_device()
        
        # Model setup
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # Data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            self.config['data']
        )
        
        # Loss and optimizer
        self.criterion = self._create_loss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Memory optimization
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        self.mixed_precision = config['training'].get('mixed_precision', False)
        
        # Setup mixed precision training
        if self.mixed_precision and self.device.type == 'cuda':
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Experiment tracking
        self.tracker = ExperimentTracker(config)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf') if config['training'].get('minimize_metric', True) else -float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Metrics
        self.evaluator = AgroforestryMetrics()
        self.metrics_history = defaultdict(list)
        
        # Checkpointing
        self.checkpoint_dir = Path(config['output_dir']) / self.tracker.experiment_name / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_device(self) -> torch.device:
        """Setup compute device with MPS support for macOS."""
        device_name = self.config.get('device', 'auto')
        
        if device_name == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                print("Using Apple Metal Performance Shaders (MPS)")
            else:
                device = torch.device('cpu')
                print(f"Using CPU: {platform.processor()}")
        else:
            device = torch.device(device_name)
            print(f"Using specified device: {device}")
        
        # Log system info
        self._log_system_info(device)
        
        return device
    
    def _log_system_info(self, device: torch.device):
        """Log system information."""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'device': str(device),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
        }
        
        if device.type == 'cuda':
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print("\nSystem Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Log to trackers
        self.tracker.log_metrics({f'system/{k}': v for k, v in info.items() if isinstance(v, (int, float))})
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        model_config = self.config['model']
        model = create_cabruca_model(model_config)
        
        # Load pretrained weights if specified
        if 'pretrained_checkpoint' in model_config:
            checkpoint_path = model_config['pretrained_checkpoint']
            if os.path.exists(checkpoint_path):
                print(f"Loading pretrained weights from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
        
        return model
    
    def _create_loss(self) -> nn.Module:
        """Create loss function."""
        loss_config = self.config.get('loss', {})
        return CabrucaLoss(
            instance_weight=loss_config.get('instance_weight', 1.0),
            semantic_weight=loss_config.get('semantic_weight', 1.0),
            crown_weight=loss_config.get('crown_weight', 0.5),
            density_weight=loss_config.get('density_weight', 0.5)
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        opt_config = self.config['optimizer']
        opt_type = opt_config['type'].lower()
        
        # Get parameters with different learning rates if specified
        if 'param_groups' in opt_config:
            param_groups = self._get_param_groups(opt_config['param_groups'])
        else:
            param_groups = self.model.parameters()
        
        # Create optimizer
        if opt_type == 'adam':
            optimizer = optim.Adam(
                param_groups,
                lr=opt_config['lr'],
                betas=opt_config.get('betas', (0.9, 0.999)),
                weight_decay=opt_config.get('weight_decay', 0.0001)
            )
        elif opt_type == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                lr=opt_config['lr'],
                betas=opt_config.get('betas', (0.9, 0.999)),
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        elif opt_type == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 0.0001),
                nesterov=opt_config.get('nesterov', True)
            )
        elif opt_type == 'rmsprop':
            optimizer = optim.RMSprop(
                param_groups,
                lr=opt_config['lr'],
                alpha=opt_config.get('alpha', 0.99),
                weight_decay=opt_config.get('weight_decay', 0.0001)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
        
        return optimizer
    
    def _get_param_groups(self, group_config: List[Dict]) -> List[Dict]:
        """Get parameter groups with different learning rates."""
        param_groups = []
        
        for group in group_config:
            if group['name'] == 'backbone':
                params = self.model.instance_head.backbone.parameters()
            elif group['name'] == 'heads':
                params = list(self.model.instance_head.roi_heads.parameters()) + \
                        list(self.model.semantic_head.classifier.parameters())
            elif group['name'] == 'custom':
                params = list(self.model.crown_estimator.parameters()) + \
                        list(self.model.density_estimator.parameters())
            else:
                continue
            
            param_groups.append({
                'params': params,
                'lr': group['lr'],
                'weight_decay': group.get('weight_decay', 0.0001)
            })
        
        return param_groups
    
    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler."""
        if 'scheduler' not in self.config:
            return None
        
        sched_config = self.config['scheduler']
        sched_type = sched_config['type'].lower()
        
        if sched_type == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get('T_max', self.config['training']['num_epochs']),
                eta_min=sched_config.get('eta_min', 1e-7)
            )
        elif sched_type == 'cosine_warm_restarts':
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config.get('T_0', 10),
                T_mult=sched_config.get('T_mult', 2),
                eta_min=sched_config.get('eta_min', 1e-7)
            )
        elif sched_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min' if self.config['training'].get('minimize_metric', True) else 'max',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 10),
                min_lr=sched_config.get('min_lr', 1e-7),
                verbose=True
            )
        elif sched_type == 'onecycle':
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=sched_config.get('max_lr', self.config['optimizer']['lr'] * 10),
                total_steps=len(self.train_loader) * self.config['training']['num_epochs'],
                pct_start=sched_config.get('pct_start', 0.3)
            )
        elif sched_type == 'exponential':
            scheduler = ExponentialLR(
                self.optimizer,
                gamma=sched_config.get('gamma', 0.95)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with memory-efficient techniques.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = defaultdict(float)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
        
        # Gradient accumulation
        self.optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # Mixed precision training
            if self.scaler:
                with autocast():
                    outputs = self.model(images, targets)
                    loss, loss_dict = self.criterion(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images, targets)
                loss, loss_dict = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config['training'].get('gradient_clip'):
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update learning rate (for OneCycleLR)
                if self.scheduler and isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
            
            # Update metrics
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                epoch_metrics[f'train/{key}_loss'] += value
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            memory_used = self._get_memory_usage()
            pbar.set_postfix({
                'loss': loss.item() * self.gradient_accumulation_steps,
                'lr': f'{current_lr:.2e}',
                'mem': f'{memory_used:.1f}GB'
            })
            
            # Log batch metrics
            if batch_idx % self.config['training'].get('log_interval', 10) == 0:
                batch_metrics = {
                    'train/batch_loss': loss.item() * self.gradient_accumulation_steps,
                    'train/learning_rate': current_lr,
                    'system/memory_gb': memory_used
                }
                self.tracker.log_metrics(batch_metrics, self.global_step)
            
            self.global_step += 1
            
            # Memory cleanup for MPS
            if self.device.type == 'mps' and batch_idx % 10 == 0:
                torch.mps.empty_cache()
        
        # Average epoch metrics
        num_batches = len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return dict(epoch_metrics)
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model with agroforestry metrics.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_metrics = defaultdict(float)
        all_predictions = []
        all_ground_truth = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # Forward pass
                if self.mixed_precision and self.device.type == 'cuda':
                    with autocast():
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                
                # Calculate loss
                loss, loss_dict = self.criterion(outputs, targets)
                
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    val_metrics[f'val/{key}_loss'] += value
                
                # Store for agroforestry metrics
                all_predictions.append(outputs)
                all_ground_truth.extend(targets)
        
        # Average loss metrics
        num_batches = len(self.val_loader)
        for key in list(val_metrics.keys()):
            val_metrics[key] /= num_batches
        
        # Calculate agroforestry-specific metrics
        if all_predictions and all_ground_truth:
            agro_metrics = self._calculate_agroforestry_metrics(
                all_predictions, all_ground_truth
            )
            val_metrics.update({f'val/{k}': v for k, v in agro_metrics.items()})
        
        return dict(val_metrics)
    
    def _calculate_agroforestry_metrics(self, predictions: List, 
                                       ground_truth: List) -> Dict[str, float]:
        """Calculate domain-specific agroforestry metrics."""
        metrics = {}
        
        # Aggregate batch predictions
        # This is simplified - in practice you'd properly aggregate
        if predictions and ground_truth:
            # Use first batch for demonstration
            pred = predictions[0]
            gt = ground_truth[0]
            
            # Convert to proper format for evaluator
            pred_dict = {
                'instances': {
                    'boxes': pred.get('boxes', torch.tensor([])).cpu().numpy(),
                    'labels': pred.get('labels', torch.tensor([])).cpu().numpy(),
                    'scores': pred.get('scores', torch.tensor([])).cpu().numpy()
                }
            }
            
            gt_dict = {
                'instances': {
                    'boxes': gt.get('boxes', torch.tensor([])).cpu().numpy(),
                    'labels': gt.get('labels', torch.tensor([])).cpu().numpy()
                }
            }
            
            # Calculate metrics
            eval_metrics = self.evaluator.evaluate_batch(pred_dict, gt_dict)
            
            # Select key metrics
            key_metrics = [
                'total_count_accuracy',
                'species_classification_accuracy',
                'mAP_50',
                'canopy_iou',
                'shade_distribution_score'
            ]
            
            for metric in key_metrics:
                if metric in eval_metrics:
                    metrics[metric] = eval_metrics[metric]
        
        return metrics
    
    def train(self):
        """
        Main training loop with experiment tracking.
        """
        print("\n" + "="*60)
        print(f" Starting Training: {self.tracker.experiment_name}")
        print("="*60)
        
        num_epochs = self.config['training']['num_epochs']
        early_stopping_patience = self.config['training'].get('early_stopping_patience', 20)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate(epoch)
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics['epoch'] = epoch
            
            # Log metrics
            self.tracker.log_metrics(all_metrics, epoch)
            
            # Store history
            for key, value in all_metrics.items():
                self.metrics_history[key].append(value)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    metric_for_scheduler = val_metrics.get(
                        self.config['training'].get('monitor_metric', 'val/total_loss'),
                        val_metrics.get('val/total_loss', 0)
                    )
                    self.scheduler.step(metric_for_scheduler)
                elif not isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
            
            # Check for improvement
            monitor_metric = self.config['training'].get('monitor_metric', 'val/total_loss')
            current_metric = val_metrics.get(monitor_metric, val_metrics.get('val/total_loss', 0))
            
            is_best = False
            if self.config['training'].get('minimize_metric', True):
                is_best = current_metric < self.best_metric
            else:
                is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint('best')
                print(f"✓ New best {monitor_metric}: {current_metric:.4f}")
            else:
                self.patience_counter += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config['training'].get('save_frequency', 10) == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')
            
            # Save latest checkpoint
            self.save_checkpoint('latest')
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            self._print_epoch_summary(epoch, all_metrics, epoch_time)
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\n✋ Early stopping triggered after {early_stopping_patience} epochs without improvement")
                break
            
            # Memory cleanup
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Training completed
        print("\n" + "="*60)
        print(" Training Completed")
        print("="*60)
        print(f"Best epoch: {self.best_epoch+1}")
        print(f"Best {self.config['training'].get('monitor_metric', 'val/total_loss')}: {self.best_metric:.4f}")
        
        # Save final model
        self.save_checkpoint('final')
        
        # Log model to trackers
        self.tracker.log_model(self.model, self.optimizer)
        
        # Save training history
        self._save_training_history()
        
        # Close tracking
        self.tracker.finish()
    
    def save_checkpoint(self, name: str):
        """
        Save training checkpoint.
        
        Args:
            name: Checkpoint name (e.g., 'best', 'latest', 'epoch_10')
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'config': self.config,
            'metrics_history': dict(self.metrics_history)
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{name}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Log artifact
        self.tracker.log_artifact(str(checkpoint_path))
        
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint for resuming.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint['best_metric']
        self.best_epoch = checkpoint['best_epoch']
        self.metrics_history = defaultdict(list, checkpoint.get('metrics_history', {}))
        
        print(f"✓ Resumed from epoch {self.current_epoch}")
    
    def _print_epoch_summary(self, epoch: int, metrics: Dict, epoch_time: float):
        """Print epoch training summary."""
        print(f"\n📊 Epoch {epoch+1} Summary (Time: {epoch_time:.1f}s)")
        print("-" * 50)
        
        # Key metrics to display
        display_metrics = [
            ('train/total_loss', 'Train Loss'),
            ('val/total_loss', 'Val Loss'),
            ('val/mAP_50', 'mAP@0.5'),
            ('val/total_count_accuracy', 'Tree Count Acc'),
            ('val/canopy_iou', 'Canopy IoU'),
            ('val/shade_distribution_score', 'Shade Score')
        ]
        
        for metric_key, display_name in display_metrics:
            if metric_key in metrics:
                value = metrics[metric_key]
                if 'loss' in metric_key:
                    print(f"  {display_name}: {value:.4f}")
                else:
                    print(f"  {display_name}: {value:.3f}")
        
        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Memory usage
        memory_used = self._get_memory_usage()
        print(f"  Memory Usage: {memory_used:.2f} GB")
        
        # Patience status
        print(f"  Patience: {self.patience_counter}/{self.config['training'].get('early_stopping_patience', 20)}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / (1024**3)
        elif self.device.type == 'mps':
            # MPS doesn't have direct memory query yet
            return psutil.virtual_memory().used / (1024**3)
        else:
            return psutil.Process().memory_info().rss / (1024**3)
    
    def _save_training_history(self):
        """Save training history to JSON file."""
        history_path = self.checkpoint_dir.parent / 'training_history.json'
        
        # Convert defaultdict to regular dict for JSON serialization
        history = dict(self.metrics_history)
        
        # Ensure all values are JSON serializable
        for key in history:
            history[key] = [float(v) if isinstance(v, torch.Tensor) else v for v in history[key]]
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.tracker.log_artifact(str(history_path))
        print(f"📝 Saved training history to {history_path}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML or JSON file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    return config


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train Cabruca Segmentation Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for the experiment')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    
    # Override config parameters
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.device != 'auto':
        config['device'] = args.device
    if args.lr:
        config['optimizer']['lr'] = args.lr
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    
    # Create trainer
    trainer = MemoryEfficientTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
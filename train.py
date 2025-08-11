#!/usr/bin/env python
"""
Main training script for Cabruca Segmentation Model.
Supports multiple training backends and experiment tracking.
"""

import agentops
import argparse
import os
import sys
import yaml
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

from training.advanced_trainer import MemoryEfficientTrainer, load_config


def main():
    # Initialize AgentOps for training session tracking
    agentops.init(auto_start_session=False, tags=["cabruca", "training", "segmentation"])
    
    parser = argparse.ArgumentParser(
        description='Train Cabruca Segmentation Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default config
  python train.py --config configs/training_config.yaml
  
  # Resume training from checkpoint
  python train.py --config configs/training_config.yaml --resume outputs/checkpoint_latest.pth
  
  # Train on CPU with lightweight config
  python train.py --config configs/training_config_light.yaml --device cpu
  
  # Train with experiment tracking
  python train.py --config configs/training_config.yaml --use-mlflow --use-wandb
  
  # Override hyperparameters
  python train.py --config configs/training_config.yaml --lr 0.001 --batch-size 8 --epochs 50
        """
    )
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file (YAML or JSON)')
    
    # Optional arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for the experiment (auto-generated if not provided)')
    
    # Device selection
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training (default: auto)')
    
    # Hyperparameter overrides
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--grad-accum', type=int, default=None,
                       help='Override gradient accumulation steps')
    
    # Experiment tracking
    parser.add_argument('--use-mlflow', action='store_true',
                       help='Enable MLflow tracking')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Enable Weights & Biases tracking')
    parser.add_argument('--no-tensorboard', action='store_true',
                       help='Disable TensorBoard logging')
    
    # Memory optimization
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Enable mixed precision training (CUDA only)')
    parser.add_argument('--compile', action='store_true',
                       help='Compile model with torch.compile (PyTorch 2.0+)')
    
    # Validation
    parser.add_argument('--val-frequency', type=int, default=None,
                       help='Validation frequency (epochs)')
    parser.add_argument('--save-frequency', type=int, default=None,
                       help='Checkpoint save frequency (epochs)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"📖 Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.device != 'auto':
        config['device'] = args.device
    
    if args.lr is not None:
        config['optimizer']['lr'] = args.lr
        print(f"  Override: learning rate = {args.lr}")
    
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
        print(f"  Override: batch size = {args.batch_size}")
    
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
        print(f"  Override: epochs = {args.epochs}")
    
    if args.grad_accum is not None:
        config['training']['gradient_accumulation_steps'] = args.grad_accum
        print(f"  Override: gradient accumulation = {args.grad_accum}")
    
    if args.mixed_precision:
        config['training']['mixed_precision'] = True
        print(f"  Override: mixed precision enabled")
    
    if args.val_frequency is not None:
        config['validation']['frequency'] = args.val_frequency
    
    if args.save_frequency is not None:
        config['training']['save_frequency'] = args.save_frequency
    
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    
    # Configure experiment tracking
    if 'tracking' not in config:
        config['tracking'] = {}
    
    if args.use_mlflow:
        config['tracking']['mlflow'] = True
        print("  Tracking: MLflow enabled")
    
    if args.use_wandb:
        config['tracking']['wandb'] = True
        print("  Tracking: Weights & Biases enabled")
    
    if args.no_tensorboard:
        config['tracking']['tensorboard'] = False
    else:
        config['tracking']['tensorboard'] = True
        print("  Tracking: TensorBoard enabled")
    
    # Display configuration summary
    if not args.quiet:
        print("\n🔧 Configuration Summary:")
        print(f"  Model: {config['model'].get('num_instance_classes', 3)} instance classes, "
              f"{config['model'].get('num_semantic_classes', 6)} semantic classes")
        print(f"  Data: batch_size={config['data']['batch_size']}, "
              f"tile_size={config['data'].get('tile_size', 512)}")
        print(f"  Optimizer: {config['optimizer']['type']}, lr={config['optimizer']['lr']}")
        print(f"  Training: {config['training']['num_epochs']} epochs")
        print(f"  Device: {config.get('device', 'auto')}")
        
        effective_batch = config['data']['batch_size'] * \
                         config['training'].get('gradient_accumulation_steps', 1)
        print(f"  Effective batch size: {effective_batch}")
    
    # Create trainer
    print("\n🚀 Initializing trainer...")
    trainer = MemoryEfficientTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Compile model if requested (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("⚡ Compiling model with torch.compile...")
        trainer.model = torch.compile(trainer.model)
    
    # Start training
    print("\n" + "="*60)
    print(" Starting Training")
    print("="*60)
    
    # Start AgentOps session for this training run
    experiment_name = args.experiment_name or f"cabruca_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tracer = agentops.start_trace(trace_name=experiment_name, tags=["training", "model-training"])
    
    try:
        trainer.train()
        print("\n✅ Training completed successfully!")
        agentops.end_trace(tracer, end_state="Success")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        trainer.save_checkpoint('interrupted')
        trainer.tracker.finish()
        agentops.end_trace(tracer, end_state="Fail")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        trainer.save_checkpoint('error')
        trainer.tracker.finish()
        agentops.end_trace(tracer, end_state="Fail")
        raise
    
    print("\n📊 To monitor training progress, run:")
    print(f"  streamlit run src/training/training_monitor.py")
    
    print("\n📈 To view TensorBoard, run:")
    print(f"  tensorboard --logdir {trainer.checkpoint_dir.parent / 'tensorboard'}")


if __name__ == '__main__':
    import torch
    from datetime import datetime
    
    # Check PyTorch version
    torch_version = torch.__version__
    print(f"🔥 PyTorch version: {torch_version}")
    
    # Check available devices
    if torch.cuda.is_available():
        print(f"🎮 CUDA available: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 Apple MPS available")
    else:
        print("💻 Using CPU")
    
    main()
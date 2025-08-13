"""
Training pipeline for Cabruca multi-class segmentation model.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.cabruca_dataset import create_data_loaders
from models.cabruca_segmentation_model import (
    CabrucaLoss,
    CabrucaSegmentationModel,
    create_cabruca_model,
)


class CabrucaTrainer:
    """
    Trainer class for Cabruca segmentation model.
    """

    def __init__(self, config: Dict):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model
        self.model = create_cabruca_model(config["model"])
        self.model = self.model.to(self.device)

        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            config["data"]
        )

        # Loss function
        self.criterion = CabrucaLoss(
            instance_weight=config["loss"].get("instance_weight", 1.0),
            semantic_weight=config["loss"].get("semantic_weight", 1.0),
            crown_weight=config["loss"].get("crown_weight", 0.5),
            density_weight=config["loss"].get("density_weight", 0.5),
        )

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Metrics tracking
        self.best_val_loss = float("inf")
        self.best_val_metrics = {}
        self.current_epoch = 0

        # Logging
        self.setup_logging()

    def _create_optimizer(self):
        """Create optimizer based on config."""
        opt_config = self.config["optimizer"]
        opt_type = opt_config["type"].lower()

        if opt_type == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config["lr"],
                weight_decay=opt_config.get("weight_decay", 0.0001),
            )
        elif opt_type == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config["lr"],
                weight_decay=opt_config.get("weight_decay", 0.01),
            )
        elif opt_type == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config["lr"],
                momentum=opt_config.get("momentum", 0.9),
                weight_decay=opt_config.get("weight_decay", 0.0001),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        sched_config = self.config["scheduler"]
        sched_type = sched_config["type"].lower()

        if sched_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get("T_max", 50),
                eta_min=sched_config.get("eta_min", 1e-6),
            )
        elif sched_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=sched_config.get("factor", 0.5),
                patience=sched_config.get("patience", 10),
                min_lr=sched_config.get("min_lr", 1e-6),
            )
        else:
            return None

    def setup_logging(self):
        """Setup logging with TensorBoard and optionally W&B."""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(
            self.config["output_dir"], f"cabruca_{timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.output_dir, "tensorboard"))

        # Weights & Biases
        if self.config.get("use_wandb", False):
            wandb.init(
                project="cabruca-segmentation",
                config=self.config,
                name=f"cabruca_{timestamp}",
            )

    def train_epoch(self, epoch: int):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_losses = {"instance": 0, "semantic": 0, "crown": 0, "density": 0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            targets = [
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, targets)

            # Calculate loss
            loss, loss_dict = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config["training"].get("gradient_clip"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["training"]["gradient_clip"]
                )

            self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            for k, v in loss_dict.items():
                if k != "total":
                    epoch_losses[k] += v.item() if isinstance(v, torch.Tensor) else v

            # Update progress bar
            pbar.set_postfix(
                {"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}
            )

            # Log to TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("train/loss", loss.item(), global_step)

            for k, v in loss_dict.items():
                if k != "total":
                    self.writer.add_scalar(
                        f"train/{k}_loss",
                        v.item() if isinstance(v, torch.Tensor) else v,
                        global_step,
                    )

        # Average losses
        avg_loss = epoch_loss / len(self.train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= len(self.train_loader)

        return avg_loss, epoch_losses

    def validate(self, epoch: int):
        """
        Validate the model.

        Args:
            epoch: Current epoch number
        """
        self.model.eval()
        val_loss = 0.0
        val_losses = {"instance": 0, "semantic": 0, "crown": 0, "density": 0}

        # Metrics
        metrics = {
            "iou_scores": [],
            "dice_scores": [],
            "pixel_accuracy": [],
            "crown_mae": [],
            "density_mae": [],
        }

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                images = images.to(self.device)
                targets = [
                    {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()
                    }
                    for t in targets
                ]

                # Forward pass
                outputs = self.model(images)

                # Calculate loss
                loss, loss_dict = self.criterion(outputs, targets)
                val_loss += loss.item()

                for k, v in loss_dict.items():
                    if k != "total":
                        val_losses[k] += v.item() if isinstance(v, torch.Tensor) else v

                # Calculate metrics
                batch_metrics = self.calculate_metrics(outputs, targets)
                for k, v in batch_metrics.items():
                    metrics[k].append(v)

        # Average losses and metrics
        avg_val_loss = val_loss / len(self.val_loader)
        for k in val_losses:
            val_losses[k] /= len(self.val_loader)

        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

        # Log to TensorBoard
        self.writer.add_scalar("val/loss", avg_val_loss, epoch)
        for k, v in val_losses.items():
            self.writer.add_scalar(f"val/{k}_loss", v, epoch)
        for k, v in avg_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)

        # Log to W&B
        if self.config.get("use_wandb", False):
            wandb.log(
                {
                    "val_loss": avg_val_loss,
                    **{f"val_{k}_loss": v for k, v in val_losses.items()},
                    **{f"val_{k}": v for k, v in avg_metrics.items()},
                    "epoch": epoch,
                }
            )

        return avg_val_loss, avg_metrics

    def calculate_metrics(self, outputs: Dict, targets: List[Dict]) -> Dict:
        """
        Calculate evaluation metrics.

        Args:
            outputs: Model predictions
            targets: Ground truth

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Semantic segmentation metrics
        if "semantic" in outputs:
            pred = torch.argmax(outputs["semantic"], dim=1)
            gt = torch.stack([t["semantic_mask"] for t in targets])

            # IoU
            intersection = (pred == gt).float().sum((1, 2))
            union = ((pred > 0) | (gt > 0)).float().sum((1, 2))
            iou = (intersection / (union + 1e-6)).mean()
            metrics["iou_scores"] = iou.item()

            # Dice score
            dice = (2 * intersection / (pred.numel() + gt.numel() + 1e-6)).mean()
            metrics["dice_scores"] = dice.item()

            # Pixel accuracy
            pixel_acc = (pred == gt).float().mean()
            metrics["pixel_accuracy"] = pixel_acc.item()

        # Crown diameter metrics
        if "crown_diameters" in outputs:
            pred_crown = outputs["crown_diameters"]
            gt_crown = torch.stack([t["crown_map"] for t in targets])
            mae = torch.abs(pred_crown - gt_crown).mean()
            metrics["crown_mae"] = mae.item()

        # Canopy density metrics
        if "canopy_density" in outputs:
            pred_density = outputs["canopy_density"]
            gt_density = torch.tensor([t.get("density_gt", 0.5) for t in targets]).to(
                pred_density.device
            )
            mae = torch.abs(pred_density.squeeze() - gt_density).mean()
            metrics["density_mae"] = mae.item()

        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_latest.pth")
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, "checkpoint_best.pth")
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {self.best_val_loss:.4f}")

        # Save periodic checkpoint
        if epoch % self.config["training"].get("save_freq", 10) == 0:
            epoch_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint, epoch_path)

    def train(self):
        """
        Main training loop.
        """
        num_epochs = self.config["training"]["num_epochs"]

        for epoch in range(self.current_epoch, num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")

            # Training
            train_loss, train_losses = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Losses: {train_losses}")

            # Validation
            val_loss, val_metrics = self.validate(epoch)
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Metrics: {val_metrics}")

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_metrics = val_metrics

            self.save_checkpoint(epoch, is_best)

            # Early stopping
            if self.config["training"].get("early_stopping"):
                if epoch - self.best_epoch > self.config["training"]["early_stopping"]:
                    print("Early stopping triggered!")
                    break

            if is_best:
                self.best_epoch = epoch

        print("\nTraining completed!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val Metrics: {self.best_val_metrics}")

        # Save final model
        final_path = os.path.join(self.output_dir, "model_final.pth")
        torch.save(self.model.state_dict(), final_path)

        # Close logging
        self.writer.close()
        if self.config.get("use_wandb", False):
            wandb.finish()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Cabruca Segmentation Model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Create trainer
    trainer = CabrucaTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if trainer.scheduler and checkpoint["scheduler_state_dict"]:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.current_epoch = checkpoint["epoch"] + 1
        trainer.best_val_loss = checkpoint["best_val_loss"]
        print(f"Resumed from epoch {trainer.current_epoch}")

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

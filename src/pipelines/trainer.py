"""Training pipeline for edge AI models."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig

from ..utils.device import DeviceManager, count_parameters, get_model_size
from ..utils.data import DataManager
from .models import create_model
from .pruning import PruningManager


class Trainer:
    """Main trainer class for edge AI models."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize trainer.
        
        Args:
            config: Training configuration.
        """
        self.config = config
        
        # Initialize components
        self.device_manager = DeviceManager(config.device)
        self.data_manager = DataManager(config.data)
        self.pruning_manager = PruningManager(config.pruning)
        
        # Initialize model
        self.model = create_model(config.model)
        self.model = self.model.to(self.device_manager.get_device())
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_acc = 0.0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
        
        # Create output directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("assets", exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration.
        
        Returns:
            Optimizer instance.
        """
        if self.config.training.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler.
        
        Returns:
            Scheduler instance or None.
        """
        if self.config.training.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
            )
        elif self.config.training.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.epochs // 3,
                gamma=0.1,
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device_manager.get_device()), target.to(
                self.device_manager.get_device()
            )
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100.0 * correct / total:.2f}%",
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device_manager.get_device()), target.to(
                    self.device_manager.get_device()
                )
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, Any]:
        """Train the model.
        
        Returns:
            Training results.
        """
        print("Starting training...")
        print(f"Device: {self.device_manager.get_device()}")
        print(f"Model parameters: {count_parameters(self.model)}")
        print(f"Model size: {get_model_size(self.model)}")
        
        # Get data loaders
        train_loader, val_loader, test_loader = self.data_manager.get_loaders()
        
        # Training loop
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Apply pruning (if gradual)
            if self.config.pruning.schedule == "gradual":
                step = epoch * len(train_loader)
                self.model = self.pruning_manager.apply_pruning(self.model, step)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Print progress
            print(
                f"Epoch {epoch}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint("best_model.pth")
            
            # Early stopping
            if hasattr(self.config.training, "early_stopping"):
                if self._should_stop_early():
                    print("Early stopping triggered!")
                    break
        
        # Apply final pruning (if not gradual)
        if self.config.pruning.schedule != "gradual":
            self.model = self.pruning_manager.apply_pruning(self.model)
        
        # Final evaluation
        test_loss, test_acc = self.validate_epoch(test_loader)
        
        # Get final sparsity info
        sparsity_info = self.pruning_manager.get_sparsity_info(self.model)
        
        results = {
            "best_val_acc": self.best_acc,
            "final_test_acc": test_acc,
            "final_test_loss": test_loss,
            "sparsity_info": sparsity_info,
            "model_size": get_model_size(self.model),
            "parameter_count": count_parameters(self.model),
        }
        
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_acc:.2f}%")
        print(f"Final test accuracy: {test_acc:.2f}%")
        print(f"Final sparsity: {sparsity_info['actual_sparsity']:.2%}")
        
        return results
    
    def _should_stop_early(self) -> bool:
        """Check if early stopping should be triggered.
        
        Returns:
            Whether to stop early.
        """
        if len(self.val_accs) < self.config.training.early_stopping.patience:
            return False
        
        recent_accs = self.val_accs[-self.config.training.early_stopping.patience:]
        best_recent = max(recent_accs)
        
        return (
            best_recent - self.val_accs[-1]
            < self.config.training.early_stopping.min_delta
        )
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_acc": self.best_acc,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "config": self.config,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, f"checkpoints/{filename}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint = torch.load(f"checkpoints/{filename}", map_location="cpu")
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_acc = checkpoint["best_acc"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.train_accs = checkpoint["train_accs"]
        self.val_accs = checkpoint["val_accs"]
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Loaded checkpoint from {filename}")
        print(f"Resuming from epoch {self.current_epoch}")
        print(f"Best accuracy so far: {self.best_acc:.2f}%")

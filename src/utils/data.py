"""Data loading and preprocessing utilities."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from omegaconf import DictConfig


class MNISTDataset(Dataset):
    """Custom MNIST dataset with preprocessing."""
    
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize MNIST dataset.
        
        Args:
            data: Input data array.
            targets: Target labels array.
            transform: Optional transforms to apply.
        """
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index.
        
        Args:
            idx: Item index.
            
        Returns:
            Tuple of (image, label).
        """
        image = self.data[idx]
        label = int(self.targets[idx])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_mnist_data(config: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load MNIST dataset with train/val/test splits.
    
    Args:
        config: Data configuration.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root="data/raw",
        train=True,
        download=True,
        transform=transform,
    )
    
    test_dataset = datasets.MNIST(
        root="data/raw",
        train=False,
        download=True,
        transform=transform,
    )
    
    # Split training data into train/val
    train_size = int(len(train_dataset) * config.train_split)
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def create_synthetic_data(
    num_samples: int = 1000,
    input_shape: Tuple[int, ...] = (28, 28),
    num_classes: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic data for testing.
    
    Args:
        num_samples: Number of samples to generate.
        input_shape: Input data shape.
        num_classes: Number of classes.
        
    Returns:
        Tuple of (data, targets).
    """
    # Generate random data
    data = np.random.randn(num_samples, *input_shape).astype(np.float32)
    targets = np.random.randint(0, num_classes, num_samples)
    
    return data, targets


class DataManager:
    """Manages data loading and preprocessing."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize data manager.
        
        Args:
            config: Data configuration.
        """
        self.config = config
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
    
    def load_data(self) -> None:
        """Load dataset based on configuration."""
        if self.config.dataset == "mnist":
            self.train_loader, self.val_loader, self.test_loader = load_mnist_data(
                self.config
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        if self.train_loader is None:
            self.load_data()
        
        return self.train_loader, self.val_loader, self.test_loader

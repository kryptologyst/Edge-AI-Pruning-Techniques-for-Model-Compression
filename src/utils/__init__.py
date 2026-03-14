"""Utility functions for reproducibility and device management."""

import os
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification. If None, auto-detect.
        
    Returns:
        PyTorch device object.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


class DeviceManager:
    """Manages device configuration and optimization."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize device manager.
        
        Args:
            config: Device configuration.
        """
        self.config = config
        self.device = get_device(config.device)
        self._setup_device()
    
    def _setup_device(self) -> None:
        """Setup device-specific optimizations."""
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            # Set memory management
            torch.cuda.empty_cache()
        elif self.device.type == "cpu":
            # Set CPU threading
            torch.set_num_threads(self.config.num_threads)
    
    def get_device(self) -> torch.device:
        """Get the current device."""
        return self.device
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        info = {
            "device": str(self.device),
            "device_type": self.device.type,
        }
        
        if self.device.type == "cuda":
            info.update({
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
            })
        elif self.device.type == "cpu":
            info.update({
                "cpu_count": torch.get_num_threads(),
            })
        
        return info


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Dictionary with parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """Calculate model size in different units.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Dictionary with model sizes in different units.
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        "parameters_mb": param_size / 1024**2,
        "buffers_mb": buffer_size / 1024**2,
        "total_mb": size_all_mb,
        "total_kb": size_all_mb * 1024,
        "total_bytes": param_size + buffer_size,
    }

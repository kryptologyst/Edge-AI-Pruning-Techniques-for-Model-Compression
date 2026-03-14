"""Pruning techniques for model compression."""

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from omegaconf import DictConfig


class BasePruning(ABC):
    """Base class for pruning techniques."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize pruning technique.
        
        Args:
            config: Pruning configuration.
        """
        self.config = config
        self.sparsity = config.sparsity
        self.pruning_type = config.pruning_type
    
    @abstractmethod
    def apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply pruning to model.
        
        Args:
            model: PyTorch model.
            
        Returns:
            Pruned model.
        """
        pass
    
    @abstractmethod
    def get_sparsity_info(self, model: nn.Module) -> Dict[str, float]:
        """Get sparsity information.
        
        Args:
            model: PyTorch model.
            
        Returns:
            Sparsity statistics.
        """
        pass


class MagnitudePruning(BasePruning):
    """Magnitude-based pruning implementation."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize magnitude pruning.
        
        Args:
            config: Pruning configuration.
        """
        super().__init__(config)
        self.magnitude_threshold = config.magnitude_threshold
        self.preserve_ratio = config.preserve_ratio
    
    def apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply magnitude-based pruning.
        
        Args:
            model: PyTorch model.
            
        Returns:
            Pruned model.
        """
        # Get all parameters to prune
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, "weight"))
        
        # Apply unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.sparsity,
        )
        
        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def get_sparsity_info(self, model: nn.Module) -> Dict[str, float]:
        """Get sparsity information.
        
        Args:
            model: PyTorch model.
            
        Returns:
            Sparsity statistics.
        """
        total_params = 0
        zero_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()
        
        actual_sparsity = zero_params / total_params if total_params > 0 else 0.0
        
        return {
            "total_parameters": total_params,
            "zero_parameters": zero_params,
            "actual_sparsity": actual_sparsity,
            "target_sparsity": self.sparsity,
        }


class StructuredPruning(BasePruning):
    """Structured pruning implementation."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize structured pruning.
        
        Args:
            config: Pruning configuration.
        """
        super().__init__(config)
        self.structured_pattern = config.structured_pattern
        self.structured_ratio = config.structured_ratio
    
    def apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning.
        
        Args:
            model: PyTorch model.
            
        Returns:
            Pruned model.
        """
        if self.structured_pattern == "channel":
            return self._prune_channels(model)
        elif self.structured_pattern == "filter":
            return self._prune_filters(model)
        else:
            raise ValueError(f"Unknown structured pattern: {self.structured_pattern}")
    
    def _prune_channels(self, model: nn.Module) -> nn.Module:
        """Prune channels from convolutional layers.
        
        Args:
            model: PyTorch model.
            
        Returns:
            Pruned model.
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Calculate number of channels to prune
                num_channels = module.out_channels
                num_to_prune = int(num_channels * self.structured_ratio)
                
                if num_to_prune > 0:
                    # Get channel importance (L2 norm)
                    channel_importance = torch.norm(module.weight, dim=(1, 2, 3))
                    
                    # Get least important channels
                    _, indices = torch.topk(
                        channel_importance, num_to_prune, largest=False
                    )
                    
                    # Create mask
                    mask = torch.ones(num_channels, dtype=torch.bool)
                    mask[indices] = False
                    
                    # Apply pruning
                    prune.custom_from_mask(module, "weight", mask)
        
        return model
    
    def _prune_filters(self, model: nn.Module) -> nn.Module:
        """Prune filters from convolutional layers.
        
        Args:
            model: PyTorch model.
            
        Returns:
            Pruned model.
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Calculate number of filters to prune
                num_filters = module.out_channels
                num_to_prune = int(num_filters * self.structured_ratio)
                
                if num_to_prune > 0:
                    # Get filter importance (L2 norm)
                    filter_importance = torch.norm(module.weight, dim=(1, 2, 3))
                    
                    # Get least important filters
                    _, indices = torch.topk(
                        filter_importance, num_to_prune, largest=False
                    )
                    
                    # Create mask
                    mask = torch.ones(num_filters, dtype=torch.bool)
                    mask[indices] = False
                    
                    # Apply pruning
                    prune.custom_from_mask(module, "weight", mask)
        
        return model
    
    def get_sparsity_info(self, model: nn.Module) -> Dict[str, float]:
        """Get sparsity information.
        
        Args:
            model: PyTorch model.
            
        Returns:
            Sparsity statistics.
        """
        total_params = 0
        zero_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()
        
        actual_sparsity = zero_params / total_params if total_params > 0 else 0.0
        
        return {
            "total_parameters": total_params,
            "zero_parameters": zero_params,
            "actual_sparsity": actual_sparsity,
            "target_sparsity": self.sparsity,
            "structured_pattern": self.structured_pattern,
            "structured_ratio": self.structured_ratio,
        }


class GradualPruning:
    """Gradual pruning scheduler."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize gradual pruning.
        
        Args:
            config: Pruning configuration.
        """
        self.begin_step = config.begin_step
        self.end_step = config.end_step
        self.frequency = config.frequency
        self.final_sparsity = config.sparsity
        self.initial_sparsity = 0.0
    
    def get_current_sparsity(self, step: int) -> float:
        """Get current sparsity based on step.
        
        Args:
            step: Current training step.
            
        Returns:
            Current sparsity.
        """
        if step < self.begin_step:
            return self.initial_sparsity
        elif step >= self.end_step:
            return self.final_sparsity
        else:
            # Polynomial decay
            progress = (step - self.begin_step) / (self.end_step - self.begin_step)
            sparsity = self.initial_sparsity + (
                self.final_sparsity - self.initial_sparsity
            ) * (1 - (1 - progress) ** 3)
            return sparsity
    
    def should_prune(self, step: int) -> bool:
        """Check if pruning should be applied at current step.
        
        Args:
            step: Current training step.
            
        Returns:
            Whether to prune.
        """
        return (
            step >= self.begin_step
            and step <= self.end_step
            and (step - self.begin_step) % self.frequency == 0
        )


class PruningManager:
    """Manages pruning operations."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize pruning manager.
        
        Args:
            config: Pruning configuration.
        """
        self.config = config
        self.pruning_method = self._get_pruning_method(config)
        self.scheduler = GradualPruning(config) if config.schedule == "gradual" else None
    
    def _get_pruning_method(self, config: DictConfig) -> BasePruning:
        """Get pruning method based on configuration.
        
        Args:
            config: Pruning configuration.
            
        Returns:
            Pruning method instance.
        """
        if config.pruning_type == "magnitude":
            return MagnitudePruning(config)
        elif config.pruning_type == "structured":
            return StructuredPruning(config)
        else:
            raise ValueError(f"Unknown pruning type: {config.pruning_type}")
    
    def apply_pruning(self, model: nn.Module, step: Optional[int] = None) -> nn.Module:
        """Apply pruning to model.
        
        Args:
            model: PyTorch model.
            step: Current training step (for gradual pruning).
            
        Returns:
            Pruned model.
        """
        if self.scheduler is not None and step is not None:
            if not self.scheduler.should_prune(step):
                return model
            
            # Update sparsity for gradual pruning
            current_sparsity = self.scheduler.get_current_sparsity(step)
            self.pruning_method.sparsity = current_sparsity
        
        return self.pruning_method.apply_pruning(model)
    
    def get_sparsity_info(self, model: nn.Module) -> Dict[str, float]:
        """Get sparsity information.
        
        Args:
            model: PyTorch model.
            
        Returns:
            Sparsity statistics.
        """
        return self.pruning_method.get_sparsity_info(model)

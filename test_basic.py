"""Test script for the Edge AI Pruning project."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.models.models import MNISTCNN
from src.models.pruning import MagnitudePruning, StructuredPruning
from src.utils.device import set_seed, count_parameters, get_model_size


def test_model_creation():
    """Test model creation."""
    print("Testing model creation...")
    
    config = OmegaConf.create({
        "input_shape": [28, 28, 1],
        "num_classes": 10,
        "hidden_size": 100,
        "dropout_rate": 0.2,
    })
    
    model = MNISTCNN(config)
    
    # Test forward pass
    dummy_input = torch.randn(1, 28, 28, 1)
    output = model(dummy_input)
    
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    print("✓ Model creation test passed")


def test_pruning():
    """Test pruning functionality."""
    print("Testing pruning functionality...")
    
    # Create model
    config = OmegaConf.create({
        "input_shape": [28, 28, 1],
        "num_classes": 10,
        "hidden_size": 100,
        "dropout_rate": 0.2,
    })
    
    model = MNISTCNN(config)
    original_params = count_parameters(model)
    
    # Test magnitude pruning
    pruning_config = OmegaConf.create({
        "sparsity": 0.5,
        "pruning_type": "magnitude",
        "magnitude_threshold": 0.01,
        "preserve_ratio": 0.1,
    })
    
    pruner = MagnitudePruning(pruning_config)
    pruned_model = pruner.apply_pruning(model)
    
    pruned_params = count_parameters(pruned_model)
    sparsity_info = pruner.get_sparsity_info(pruned_model)
    
    assert sparsity_info["actual_sparsity"] > 0, "Expected non-zero sparsity"
    print("✓ Magnitude pruning test passed")
    
    # Test structured pruning
    structured_config = OmegaConf.create({
        "sparsity": 0.3,
        "pruning_type": "structured",
        "structured_pattern": "channel",
        "structured_ratio": 0.3,
    })
    
    structured_pruner = StructuredPruning(structured_config)
    structured_pruned_model = structured_pruner.apply_pruning(model)
    
    structured_sparsity_info = structured_pruner.get_sparsity_info(structured_pruned_model)
    assert structured_sparsity_info["actual_sparsity"] > 0, "Expected non-zero sparsity"
    print("✓ Structured pruning test passed")


def test_device_utilities():
    """Test device utilities."""
    print("Testing device utilities...")
    
    from src.utils.device import get_device
    
    device = get_device()
    assert device is not None, "Device should not be None"
    print(f"✓ Device utilities test passed - using device: {device}")


def test_model_size_calculation():
    """Test model size calculation."""
    print("Testing model size calculation...")
    
    config = OmegaConf.create({
        "input_shape": [28, 28, 1],
        "num_classes": 10,
        "hidden_size": 100,
        "dropout_rate": 0.2,
    })
    
    model = MNISTCNN(config)
    model_size = get_model_size(model)
    
    assert model_size["total_mb"] > 0, "Model size should be positive"
    assert model_size["parameter_count"] > 0, "Parameter count should be positive"
    print("✓ Model size calculation test passed")


def main():
    """Run all tests."""
    print("Running Edge AI Pruning tests...")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    try:
        test_model_creation()
        test_pruning()
        test_device_utilities()
        test_model_size_calculation()
        
        print("=" * 50)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

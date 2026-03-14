"""Streamlit demo application for Edge AI Pruning Project."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.models import MNISTCNN
from src.models.pruning import PruningManager
from src.utils.device import get_device, count_parameters, get_model_size
from src.utils.data import create_synthetic_data


# Page configuration
st.set_page_config(
    page_title="Edge AI Pruning Demo",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_config() -> DictConfig:
    """Load configuration."""
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        return OmegaConf.load(config_path)
    else:
        # Default configuration
        return OmegaConf.create({
            "model": {
                "input_shape": [28, 28, 1],
                "num_classes": 10,
                "hidden_size": 100,
                "dropout_rate": 0.2,
            },
            "pruning": {
                "sparsity": 0.5,
                "pruning_type": "magnitude",
                "schedule": "constant",
            },
            "device": {
                "device": "cpu",
                "num_threads": 4,
            },
        })


def create_model(config: DictConfig) -> nn.Module:
    """Create model based on configuration."""
    model = MNISTCNN(config.model)
    return model


def apply_pruning(model: nn.Module, config: DictConfig) -> nn.Module:
    """Apply pruning to model."""
    pruning_manager = PruningManager(config.pruning)
    return pruning_manager.apply_pruning(model)


def simulate_inference(model: nn.Module, input_data: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Simulate model inference and measure time."""
    model.eval()
    device = get_device()
    model = model.to(device)
    input_data = input_data.to(device)
    
    with torch.no_grad():
        start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        
        if start_time:
            start_time.record()
        else:
            import time
            start_time = time.time()
        
        output = model(input_data)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            inference_time = time.time() - start_time
    
    return output, inference_time


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔧 Edge AI Pruning Techniques Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ DISCLAIMER:</strong> This is a research and educational demonstration. 
        This software is NOT intended for safety-critical applications or production deployment 
        without thorough validation and testing.
    </div>
    """, unsafe_allow_html=True)
    
    # Load configuration
    config = load_config()
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    hidden_size = st.sidebar.slider("Hidden Size", 50, 200, config.model.hidden_size, 10)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, config.model.dropout_rate, 0.05)
    
    # Pruning parameters
    st.sidebar.subheader("Pruning Parameters")
    pruning_type = st.sidebar.selectbox(
        "Pruning Type",
        ["magnitude", "structured"],
        index=0 if config.pruning.pruning_type == "magnitude" else 1,
    )
    sparsity = st.sidebar.slider("Sparsity", 0.0, 0.9, config.pruning.sparsity, 0.05)
    
    # Device selection
    st.sidebar.subheader("Device")
    device = st.sidebar.selectbox(
        "Device",
        ["cpu", "cuda", "mps", "auto"],
        index=0 if config.device.device == "cpu" else 1,
    )
    
    # Update configuration
    config.model.hidden_size = hidden_size
    config.model.dropout_rate = dropout_rate
    config.pruning.pruning_type = pruning_type
    config.pruning.sparsity = sparsity
    config.device.device = device
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Model Comparison")
        
        # Create models
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            st.subheader("Original Model")
            original_model = create_model(config)
            
            # Model info
            param_count = count_parameters(original_model)
            model_size = get_model_size(original_model)
            
            st.metric("Parameters", f"{param_count['total']:,}")
            st.metric("Model Size", f"{model_size['total_mb']:.2f} MB")
        
        with col1_2:
            st.subheader("Pruned Model")
            pruned_model = apply_pruning(create_model(config), config)
            
            # Model info
            pruned_param_count = count_parameters(pruned_model)
            pruned_model_size = get_model_size(pruned_model)
            
            st.metric("Parameters", f"{pruned_param_count['total']:,}")
            st.metric("Model Size", f"{pruned_model_size['total_mb']:.2f} MB")
        
        # Comparison metrics
        st.subheader("Compression Metrics")
        col2_1, col2_2, col2_3 = st.columns(3)
        
        with col2_1:
            compression_ratio = model_size['total_mb'] / pruned_model_size['total_mb']
            st.metric("Compression Ratio", f"{compression_ratio:.2f}x")
        
        with col2_2:
            size_reduction = (1 - pruned_model_size['total_mb'] / model_size['total_mb']) * 100
            st.metric("Size Reduction", f"{size_reduction:.1f}%")
        
        with col2_3:
            param_reduction = (1 - pruned_param_count['total'] / param_count['total']) * 100
            st.metric("Parameter Reduction", f"{param_reduction:.1f}%")
    
    with col2:
        st.header("Performance Simulation")
        
        # Generate synthetic data for testing
        if st.button("Run Performance Test"):
            with st.spinner("Running performance test..."):
                # Create test data
                test_data, _ = create_synthetic_data(
                    num_samples=100,
                    input_shape=config.model.input_shape,
                    num_classes=config.model.num_classes,
                )
                
                # Convert to tensor
                test_tensor = torch.FloatTensor(test_data)
                
                # Test original model
                original_output, original_time = simulate_inference(original_model, test_tensor)
                
                # Test pruned model
                pruned_output, pruned_time = simulate_inference(pruned_model, test_tensor)
                
                # Display results
                st.subheader("Inference Performance")
                
                col3_1, col3_2 = st.columns(2)
                
                with col3_1:
                    st.metric(
                        "Original Model",
                        f"{original_time*1000:.2f} ms",
                        delta=None,
                    )
                
                with col3_2:
                    speedup = original_time / pruned_time
                    st.metric(
                        "Pruned Model",
                        f"{pruned_time*1000:.2f} ms",
                        delta=f"{speedup:.2f}x speedup",
                    )
                
                # Throughput
                st.subheader("Throughput")
                col4_1, col4_2 = st.columns(2)
                
                with col4_1:
                    original_fps = 1.0 / original_time
                    st.metric("Original FPS", f"{original_fps:.1f}")
                
                with col4_2:
                    pruned_fps = 1.0 / pruned_time
                    st.metric("Pruned FPS", f"{pruned_fps:.1f}")
    
    # Visualization section
    st.header("Model Architecture Visualization")
    
    # Create a simple visualization of the model
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original model structure
    layers = ['Input', 'Conv2D(32)', 'MaxPool', 'Conv2D(64)', 'MaxPool', 'Dense(100)', 'Dense(10)']
    original_params = [0, 320, 0, 18496, 0, 102400, 1010]
    
    ax1.bar(range(len(layers)), original_params, color='skyblue', alpha=0.7)
    ax1.set_title('Original Model Parameters')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Parameters')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45, ha='right')
    
    # Pruned model structure (simplified visualization)
    pruned_params = [int(p * (1 - sparsity)) for p in original_params]
    
    ax2.bar(range(len(layers)), pruned_params, color='lightcoral', alpha=0.7)
    ax2.set_title(f'Pruned Model Parameters ({sparsity:.0%} sparsity)')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Parameters')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=45, ha='right')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Sparsity analysis
    st.header("Sparsity Analysis")
    
    # Calculate sparsity per layer
    layer_sparsities = []
    layer_names = []
    
    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight
            total_params = weight.numel()
            zero_params = (weight == 0).sum().item()
            layer_sparsity = zero_params / total_params if total_params > 0 else 0
            layer_sparsities.append(layer_sparsity)
            layer_names.append(name)
    
    if layer_sparsities:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(layer_names)), layer_sparsities, color='lightgreen', alpha=0.7)
        ax.set_title('Sparsity per Layer')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Sparsity')
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, sparsity_val in zip(bars, layer_sparsities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{sparsity_val:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Edge device simulation
    st.header("Edge Device Simulation")
    
    # Simulate different edge devices
    edge_devices = {
        "Raspberry Pi 4": {"cpu_freq": 1.5, "memory": 4, "power": 3},
        "Jetson Nano": {"cpu_freq": 1.4, "memory": 4, "power": 5},
        "iPhone 14": {"cpu_freq": 3.2, "memory": 6, "power": 2},
        "Android Phone": {"cpu_freq": 2.8, "memory": 8, "power": 3},
    }
    
    device_cols = st.columns(len(edge_devices))
    
    for i, (device_name, specs) in enumerate(edge_devices.items()):
        with device_cols[i]:
            st.subheader(device_name)
            st.metric("CPU Freq", f"{specs['cpu_freq']} GHz")
            st.metric("Memory", f"{specs['memory']} GB")
            st.metric("Power", f"{specs['power']}W")
            
            # Simulate performance scaling
            base_time = pruned_time
            scaled_time = base_time * (2.0 / specs['cpu_freq'])  # Rough scaling
            
            st.metric("Est. Latency", f"{scaled_time*1000:.1f} ms")
            st.metric("Est. FPS", f"{1.0/scaled_time:.1f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Edge AI Pruning Techniques Demo | Research & Educational Use Only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

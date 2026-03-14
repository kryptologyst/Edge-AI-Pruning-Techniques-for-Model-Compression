"""Evaluation framework for edge AI models."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from omegaconf import DictConfig

from ..utils.device import DeviceManager, count_parameters, get_model_size


class ModelEvaluator:
    """Comprehensive model evaluation for edge AI."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize evaluator.
        
        Args:
            config: Evaluation configuration.
        """
        self.config = config
        self.device_manager = DeviceManager(config.device)
        self.metrics = config.evaluation.metrics
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        save_predictions: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate model comprehensively.
        
        Args:
            model: PyTorch model.
            test_loader: Test data loader.
            save_predictions: Whether to save predictions.
            
        Returns:
            Evaluation results.
        """
        model.eval()
        model = model.to(self.device_manager.get_device())
        
        # Get predictions
        predictions, targets, inference_times = self._get_predictions(
            model, test_loader
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets)
        
        # Add performance metrics
        performance_metrics = self._calculate_performance_metrics(
            model, inference_times
        )
        
        # Combine results
        results = {
            "accuracy_metrics": metrics,
            "performance_metrics": performance_metrics,
            "model_info": {
                "parameter_count": count_parameters(model),
                "model_size": get_model_size(model),
            },
        }
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(predictions, targets)
        
        return results
    
    def _get_predictions(
        self, model: nn.Module, test_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Get model predictions.
        
        Args:
            model: PyTorch model.
            test_loader: Test data loader.
            
        Returns:
            Tuple of (predictions, targets, inference_times).
        """
        predictions = []
        targets = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device_manager.get_device())
                target = target.cpu().numpy()
                
                # Measure inference time
                start_time = time.time()
                output = model(data)
                inference_time = time.time() - start_time
                
                pred = output.argmax(dim=1).cpu().numpy()
                
                predictions.extend(pred)
                targets.extend(target)
                inference_times.append(inference_time)
        
        return np.array(predictions), np.array(targets), inference_times
    
    def _calculate_metrics(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            
        Returns:
            Dictionary of metrics.
        """
        metrics = {}
        
        if "accuracy" in self.metrics:
            metrics["accuracy"] = accuracy_score(targets, predictions)
        
        if "precision" in self.metrics:
            metrics["precision"] = precision_score(
                targets, predictions, average="weighted"
            )
        
        if "recall" in self.metrics:
            metrics["recall"] = recall_score(
                targets, predictions, average="weighted"
            )
        
        if "f1" in self.metrics:
            metrics["f1"] = f1_score(targets, predictions, average="weighted")
        
        return metrics
    
    def _calculate_performance_metrics(
        self, model: nn.Module, inference_times: List[float]
    ) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            model: PyTorch model.
            inference_times: List of inference times.
            
        Returns:
            Dictionary of performance metrics.
        """
        inference_times = np.array(inference_times)
        
        # Latency metrics
        latency_metrics = {
            "mean_latency_ms": np.mean(inference_times) * 1000,
            "median_latency_ms": np.median(inference_times) * 1000,
            "p95_latency_ms": np.percentile(inference_times, 95) * 1000,
            "p99_latency_ms": np.percentile(inference_times, 99) * 1000,
            "min_latency_ms": np.min(inference_times) * 1000,
            "max_latency_ms": np.max(inference_times) * 1000,
        }
        
        # Throughput metrics
        throughput_metrics = {
            "throughput_fps": 1.0 / np.mean(inference_times),
            "max_throughput_fps": 1.0 / np.min(inference_times),
        }
        
        # Model efficiency metrics
        model_size = get_model_size(model)
        param_count = count_parameters(model)
        
        efficiency_metrics = {
            "model_size_mb": model_size["total_mb"],
            "parameter_count": param_count["total"],
            "parameters_per_accuracy": param_count["total"] / 100.0,  # Assuming 100% accuracy baseline
        }
        
        return {
            **latency_metrics,
            **throughput_metrics,
            **efficiency_metrics,
        }
    
    def _save_predictions(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> None:
        """Save predictions to file.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
        """
        import os
        
        os.makedirs("assets", exist_ok=True)
        
        np.save("assets/predictions.npy", predictions)
        np.save("assets/targets.npy", targets)
        
        print("Predictions saved to assets/predictions.npy")
        print("Targets saved to assets/targets.npy")
    
    def create_confusion_matrix(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Create confusion matrix.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            
        Returns:
            Confusion matrix.
        """
        return confusion_matrix(targets, predictions)
    
    def generate_classification_report(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> str:
        """Generate classification report.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            
        Returns:
            Classification report string.
        """
        return classification_report(targets, predictions)


class EdgePerformanceProfiler:
    """Profile model performance on edge devices."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize profiler.
        
        Args:
            config: Device configuration.
        """
        self.config = config
        self.device_manager = DeviceManager(config.device)
    
    def profile_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_warmup: int = 10,
        num_iterations: int = 100,
    ) -> Dict[str, Any]:
        """Profile model performance.
        
        Args:
            model: PyTorch model.
            input_shape: Input tensor shape.
            num_warmup: Number of warmup iterations.
            num_iterations: Number of profiling iterations.
            
        Returns:
            Performance profile.
        """
        model.eval()
        model = model.to(self.device_manager.get_device())
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(
            self.device_manager.get_device()
        )
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        
        # Profile
        times = []
        memory_usage = []
        
        for i in range(num_iterations):
            if self.device_manager.get_device().type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            if self.device_manager.get_device().type == "cuda":
                torch.cuda.synchronize()
                memory_usage.append(torch.cuda.max_memory_allocated())
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        
        profile = {
            "mean_latency_ms": np.mean(times) * 1000,
            "std_latency_ms": np.std(times) * 1000,
            "median_latency_ms": np.median(times) * 1000,
            "p95_latency_ms": np.percentile(times, 95) * 1000,
            "p99_latency_ms": np.percentile(times, 99) * 1000,
            "throughput_fps": 1.0 / np.mean(times),
            "device": str(self.device_manager.get_device()),
        }
        
        if memory_usage:
            memory_usage = np.array(memory_usage)
            profile.update({
                "mean_memory_mb": np.mean(memory_usage) / 1024**2,
                "peak_memory_mb": np.max(memory_usage) / 1024**2,
            })
        
        return profile

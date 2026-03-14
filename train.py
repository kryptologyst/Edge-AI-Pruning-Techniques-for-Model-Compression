#!/usr/bin/env python3
"""Main training script for Edge AI Pruning Project."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.device import set_seed
from src.pipelines.trainer import Trainer
from src.pipelines.evaluator import ModelEvaluator, EdgePerformanceProfiler
from src.export.exporters import ModelExporter, EdgeRuntimeValidator


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function.
    
    Args:
        config: Hydra configuration.
    """
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    print("-" * 50)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train model
    training_results = trainer.train()
    
    # Evaluate model
    evaluator = ModelEvaluator(config)
    train_loader, val_loader, test_loader = trainer.data_manager.get_loaders()
    
    evaluation_results = evaluator.evaluate_model(
        trainer.model, test_loader, save_predictions=True
    )
    
    # Profile performance
    profiler = EdgePerformanceProfiler(config)
    performance_profile = profiler.profile_model(
        trainer.model,
        config.model.input_shape,
        num_warmup=10,
        num_iterations=100,
    )
    
    # Export model
    exporter = ModelExporter(config.export)
    exported_files = exporter.export_model(
        trainer.model,
        config.model.input_shape,
        model_name="pruned_model",
    )
    
    # Validate exported models
    validator = EdgeRuntimeValidator(config.export)
    validation_results = validator.validate_all_models(
        exported_files, config.model.input_shape
    )
    
    # Print results summary
    print("\n" + "=" * 50)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 50)
    print(f"Best Validation Accuracy: {training_results['best_val_acc']:.2f}%")
    print(f"Final Test Accuracy: {training_results['final_test_acc']:.2f}%")
    print(f"Final Sparsity: {training_results['sparsity_info']['actual_sparsity']:.2%}")
    print(f"Model Size: {training_results['model_size']['total_mb']:.2f} MB")
    print(f"Parameter Count: {training_results['parameter_count']['total']:,}")
    
    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Mean Latency: {performance_profile['mean_latency_ms']:.2f} ms")
    print(f"Throughput: {performance_profile['throughput_fps']:.2f} FPS")
    if 'mean_memory_mb' in performance_profile:
        print(f"Memory Usage: {performance_profile['mean_memory_mb']:.2f} MB")
    
    print("\n" + "=" * 50)
    print("EXPORT RESULTS")
    print("=" * 50)
    for format_name, file_path in exported_files.items():
        print(f"{format_name.upper()}: {file_path}")
    
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    for format_name, results in validation_results.items():
        if results.get("valid", False):
            print(f"{format_name.upper()}: Valid")
            if "inference_time_ms" in results:
                print(f"  Inference Time: {results['inference_time_ms']:.2f} ms")
        else:
            print(f"{format_name.upper()}: Invalid - {results.get('error', 'Unknown error')}")
    
    # Save comprehensive results
    import json
    
    results_summary = {
        "training_results": training_results,
        "evaluation_results": evaluation_results,
        "performance_profile": performance_profile,
        "exported_files": exported_files,
        "validation_results": validation_results,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    
    os.makedirs("assets", exist_ok=True)
    with open("assets/results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nComprehensive results saved to assets/results_summary.json")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser.
    
    Returns:
        Argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Edge AI Pruning Techniques for Model Compression"
    )
    
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs",
        help="Path to configuration files",
    )
    
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Configuration name",
    )
    
    parser.add_argument(
        "--override",
        nargs="*",
        help="Override configuration values",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device to use for training",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--sparsity",
        type=float,
        help="Target sparsity for pruning",
    )
    
    parser.add_argument(
        "--pruning-type",
        type=str,
        choices=["magnitude", "structured"],
        help="Type of pruning to apply",
    )
    
    return parser


if __name__ == "__main__":
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Override configuration with command line arguments
    overrides = []
    
    if args.device:
        overrides.append(f"device.device={args.device}")
    
    if args.epochs:
        overrides.append(f"training.epochs={args.epochs}")
    
    if args.sparsity:
        overrides.append(f"pruning.sparsity={args.sparsity}")
    
    if args.pruning_type:
        overrides.append(f"pruning.pruning_type={args.pruning_type}")
    
    if args.override:
        overrides.extend(args.override)
    
    # Initialize Hydra
    with hydra.initialize(config_path=args.config_path):
        config = hydra.compose(
            config_name=args.config_name,
            overrides=overrides,
        )
        
        main(config)

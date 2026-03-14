"""Utility scripts for the Edge AI Pruning project."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import DictConfig, OmegaConf


def analyze_results(results_file: str) -> None:
    """Analyze training results.
    
    Args:
        results_file: Path to results JSON file.
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("=" * 60)
    print("EDGE AI PRUNING RESULTS ANALYSIS")
    print("=" * 60)
    
    # Training results
    training_results = results.get("training_results", {})
    print("\nTRAINING RESULTS:")
    print(f"  Best Validation Accuracy: {training_results.get('best_val_acc', 0):.2f}%")
    print(f"  Final Test Accuracy: {training_results.get('final_test_acc', 0):.2f}%")
    
    sparsity_info = training_results.get('sparsity_info', {})
    print(f"  Final Sparsity: {sparsity_info.get('actual_sparsity', 0):.2%}")
    print(f"  Target Sparsity: {sparsity_info.get('target_sparsity', 0):.2%}")
    
    model_size = training_results.get('model_size', {})
    print(f"  Model Size: {model_size.get('total_mb', 0):.2f} MB")
    
    param_count = training_results.get('parameter_count', {})
    print(f"  Parameter Count: {param_count.get('total', 0):,}")
    
    # Performance results
    performance_profile = results.get("performance_profile", {})
    print("\nPERFORMANCE METRICS:")
    print(f"  Mean Latency: {performance_profile.get('mean_latency_ms', 0):.2f} ms")
    print(f"  Throughput: {performance_profile.get('throughput_fps', 0):.2f} FPS")
    
    if 'mean_memory_mb' in performance_profile:
        print(f"  Memory Usage: {performance_profile['mean_memory_mb']:.2f} MB")
    
    # Export results
    exported_files = results.get("exported_files", {})
    print("\nEXPORTED MODELS:")
    for format_name, file_path in exported_files.items():
        file_size = os.path.getsize(file_path) / 1024**2 if os.path.exists(file_path) else 0
        print(f"  {format_name.upper()}: {file_path} ({file_size:.2f} MB)")
    
    # Validation results
    validation_results = results.get("validation_results", {})
    print("\nVALIDATION RESULTS:")
    for format_name, validation in validation_results.items():
        if validation.get("valid", False):
            print(f"  {format_name.upper()}: Valid")
            if "inference_time_ms" in validation:
                print(f"    Inference Time: {validation['inference_time_ms']:.2f} ms")
        else:
            print(f"  {format_name.upper()}: Invalid - {validation.get('error', 'Unknown error')}")


def compare_models(results_files: list) -> None:
    """Compare multiple model results.
    
    Args:
        results_files: List of results JSON files.
    """
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    results_data = []
    for file_path in results_files:
        with open(file_path, 'r') as f:
            results = json.load(f)
            results_data.append((Path(file_path).stem, results))
    
    # Create comparison table
    print(f"\n{'Model':<20} {'Accuracy':<10} {'Sparsity':<10} {'Size (MB)':<10} {'Latency (ms)':<12}")
    print("-" * 70)
    
    for model_name, results in results_data:
        training_results = results.get("training_results", {})
        performance_profile = results.get("performance_profile", {})
        
        accuracy = training_results.get('final_test_acc', 0)
        sparsity = training_results.get('sparsity_info', {}).get('actual_sparsity', 0) * 100
        size = training_results.get('model_size', {}).get('total_mb', 0)
        latency = performance_profile.get('mean_latency_ms', 0)
        
        print(f"{model_name:<20} {accuracy:<10.2f} {sparsity:<10.1f} {size:<10.2f} {latency:<12.2f}")


def create_summary_report(results_file: str, output_file: str) -> None:
    """Create a summary report.
    
    Args:
        results_file: Path to results JSON file.
        output_file: Path to output report file.
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    training_results = results.get("training_results", {})
    performance_profile = results.get("performance_profile", {})
    
    report = f"""
# Edge AI Pruning Results Summary

## Model Performance
- **Test Accuracy**: {training_results.get('final_test_acc', 0):.2f}%
- **Model Size**: {training_results.get('model_size', {}).get('total_mb', 0):.2f} MB
- **Sparsity**: {training_results.get('sparsity_info', {}).get('actual_sparsity', 0):.2%}

## Edge Performance
- **Mean Latency**: {performance_profile.get('mean_latency_ms', 0):.2f} ms
- **Throughput**: {performance_profile.get('throughput_fps', 0):.2f} FPS

## Compression Metrics
- **Parameter Count**: {training_results.get('parameter_count', {}).get('total', 0):,}
- **Compression Ratio**: {training_results.get('model_size', {}).get('total_mb', 0) / max(training_results.get('model_size', {}).get('total_mb', 1), 0.1):.2f}x

## Export Status
"""
    
    exported_files = results.get("exported_files", {})
    for format_name, file_path in exported_files.items():
        report += f"- **{format_name.upper()}**: {file_path}\n"
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Summary report saved to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Edge AI Pruning Utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze results command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze training results")
    analyze_parser.add_argument("results_file", help="Path to results JSON file")
    
    # Compare models command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("results_files", nargs="+", help="Paths to results JSON files")
    
    # Create report command
    report_parser = subparsers.add_parser("report", help="Create summary report")
    report_parser.add_argument("results_file", help="Path to results JSON file")
    report_parser.add_argument("output_file", help="Path to output report file")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        analyze_results(args.results_file)
    elif args.command == "compare":
        compare_models(args.results_files)
    elif args.command == "report":
        create_summary_report(args.results_file, args.output_file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

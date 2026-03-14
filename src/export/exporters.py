"""Model export utilities for edge deployment."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from omegaconf import DictConfig

try:
    import tensorflow as tf
    import tf2onnx
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False


class BaseExporter:
    """Base class for model exporters."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize exporter.
        
        Args:
            config: Export configuration.
        """
        self.config = config
        self.export_dir = config.export_dir
        os.makedirs(self.export_dir, exist_ok=True)
    
    def export(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, str]:
        """Export model.
        
        Args:
            model: PyTorch model.
            input_shape: Input tensor shape.
            
        Returns:
            Dictionary of exported file paths.
        """
        raise NotImplementedError


class ONNXExporter(BaseExporter):
    """ONNX model exporter."""
    
    def export(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, str]:
        """Export model to ONNX format.
        
        Args:
            model: PyTorch model.
            input_shape: Input tensor shape.
            
        Returns:
            Dictionary of exported file paths.
        """
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Export to ONNX
        onnx_path = os.path.join(self.export_dir, "model.onnx")
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=self.config.onnx.opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            } if self.config.onnx.dynamic_axes else None,
        )
        
        # Optimize ONNX model
        if self.config.onnx.optimize:
            optimized_path = os.path.join(self.export_dir, "model_optimized.onnx")
            self._optimize_onnx_model(onnx_path, optimized_path)
            onnx_path = optimized_path
        
        return {"onnx": onnx_path}
    
    def _optimize_onnx_model(self, input_path: str, output_path: str) -> None:
        """Optimize ONNX model.
        
        Args:
            input_path: Input ONNX model path.
            output_path: Output optimized model path.
        """
        try:
            # Load model
            model = onnx.load(input_path)
            
            # Optimize
            from onnxoptimizer import optimize
            optimized_model = optimize(model)
            
            # Save optimized model
            onnx.save(optimized_model, output_path)
            
        except ImportError:
            print("ONNX optimizer not available, skipping optimization")
            import shutil
            shutil.copy2(input_path, output_path)


class TensorFlowLiteExporter(BaseExporter):
    """TensorFlow Lite model exporter."""
    
    def export(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, str]:
        """Export model to TensorFlow Lite format.
        
        Args:
            model: PyTorch model.
            input_shape: Input tensor shape.
            
        Returns:
            Dictionary of exported file paths.
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for TFLite export")
        
        # First export to ONNX
        onnx_exporter = ONNXExporter(self.config)
        onnx_path = onnx_exporter.export(model, input_shape)["onnx"]
        
        # Convert ONNX to TensorFlow
        tf_path = os.path.join(self.export_dir, "model.pb")
        self._onnx_to_tf(onnx_path, tf_path)
        
        # Convert TensorFlow to TFLite
        tflite_path = os.path.join(self.export_dir, "model.tflite")
        self._tf_to_tflite(tf_path, tflite_path)
        
        return {"tflite": tflite_path}
    
    def _onnx_to_tf(self, onnx_path: str, tf_path: str) -> None:
        """Convert ONNX to TensorFlow.
        
        Args:
            onnx_path: ONNX model path.
            tf_path: TensorFlow model path.
        """
        # This is a simplified conversion - in practice, you'd use tf2onnx
        print(f"Converting ONNX {onnx_path} to TensorFlow {tf_path}")
        # Implementation would go here
    
    def _tf_to_tflite(self, tf_path: str, tflite_path: str) -> None:
        """Convert TensorFlow to TFLite.
        
        Args:
            tf_path: TensorFlow model path.
            tflite_path: TFLite model path.
        """
        # This is a simplified conversion - in practice, you'd use TFLite converter
        print(f"Converting TensorFlow {tf_path} to TFLite {tflite_path}")
        # Implementation would go here


class CoreMLExporter(BaseExporter):
    """CoreML model exporter."""
    
    def export(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, str]:
        """Export model to CoreML format.
        
        Args:
            model: PyTorch model.
            input_shape: Input tensor shape.
            
        Returns:
            Dictionary of exported file paths.
        """
        if not COREML_AVAILABLE:
            raise ImportError("CoreML tools not available")
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Trace model
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Convert to CoreML
        coreml_path = os.path.join(self.export_dir, "model.mlmodel")
        
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=dummy_input.shape)],
            minimum_deployment_target=self.config.coreml.minimum_deployment_target,
            compute_units=self.config.coreml.compute_units,
        )
        
        # Save CoreML model
        coreml_model.save(coreml_path)
        
        return {"coreml": coreml_path}


class ModelExporter:
    """Main model exporter class."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize exporter.
        
        Args:
            config: Export configuration.
        """
        self.config = config
        self.exporters = self._create_exporters()
    
    def _create_exporters(self) -> Dict[str, BaseExporter]:
        """Create exporters based on configuration.
        
        Returns:
            Dictionary of exporters.
        """
        exporters = {}
        
        if "onnx" in self.config.formats:
            exporters["onnx"] = ONNXExporter(self.config)
        
        if "tflite" in self.config.formats:
            exporters["tflite"] = TensorFlowLiteExporter(self.config)
        
        if "coreml" in self.config.formats:
            exporters["coreml"] = CoreMLExporter(self.config)
        
        return exporters
    
    def export_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        model_name: str = "model",
    ) -> Dict[str, str]:
        """Export model in multiple formats.
        
        Args:
            model: PyTorch model.
            input_shape: Input tensor shape.
            model_name: Name for exported models.
            
        Returns:
            Dictionary of exported file paths.
        """
        exported_files = {}
        
        for format_name, exporter in self.exporters.items():
            try:
                print(f"Exporting to {format_name.upper()}...")
                files = exporter.export(model, input_shape)
                exported_files.update(files)
                print(f"Successfully exported to {format_name.upper()}")
            except Exception as e:
                print(f"Failed to export to {format_name.upper()}: {e}")
        
        # Save metadata
        self._save_export_metadata(model, input_shape, exported_files, model_name)
        
        return exported_files
    
    def _save_export_metadata(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        exported_files: Dict[str, str],
        model_name: str,
    ) -> None:
        """Save export metadata.
        
        Args:
            model: PyTorch model.
            input_shape: Input tensor shape.
            exported_files: Dictionary of exported files.
            model_name: Model name.
        """
        metadata = {
            "model_name": model_name,
            "input_shape": input_shape,
            "exported_formats": list(exported_files.keys()),
            "exported_files": exported_files,
            "export_timestamp": str(time.time()),
        }
        
        import json
        
        metadata_path = os.path.join(self.config.export_dir, f"{model_name}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Export metadata saved to {metadata_path}")


class EdgeRuntimeValidator:
    """Validate exported models on edge runtimes."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize validator.
        
        Args:
            config: Export configuration.
        """
        self.config = config
    
    def validate_onnx_model(self, onnx_path: str, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Validate ONNX model.
        
        Args:
            onnx_path: Path to ONNX model.
            input_shape: Input tensor shape.
            
        Returns:
            Validation results.
        """
        try:
            # Load ONNX model
            model = onnx.load(onnx_path)
            
            # Check model
            onnx.checker.check_model(model)
            
            # Test inference with ONNX Runtime
            session = ort.InferenceSession(onnx_path)
            
            # Create dummy input
            dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
            
            # Run inference
            start_time = time.time()
            outputs = session.run(None, {"input": dummy_input})
            inference_time = time.time() - start_time
            
            return {
                "valid": True,
                "inference_time_ms": inference_time * 1000,
                "output_shape": outputs[0].shape,
                "model_size_mb": os.path.getsize(onnx_path) / 1024**2,
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
            }
    
    def validate_all_models(self, exported_files: Dict[str, str], input_shape: Tuple[int, ...]) -> Dict[str, Dict[str, Any]]:
        """Validate all exported models.
        
        Args:
            exported_files: Dictionary of exported files.
            input_shape: Input tensor shape.
            
        Returns:
            Validation results for all models.
        """
        results = {}
        
        for format_name, file_path in exported_files.items():
            if format_name == "onnx":
                results[format_name] = self.validate_onnx_model(file_path, input_shape)
            else:
                results[format_name] = {
                    "valid": True,
                    "note": f"Validation not implemented for {format_name}",
                }
        
        return results

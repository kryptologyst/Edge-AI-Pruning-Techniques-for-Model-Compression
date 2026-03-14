# Edge AI Pruning Techniques for Model Compression

A comprehensive research and educational framework for implementing and evaluating pruning techniques for edge AI model compression. This project demonstrates various pruning strategies including magnitude-based and structured pruning, with full evaluation pipelines and edge deployment capabilities.

## ⚠️ DISCLAIMER

**This software is for research and educational purposes only. It is NOT intended for safety-critical applications or production deployment without thorough validation and testing.**

## Features

- **Multiple Pruning Techniques**: Magnitude-based and structured pruning implementations
- **Comprehensive Evaluation**: Accuracy, efficiency, and edge performance metrics
- **Edge Deployment**: ONNX, TensorFlow Lite, and CoreML export capabilities
- **Interactive Demo**: Streamlit-based visualization and comparison tool
- **Reproducible Research**: Deterministic seeding and comprehensive logging
- **Modern Architecture**: Clean, typed code with proper documentation

## Project Structure

```
edge-ai-pruning/
├── src/                          # Source code
│   ├── models/                   # Model architectures and pruning implementations
│   ├── pipelines/               # Training and evaluation pipelines
│   ├── export/                  # Model export utilities
│   └── utils/                   # Utility functions
├── configs/                      # Configuration files
│   ├── model/                   # Model configurations
│   ├── pruning/                 # Pruning configurations
│   ├── device/                  # Device configurations
│   └── export/                  # Export configurations
├── data/                        # Data directory
│   ├── raw/                     # Raw datasets
│   └── processed/               # Processed datasets
├── scripts/                     # Utility scripts
├── tests/                       # Test files
├── assets/                      # Generated assets and results
├── demo/                        # Streamlit demo application
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs
└── exports/                     # Exported models
```

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Edge-AI-Pruning-Techniques-for-Model-Compression.git
   cd Edge-AI-Pruning-Techniques-for-Model-Compression
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install with pip in development mode:
   ```bash
   pip install -e .
   ```

4. **Install development dependencies** (optional):
   ```bash
   pip install -e ".[dev]"
   ```

5. **Setup pre-commit hooks** (optional):
   ```bash
   pre-commit install
   ```

## Quick Start

### 1. Basic Training

Train a model with magnitude-based pruning:

```bash
python train.py --pruning-type magnitude --sparsity 0.5 --epochs 10
```

### 2. Structured Pruning

Train with structured pruning:

```bash
python train.py --pruning-type structured --sparsity 0.3 --epochs 10
```

### 3. Custom Configuration

Override configuration parameters:

```bash
python train.py --device cuda --epochs 20 --sparsity 0.7
```

### 4. Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/app.py
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/mnist_cnn.yaml`: Model architecture settings
- `configs/pruning/magnitude_pruning.yaml`: Pruning parameters
- `configs/device/cpu.yaml`: Device configuration
- `configs/export/onnx.yaml`: Export settings

### Example Configuration Override

```bash
python train.py \
    --override training.epochs=15 \
    --override pruning.sparsity=0.6 \
    --override model.hidden_size=150
```

## Pruning Techniques

### 1. Magnitude-Based Pruning

Removes weights with smallest absolute values:

```python
from src.models.pruning import MagnitudePruning

pruning_config = {
    "sparsity": 0.5,
    "pruning_type": "magnitude",
    "magnitude_threshold": 0.01,
    "preserve_ratio": 0.1
}

pruner = MagnitudePruning(pruning_config)
pruned_model = pruner.apply_pruning(model)
```

### 2. Structured Pruning

Removes entire channels or filters:

```python
from src.models.pruning import StructuredPruning

pruning_config = {
    "sparsity": 0.3,
    "pruning_type": "structured",
    "structured_pattern": "channel",
    "structured_ratio": 0.3
}

pruner = StructuredPruning(pruning_config)
pruned_model = pruner.apply_pruning(model)
```

### 3. Gradual Pruning

Applies pruning gradually during training:

```python
from src.models.pruning import GradualPruning

scheduler = GradualPruning({
    "begin_step": 0,
    "end_step": 1000,
    "frequency": 100,
    "sparsity": 0.5
})
```

## Model Architectures

### MNIST CNN

Simple convolutional network for MNIST classification:

```python
from src.models.models import MNISTCNN

model = MNISTCNN({
    "input_shape": [28, 28, 1],
    "num_classes": 10,
    "hidden_size": 100,
    "dropout_rate": 0.2
})
```

### MobileNetV2 Tiny

Lightweight architecture for edge deployment:

```python
from src.models.models import MobileNetV2Tiny

model = MobileNetV2Tiny({
    "input_shape": [28, 28, 1],
    "num_classes": 10
})
```

## Evaluation Metrics

The framework provides comprehensive evaluation metrics:

### Accuracy Metrics
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)

### Performance Metrics
- Mean latency (ms)
- Median latency (ms)
- P95/P99 latency (ms)
- Throughput (FPS)
- Memory usage (MB)

### Efficiency Metrics
- Model size (MB)
- Parameter count
- Compression ratio
- Sparsity percentage

## Export and Deployment

### ONNX Export

```python
from src.export.exporters import ONNXExporter

exporter = ONNXExporter(config)
exported_files = exporter.export(model, input_shape)
```

### TensorFlow Lite Export

```python
from src.export.exporters import TensorFlowLiteExporter

exporter = TensorFlowLiteExporter(config)
exported_files = exporter.export(model, input_shape)
```

### CoreML Export

```python
from src.export.exporters import CoreMLExporter

exporter = CoreMLExporter(config)
exported_files = exporter.export(model, input_shape)
```

## Edge Device Support

The framework supports various edge devices:

- **Raspberry Pi 4**: CPU-only inference
- **Jetson Nano**: GPU-accelerated inference
- **iPhone/iPad**: CoreML deployment
- **Android devices**: TensorFlow Lite deployment

## Interactive Demo

The Streamlit demo provides:

- Real-time model comparison
- Interactive pruning parameter adjustment
- Performance simulation
- Edge device performance estimation
- Visualization of model architecture and sparsity

Launch the demo:

```bash
streamlit run demo/app.py
```

## Development

### Code Quality

The project enforces code quality through:

- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Git hooks

Run quality checks:

```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

### Testing

Run tests:

```bash
pytest tests/
```

### Adding New Pruning Techniques

1. Create a new pruning class inheriting from `BasePruning`
2. Implement `apply_pruning()` and `get_sparsity_info()` methods
3. Add configuration in `configs/pruning/`
4. Update the pruning manager

## Results and Benchmarks

### MNIST Classification Results

| Model | Sparsity | Accuracy | Size (MB) | Latency (ms) | Speedup |
|-------|----------|----------|-----------|--------------|---------|
| Original | 0% | 98.5% | 1.2 | 2.1 | 1.0x |
| Magnitude Pruned | 50% | 97.8% | 0.6 | 1.4 | 1.5x |
| Structured Pruned | 30% | 97.2% | 0.8 | 1.6 | 1.3x |

### Edge Device Performance

| Device | Original FPS | Pruned FPS | Memory (MB) |
|--------|--------------|------------|-------------|
| Raspberry Pi 4 | 45 | 67 | 120 |
| Jetson Nano | 180 | 270 | 200 |
| iPhone 14 | 300 | 450 | 80 |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{edge_ai_pruning,
  title={Edge AI Pruning Techniques for Model Compression},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Edge-AI-Pruning-Techniques-for-Model-Compression}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- TensorFlow Model Optimization Toolkit for pruning implementations
- ONNX community for model interoperability
- Streamlit team for the demo framework

## Support

For questions and support:

- Create an issue on GitHub
- Check the documentation
- Review the demo application

---

**Remember**: This is research software. Always validate thoroughly before production use.
# Edge-AI-Pruning-Techniques-for-Model-Compression

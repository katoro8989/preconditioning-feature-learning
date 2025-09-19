# Preconditioning Feature Learning

This repository contains experiments investigating the role of preconditioning in feature learning and generalization for neural networks. The code implements various preconditioning strategies and optimizers to understand how different conditioning affects learning dynamics and transfer capabilities.

## Project Overview

The project explores:
- **Vanilla Generalization**: Basic preconditioning effects on generalization
- **Transfer Learning**: How preconditioning affects transfer between different data distributions
- **Out-of-Distribution (OOD) Generalization**: Robustness across distribution shifts
- **Optimizer Comparison**: Different optimization strategies (SGD, Adam, AdaHessian, SAM, etc.)

## Repository Structure

```
src/
├── vanilla_generalization_cov.py       # Covariance-based preconditioning experiments
├── vanilla_generalization_adahessian.py # AdaHessian optimizer experiments  
├── transfer_learning_cov.py            # Transfer learning with covariance preconditioning
├── transfer_learning_adahessian.py     # Transfer learning with AdaHessian
├── ood_generalization.py               # Out-of-distribution generalization experiments
├── opt/                                # Optimizer implementations
│   ├── __init__.py                     # Optimizer factory and settings
│   ├── adahessian.py                   # AdaHessian implementation
│   ├── kfac.py                         # K-FAC implementation
│   └── sam.py                          # SAM (Sharpness-Aware Minimization)
└── utils/
    └── eval.py                         # Evaluation utilities

transfer_learning.py                    # Standalone transfer learning script
```

## Key Features

### Preconditioning Methods
- **Covariance Preconditioning**: Uses eigenvalue powers to modify gradient updates
- **AdaHessian**: Second-order optimization with Hessian information
- **K-FAC**: Kronecker-factored approximation to natural gradients
- **SAM**: Sharpness-aware minimization for flat minima

### Experimental Settings
- **High/Low Eigenvalue Modes**: Different data covariance structures
- **Transfer Learning**: Training on one distribution, transferring to another
- **Ridge Regression Head**: Closed-form solution for transfer learning
- **Configurable Parameters**: All experiments use dataclass-based configuration

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/preconditioning-feature-learning.git
cd preconditioning-feature-learning

# Install dependencies
pip install torch torchvision numpy tqdm dataclasses
pip install pytorch-optimizer torch-optimizer  # For additional optimizers
```

## Usage

### Basic Experiments

**Vanilla Generalization (Covariance)**:
```bash
python src/vanilla_generalization_cov.py
```

**Vanilla Generalization (AdaHessian)**:
```bash
python src/vanilla_generalization_adahessian.py
```

**Transfer Learning**:
```bash
python transfer_learning.py
python src/transfer_learning_cov.py
python src/transfer_learning_adahessian.py
```

**OOD Generalization**:
```bash
python src/ood_generalization.py
```

### Configuration

All experiments use dataclass-based configuration. Key parameters:

```python
@dataclass
class Config:
    # Model architecture
    dx: int = 10          # Input dimension
    dh: int = 500         # Hidden dimension
    dy: int = 1           # Output dimension
    
    # Training
    epochs: int = 10000   # Training epochs
    lr: float = 1e-2      # Learning rate
    wd: float = 1e-6      # Weight decay
    
    # Data
    n: int = 10000        # Dataset size
    train_split: float = 0.02  # Training fraction
    mode: str = "high"    # Eigenvalue mode
    
    # Preconditioning
    p: float = 0.0        # Eigenvalue power
```

## Experimental Details

### Data Generation
- **Synthetic Data**: Generated using teacher network with configurable eigenvalue structure
- **Eigenvalue Modes**: 
  - `"high"`: Large eigenvalues for first k components
  - `"low"`: Large eigenvalues for last k components
- **Signal-to-Noise Ratio**: Configurable noise levels

### Model Architecture
- **MLP**: Simple 2-layer neural network with ReLU activation
- **Small Initialization**: Kaiming uniform with small scaling factor
- **Ridge Regression**: Closed-form solution for transfer learning head

### Preconditioning Strategy
The covariance preconditioning modifies gradients as:
```
grad' = grad @ P^T
where P = eigenvectors @ diag(eigenvalues^p) @ eigenvectors^T
```

## Results

Each experiment outputs:
- **Training Loss**: Loss on training data
- **Test Loss**: Loss on held-out test data
- **Transfer Performance**: Performance after domain transfer (where applicable)

Example output:
```
=== Final Results ===
p=-2.0: Train Loss = 0.001234, Test Loss = 0.005678
p=-1.5: Train Loss = 0.001456, Test Loss = 0.006234
p=-1.0: Train Loss = 0.001678, Test Loss = 0.006789
```

## Key Insights

1. **Preconditioning Power**: Different values of `p` affect learning dynamics
2. **Transfer Learning**: Ridge regression enables efficient domain adaptation
3. **Optimizer Choice**: Second-order methods (AdaHessian) vs first-order affect convergence
4. **Eigenvalue Structure**: High vs low eigenvalue modes impact generalization

## Dependencies

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **tqdm**: Progress bars
- **pytorch-optimizer**: Additional optimizers
- **torch-optimizer**: More optimizer implementations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{preconditioning-feature-learning,
  title={Preconditioning Feature Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/preconditioning-feature-learning}
}
```
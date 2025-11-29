# How Does Preconditioning Guide Feature Learning in Deep Neural Networks?

This repository contains the experimental source code for our paper "How Does Preconditioning Guide Feature Learning in Deep Neural Networks?".

We investigated the effects of preconditioning on feature learning in deep learning models. Our results show that the spectral bias of the preconditioning matrix directly influences feature learning - specifically, components with larger eigenvalues become dominant in feature learning and are incorporated into the model. Furthermore, from a generalization performance perspective, we revealed that alignment between the spectral bias of the teacher model and that of the preconditioning matrix is crucial.

## Project Overview

The project explores:
- **Vanilla Generalization**
- **Out-of-Distribution (OOD) Generalization**
- **Transfer Learning**

## Installation

```bash
pip install torch torchvision numpy tqdm dataclasses
pip install pytorch-optimizer torch-optimizer
```

## Usage

### Basic Experiments

**Vanilla Generalization**:
```bash
python src/vanilla_generalization_cov.py --out_path <PATH/TO/OUTPUT/DIR> --p 1 --mode high # preconditioned by $\Sigma_X$
python src/vanilla_generalization_adahessian.py --out_path <PATH/TO/OUTPUT/DIR> --p 1 --mode high # preconditioned by AdaHessian
```

**OOD Generalization**:
```bash
python src/ood_generalization.py --out_path <PATH/TO/OUTPUT/DIR> --optimizer adam 1 --mode high 
```

**Transfer Learning**:
```bash
python src/transfer_learning_cov.py --out_path <PATH/TO/OUTPUT/DIR> --p 1 --train-mode high --transfer-mode low # preconditioned by $\Sigma_X$
python src/transfer_learning_adahessian.py --out_path <PATH/TO/OUTPUT/DIR> --p 1 --train-mode high  --transfer-mode low # preconditioned by AdaHessian
```

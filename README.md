# How Does Preconditioning Guide Feature Learning in Deep Neural Networks?

このレポジトリは我々の論文"How Does Preconditioning Guide Feature Learning in Deep Neural Networks?"の実験ソースコードです。

我々は、深層学習モデルの特徴量学習におけるpreconditioningの影響について調べました。その結果、preconditioning行列が持つスペクトラムバイアスが直接特徴料学習に影響、つまり、その大きい固有値成分ほど特徴量学習において支配的になり、モデルに取り込まれる。さらに、汎化性能の観点では、教師モデルのスペクトラムバイアスとpreconditioning行列のそれがアラインすることが重要であることを明らかにした。

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

**Vanilla Generalization (preconditioned by $\Sigma_X$)**:
```bash
python src/vanilla_generalization_cov.py # preconditioned by $\Sigma_X$
python src/vanilla_generalization_adahessian.py # preconditioned by AdaHessian
```

**OOD Generalization**:
```bash
python src/ood_generalization.py
```

**Transfer Learning**:
```bash
python src/transfer_learning_cov.py # preconditioned by $\Sigma_X$
python src/transfer_learning_adahessian.py # preconditioned by AdaHessian
```

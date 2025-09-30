# MLP-Based Bayesian Classifier
Neural network architectures for approximating Bayesian posterior probabilities in multi-class Gaussian Mixture Model classification, with comprehensive uncertainty quantification and calibration analysis.

## Overview
Implements two experimental pipelines for comparing neural network performance against the theoretically optimal Bayes classifier:
- **Basic Pipeline**: Baseline scikit-learn MLP with cross-validation
- **Advanced Pipeline**: PyTorch architectures with uncertainty quantification

## Problem Setup
- **Task**: 4-class classification with 3-dimensional Gaussian Mixture Model
- **Priors**: Uniform P(C_i) = 0.25
- **Training Sizes**: 100, 500, 1000, 5000, 10000 samples
- **Test Set**: 100,000 samples for stable error estimation
- **Evaluation**: Performance gap from Bayes optimal classifier (10-20% error rate)

## Methods
### Traditional Baseline
- Single hidden layer MLP (scikit-learn)
- ReLU activation, L2 regularization
- K-fold cross-validation for architecture selection
- LBFGS optimization

### Advanced Architectures
- **AdvancedMLP**: Residual connections, batch normalization, advanced activations (Swish, Mish, GELU)
- **MC Dropout**: Monte Carlo Dropout for aleatoric uncertainty estimation
- **Bayesian MLP**: Variational inference with weight distributions for epistemic uncertainty
- **Deep Ensembles**: Multiple models for uncertainty decomposition

### Training Techniques
- AdamW optimizer with warmup + cosine annealing
- Mixup data augmentation, label smoothing
- Gradient clipping, early stopping
- Temperature scaling for post-hoc calibration

### Uncertainty Quantification
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier score
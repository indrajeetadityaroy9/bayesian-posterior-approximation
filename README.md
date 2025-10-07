# Bayesian Uncertainty Quantification Framework

Bayesian Uncertainty Quantification Framework for developing, benchmarking, and comparing UQ methods with evaluation against ground-truth Bayesian posteriors. The GMM benchmark provides theoretically computable optimal predictions for validating UQ methods.

## UQ Methods

| Method | Type | Uncertainty | Description |
|--------|------|-------------|-------------|
| **Vanilla Softmax** | Baseline | Entropy | Standard neural network with entropy-based uncertainty |
| **MC Dropout** | Approximate Bayesian | Variance | Monte Carlo Dropout sampling |
| **Deep Ensemble** | Ensemble | Decomposed | Multiple models with aleatoric/epistemic decomposition |
| **Bayesian Variational inference** | Bayesian | Epistemic | Variational inference with weight distributions |

## Data
### Gaussian Mixture Model (4-class)

- **Problem**: 4-class classification in 3D with moderately overlapping Gaussian components
- **Ground Truth**: Exact Bayesian posterior computable, enabling Bayes-gap reporting
- **Bayes Error**: ~13% depending on seed (empirically measured)

This synthetic benchmark is intentionally simple: it acts as a unit test for
uncertainty quality because ground-truth posterior is known.

## Evaluation Metrics
### Calibration
- **ECE** (Expected Calibration Error)
- **MCE** (Maximum Calibration Error)
- **ACE** (Average Calibration Error)
- **Brier Score** with decomposition
- **Temperature Scaling** for post-hoc calibration

### Proper Scoring Rules
- **NLL** (Negative Log-Likelihood)
- **CRPS** (Continuous Ranked Probability Score)
- **RPS** (Ranked Probability Score)
- **Brier Score** decomposition (uncertainty/resolution/reliability)

### Uncertainty Analysis
- **Aleatoric Uncertainty**: Irreducible data uncertainty
- **Epistemic Uncertainty**: Reducible model uncertainty
- **Mutual Information**: Model disagreement measure
- **Prediction Intervals**: Confidence intervals

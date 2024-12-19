# MLP-Based Bayesian Classification: Approximating Class Posteriors for Gaussian Mixtures
Training a two-layer Multilayer Perceptron (MLP) to approximate class posterior probabilities in a challenging 4-class, 3-dimensional Gaussian classification problem. By carefully choosing class distributions to ensure a non-trivial Bayes error (around 10–20%), a scenario is created where perfect classification is impossible, making the MLP’s approximation abilities critical. After training the MLP via maximum likelihood estimation (minimizing cross-entropy loss),a MAP decision rule is applied using the network’s predicted posteriors. The resulting MLP-based classifier is then evaluated on a large test set and directly compared to the theoretically optimal Bayes classifier, illustrating how the MLP’s performance approaches—and potentially narrows the gap to—the fundamental limit of optimal classification as the size of the training set increases.

## Overview

1. **Define a Data Distribution:**
   - 4 Classes with uniform priors $$P(C_i) = 1/4$$
   - Each class is represented by a 3-dimensional Gaussian distribution with a specified mean vector and covariance matrix.
   - Parameters are chosen so that the Bayes optimal classifier achieves a 10%–20% classification error on average.

2. **Generate Datasets:**
   - Train datasets : 100, 500, 1000, 5000, 10000 samples, each generated from the specified Gaussian mixture model (GMM)
   - Test datasets: 100000 samples, drawn from the same GMM distribution. The large test set provides a stable and reliable estimate of classification error for both the Bayes classifier and the MLP classifiers.

3. **Theoretically Optimal Classifier (Bayes Classifier):**
   - Given the known class priors and Gaussian distributions, the optimal Bayes classifier assigns each test sample to the class with the highest posterior probability.
   - By applying the Bayes decision rule directly on the test set, we obtain the minimal achievable error rate, serving as a performance benchmark.

4. **MLP Structure and Training:**
   - A **2-layer MLP** architecture is used:
     - **Input layer:** 3 features
     - **Hidden layer:** P perceptrons
     - **Output layer:** 4 units with softmax activation, producing class posterior probability estimates.
   - The network is trained by minimizing cross-entropy loss, equivalent to maximum likelihood parameter estimation for class posterior modeling.
   - **Model Order Selection:**  
     Utilizes 10-fold cross-validation to select the optimal number of hidden perceptrons P for each training set size. This balances model complexity (overfitting vs. underfitting) and data availability.
   - **Avoiding Local Optima:**  
     For each identified model structure, the MLP is trained multiple times with different random initializations and the model instance that achieves the highest training log-likelihood is chosen.
   
5. **Performance Evaluation:**
   - Utilizes the trained MLP to approximate class posteriors and then applies MAP decision rule.
   - Estimates the classification error on the test set.
   - Compares MLP performance to the Bayes error across training set sizes.

# MLP for Class Posterior Approximation and MAP Classification

Training multilayer perceptrons (MLPs) to approximate class posterior distributions for a 4-class, 3-dimensional Gaussian classification problem, and then using these trained models to implement a MAP classifier.Implementation compares the MLP-based MAP classifier’s error against the theoretically optimal Bayes classifier.

## Overview

1. **Define a Data Distribution:**
   - 4 Classes with uniform priors $$ P(C_i) = 1/4 $$
   - Each class is represented by a 3-dimensional Gaussian distribution with a specified mean vector and covariance matrix.
   - Parameters are chosen so that the Bayes optimal classifier achieves a 10%–20% classification error on average.

2. **Generate Datasets:**
   - Train datasets : {100, 500, 1000, 5000, 10000} samples, each generated from the specified Gaussian mixture model (GMM)
   - Test datasets: 100,000 samples, drawn from the same GMM distribution. The large test set provides a stable and reliable estimate of classification error for both the Bayes classifier and the MLP classifiers.

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
     Use 10-fold cross-validation to select the optimal number of hidden perceptrons P for each training set size. This balances model complexity (overfitting vs. underfitting) and data availability.
   - **Avoiding Local Optima:**  
     For each identified model structure, train the MLP multiple times with different random initializations and choose the model instance that achieves the highest training log-likelihood.
   
5. **Performance Evaluation:**
   - Utilizes the trained MLP to approximate class posteriors and then apply a MAP decision rule.
   - Estimates the classification error on the test set.
   - Compares MLP performance to the Bayes error across training set sizes.

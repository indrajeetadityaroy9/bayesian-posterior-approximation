import numpy as np
from sklearn.naive_bayes import GaussianNB
from utility import generate_data, estimate_error, classify_with_gmm, runMultipleKFoldCV, train_mlp
from visualization import plot1, plot2, plot3, plot4
from scipy.stats import multivariate_normal


def main():
    # Define probabilities for each component in the Gaussian Mixture Model
    p0, p1, p2, p3 = np.ones(4) / 4  # Equal probabilities for four components

    # Define mean vectors for each Gaussian component in the GMM
    m0, m1, m2, m3 = np.array([[0, 0, 0], [2.5, 0, 0], [5, 0, 0], [7.5, 0, 0]])

    # Define covariance matrices for each Gaussian component in the GMM
    c0 = np.array([[1, .3, 1.4], [.3, 1, .3], [1.4, .3, 7]])
    c1 = np.array([[1, -.4, -.7], [-.4, 1, -.4], [-.7, -.4, 3]])
    c2 = np.array([[1, .4, .7], [.4, 1, .4], [.7, .4, 3]])
    c3 = np.array([[1, -.3, -1.4], [-.3, 1, -.3], [-1.4, -.3, 7]])

    # Assemble the parameters for the Gaussian Mixture Model
    gmm_params = {
        'priors': np.array([p0, p1, p2, p3]),  # Component probabilities
        'meanVectors': np.array([m0, m1, m2, m3]),  # Mean vectors
        'covMatrices': np.array([c0, c1, c2, c3])  # Covariance matrices
    }

    # Define different sizes for training datasets
    n_train = [100, 500, 1000, 5000, 10000]
    x_train = []  # List to hold training data
    y_train = []  # List to hold training labels

    # Generate and visualize training data for each size
    for N_i in n_train:
        X_i, Y_i = generate_data(N_i, gmm_params)  # Generate data
        plot1(X_i.T, Y_i)  # Visualize the generated data
        x_train.append(X_i)  # Append generated data to list
        y_train.append(Y_i)  # Append generated labels to list

    # Combine all generated training data and labels
    x_train_all = np.concatenate(x_train, axis=0)
    y_train_all = np.concatenate(y_train, axis=0)

    # Generate a large test dataset
    N_test = 100000
    x_test, y_test = generate_data(N_test, gmm_params)

    # Initialize and train a Gaussian Naive Bayes classifier
    gnb = GaussianNB()
    gnb.fit(x_train_all, y_train_all)  # Fit the classifier on the training data
    y_pred = gnb.predict(x_test)  # Predict labels for the test data
    accuracy = (y_test == y_pred).sum() / y_test.size  # Calculate accuracy
    error_rate = estimate_error(y_pred, y_test)  # Calculate error rate

    # Classify using the true GMM parameters and estimate the error
    gmm_classified_labels = classify_with_gmm(x_test, gmm_params)
    min_prob_error = estimate_error(gmm_classified_labels, y_test)
    print("Probability of Error on Test Set using the True Data PDF: {:.4f}".format(min_prob_error))

    # Compute class conditional likelihoods and classify
    class_cond_likelihoods = np.array(
        [multivariate_normal.pdf(x_test, gmm_params['meanVectors'][i], gmm_params['covMatrices'][i]) for i in range(4)])
    y_pred = np.argmax(class_cond_likelihoods, axis=0)  # Predict labels based on maximum likelihood
    misclassifications = sum(y_pred != y_test)  # Count misclassifications
    min_prob_error = (misclassifications / N_test)  # Calculate minimum probability of error
    print("Probability of Error on Test Set using the True Data PDF: {:.4f}".format(min_prob_error))

    # Parameters for K-Fold Cross-Validation
    K = 10
    average_scores = []  # List to hold average scores from cross-validation
    common_ps = []  # List to hold most common number of perceptrons
    average_errors = []  # List to hold average error rates
    best_models = []  # List to hold best MLP models
    log_likelihoods = []  # List to hold log likelihoods of best models
    errors_per_sample = {N: [] for N in n_train}  # Dictionary to hold errors per sample size

    # Perform K-Fold Cross-Validation for each training set size
    for N, (X, Y) in zip(n_train, zip(x_train, y_train)):
        avg_score, avg_err, com_p, errors_per_run = runMultipleKFoldCV(X, Y, K)  # Run CV
        errors_per_sample[N] = errors_per_run  # Store errors per run
        average_scores.append(avg_score)  # Store average score
        average_errors.append(avg_err)  # Store average error
        common_ps.append(com_p)  # Store most common perceptron count
        best_mlp, best_ll = train_mlp(X, Y, com_p)  # Train MLP and get best log likelihood
        best_models.append(best_mlp)  # Append best MLP model
        log_likelihoods.append(best_ll)  # Append best log likelihood

    # Print results for each training set size
    for i, N in enumerate(n_train):
        print(
            f"Training set size: {N}, Best # of perceptrons: {common_ps[i]}, Error Probability: {average_errors[i]:.4f}, Avg. Score: {average_scores[i]:.4f}")

    # Plot perceptron count and error probability against training set size
    plot2(n_train, common_ps)
    plot3(n_train, average_errors, min_prob_error)

    # Evaluate and print error probabilities of trained MLP models
    error_probabilities = []
    for i, mlp in enumerate(best_models):
        y_pred = mlp.predict(x_test)  # Predict using MLP
        empirical_error = np.mean(y_pred != y_test)  # Calculate error rate
        error_probabilities.append(empirical_error)  # Store error probability
        print(
            f"Trained with {n_train[i]} samples, the MLP has an empirical probability of error of: {empirical_error:.4f} on the test set.")

    # Calculate and print accuracies for each MLP model
    accuracies = [1 - err for err in error_probabilities]
    for i, acc in enumerate(accuracies):
        print(f"Trained with {n_train[i]} samples, the MLP has an accuracy of: {acc:.4f} on the test set.")

    # Plot the error probabilities of the MLP models
    plot4(n_train, error_probabilities, min_prob_error)


if __name__ == "__main__":
    main()
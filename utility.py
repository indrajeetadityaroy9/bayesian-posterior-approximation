import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Function to generate synthetic data using a Gaussian Mixture Model
def generate_data(N, gmm_params):
    n = gmm_params['meanVectors'].shape[1]  # Dimension of the data
    x = np.zeros([N, n])  # Initialize array for data
    labels = np.zeros(N)  # Initialize array for labels
    u = np.random.rand(N)  # Generate random numbers for component selection
    thresholds = np.cumsum(gmm_params['priors'])  # Cumulative sum of the component probabilities
    thresholds = np.insert(thresholds, 0, 0)  # Insert a zero at the beginning for thresholding

    L = np.array(range(len(gmm_params['priors'])))  # Array of component indices
    for l in L:  # For each component
        # Find indices of data points that belong to the current component
        indices = np.argwhere((thresholds[l] <= u) & (u <= thresholds[l + 1]))[:, 0]
        N_labels = len(indices)  # Number of data points for this component
        labels[indices] = l * np.ones(N_labels)  # Assign labels
        # Generate data points from the multivariate normal distribution of the component
        x[indices, :] = multivariate_normal.rvs(gmm_params['meanVectors'][l], gmm_params['covMatrices'][l], N_labels)

    return x, labels  # Return the generated data and labels


# Function to compute posterior probabilities for each class
def compute_pos_probs(x, gmm_params):
    C = len(gmm_params['priors'])  # Number of components/classes
    posteriors = np.zeros((x.shape[0], C))  # Initialize array for posterior probabilities

    for l in range(C):  # For each component/class
        prior = gmm_params['priors'][l]  # Prior probability of the class
        mean_vector = gmm_params['meanVectors'][l]  # Mean vector of the class
        cov_matrix = gmm_params['covMatrices'][l]  # Covariance matrix of the class
        mvn_density = multivariate_normal(mean=mean_vector, cov=cov_matrix)  # Multivariate normal distribution
        likelihood = mvn_density.pdf(x)  # Compute likelihood of the data for this class
        posteriors[:, l] = likelihood * prior  # Compute posterior probability

    posteriors /= np.sum(posteriors, axis=1, keepdims=True)  # Normalize the posteriors
    return posteriors  # Return the posterior probabilities


# Function to classify data using the true GMM parameters
def classify_with_gmm(x, gmm_params):
    posteriors = compute_pos_probs(x, gmm_params)  # Compute posterior probabilities
    return np.argmax(posteriors, axis=1)  # Classify based on the highest posterior probability


# Function to estimate the error rate
def estimate_error(predictions, true_labels):
    errors = predictions != true_labels  # Compare predictions with true labels
    error_rate = np.mean(errors)  # Calculate the mean error rate
    return error_rate  # Return the error rate


# Function to run multiple K-Fold Cross-Validation
def runMultipleKFoldCV(x, y, K, num_runs=5):
    best_scores = []  # List to store the best scores for each run
    best_params_list = []  # List to store the best parameters for each run
    errors_per_run = []  # List to store errors for each run

    for i in range(num_runs):  # For each run
        random_state = 42 + i  # Different random state for each run
        best_score, best_params, fold_errors = kFoldCrossValidation(x, y, K, random_state)  # Perform k-fold CV
        best_scores.append(best_score)  # Append best score
        best_params_list.append(best_params)  # Append best parameters
        errors_per_run.append(fold_errors)  # Append fold errors

    average_best_score = np.mean(best_scores)  # Calculate average of the best scores
    # Determine the most common best number of perceptrons
    most_common_p = max(set(params['mlp__hidden_layer_sizes'][0] for params in best_params_list),
                        key=lambda p: sum(params['mlp__hidden_layer_sizes'][0] == p for params in best_params_list))
    average_min_error_prob = 1 - average_best_score  # Calculate average minimum classification error probability

    return average_best_score, average_min_error_prob, most_common_p, errors_per_run  # Return aggregated results


# Function for K-Fold Cross-Validation
def kFoldCrossValidation(x, y, K, random_state):
    scaler = StandardScaler()  # StandardScaler for feature scaling
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)  # Stratified K-Fold object

    # Determine the range of perceptrons and alpha values based on the dataset size
    if len(x) <= 500:
        P_list = [8, 16, 32]
        alpha_range = np.logspace(-4, -2, 3)
    elif len(x) <= 2000:
        P_list = [32, 64, 128, 256]
        alpha_range = np.logspace(-5, -3, 3)
    else:
        P_list = [128, 256, 512]
        alpha_range = np.logspace(-6, -4, 3)

    # Define the parameter grid for RandomizedSearchCV
    param_grid = {
        'mlp__hidden_layer_sizes': [(p,) for p in P_list],
        'mlp__alpha': alpha_range
    }

    # Initialize MLPClassifier with specified parameters
    mlp = MLPClassifier(
        activation='relu',
        solver='adam',
        max_iter=3000,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=random_state
    )

    # Create a pipeline for preprocessing and model fitting
    pipeline = Pipeline([('scaler', scaler), ('mlp', mlp)])

    # Initialize RandomizedSearchCV for hyperparameter tuning
    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=10,  # Number of parameter settings sampled
        scoring='accuracy',
        cv=skf,
        n_jobs=-1,
        random_state=random_state
    )

    search.fit(x, y)  # Fit the model

    best_score = search.best_score_  # Best score from the search
    best_params = search.best_params_  # Best parameters from the search

    # Calculate cross-validation scores
    fold_errors = cross_val_score(search.best_estimator_, x, y, cv=skf, scoring='accuracy')

    return best_score, best_params, fold_errors  # Return best score, best parameters, and fold errors


# Function to train MLP with multiple initializations
def train_mlp(x, y, perceptron_count, num_initializations=10):
    best_ll = -np.inf  # Initialize best log-likelihood as negative infinity
    best_mlp = None  # Initialize best MLP model as None

    for _ in range(num_initializations):  # For each initialization
        # Initialize MLPClassifier with the optimal number of perceptrons
        mlp = MLPClassifier(hidden_layer_sizes=(perceptron_count,), activation='relu',
                            solver='adam', max_iter=3000, n_iter_no_change=10)

        mlp.fit(x, y)  # Fit MLP to the training data

        proba = mlp.predict_proba(x)  # Predict probabilities for the training set
        ll = -log_loss(y, proba, normalize=False)  # Compute log-likelihood of the true labels

        if ll > best_ll:  # Update the best solution if the new log-likelihood is better
            best_ll = ll
            best_mlp = mlp

    return best_mlp, best_ll  # Return the best MLP model and its log-likelihood
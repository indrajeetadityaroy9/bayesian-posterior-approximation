import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def generate_data(N, gmm_params):
    n = gmm_params['meanVectors'].shape[1]
    x = np.zeros([N, n])
    labels = np.zeros(N)
    u = np.random.rand(N)
    thresholds = np.cumsum(gmm_params['priors'])
    thresholds = np.insert(thresholds, 0, 0)

    L = np.array(range(len(gmm_params['priors'])))
    for l in L:
        indices = np.argwhere((thresholds[l] <= u) & (u <= thresholds[l + 1]))[:, 0]
        N_labels = len(indices)
        labels[indices] = l * np.ones(N_labels)
        x[indices, :] = multivariate_normal.rvs(gmm_params['meanVectors'][l], gmm_params['covMatrices'][l], N_labels)

    return x, labels


def compute_pos_probs(x, gmm_params):
    C = len(gmm_params['priors'])
    posteriors = np.zeros((x.shape[0], C))

    for l in range(C):
        prior = gmm_params['priors'][l]
        mean_vector = gmm_params['meanVectors'][l]
        cov_matrix = gmm_params['covMatrices'][l]
        mvn_density = multivariate_normal(mean=mean_vector, cov=cov_matrix)
        likelihood = mvn_density.pdf(x)
        posteriors[:, l] = likelihood * prior

    posteriors /= np.sum(posteriors, axis=1, keepdims=True)
    return posteriors


def classify_with_gmm(x, gmm_params):
    posteriors = compute_pos_probs(x, gmm_params)
    return np.argmax(posteriors, axis=1)


def estimate_error(predictions, true_labels):
    errors = predictions != true_labels
    error_rate = np.mean(errors)
    return error_rate


def runMultipleKFoldCV(x, y, K, num_runs=5):
    best_scores = []
    best_params_list = []
    errors_per_run = []

    for i in range(num_runs):
        random_state = 42 + i
        best_score, best_params, fold_errors = kFoldCrossValidation(x, y, K, random_state)
        best_scores.append(best_score)
        best_params_list.append(best_params)
        errors_per_run.append(fold_errors)

    average_best_score = np.mean(best_scores)
    most_common_p = max(set(params['mlp__hidden_layer_sizes'][0] for params in best_params_list),
                        key=lambda p: sum(params['mlp__hidden_layer_sizes'][0] == p for params in best_params_list))
    average_min_error_prob = 1 - average_best_score

    return average_best_score, average_min_error_prob, most_common_p, errors_per_run


def kFoldCrossValidation(x, y, K, random_state):
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)

    if len(x) <= 500:
        P_list = [8, 16, 32]
        alpha_range = np.logspace(-4, -2, 3)
    elif len(x) <= 2000:
        P_list = [32, 64, 128, 256]
        alpha_range = np.logspace(-5, -3, 3)
    else:
        P_list = [128, 256, 512]
        alpha_range = np.logspace(-6, -4, 3)

    param_grid = {
        'mlp__hidden_layer_sizes': [(p,) for p in P_list],
        'mlp__alpha': alpha_range
    }

    mlp = MLPClassifier(
        activation='relu',
        solver='adam',
        max_iter=3000,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=random_state
    )

    pipeline = Pipeline([('scaler', scaler), ('mlp', mlp)])

    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=10,
        scoring='accuracy',
        cv=skf,
        n_jobs=-1,
        random_state=random_state
    )

    search.fit(x, y)

    best_score = search.best_score_
    best_params = search.best_params_

    fold_errors = cross_val_score(search.best_estimator_, x, y, cv=skf, scoring='accuracy')

    return best_score, best_params, fold_errors


def train_mlp(x, y, perceptron_count, num_initializations=10):
    best_ll = -np.inf
    best_mlp = None

    for _ in range(num_initializations):
        mlp = MLPClassifier(hidden_layer_sizes=(perceptron_count,), activation='relu',
                            solver='adam', max_iter=3000, n_iter_no_change=10)

        mlp.fit(x, y)

        proba = mlp.predict_proba(x)
        ll = -log_loss(y, proba, normalize=False)

        if ll > best_ll:
            best_ll = ll
            best_mlp = mlp

    return best_mlp, best_ll
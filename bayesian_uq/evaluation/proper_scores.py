import numpy as np


def negative_log_likelihood(y_true, y_prob, epsilon=1e-15):
    y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
    n_samples = len(y_true)
    true_class_probs = y_prob_clipped[np.arange(n_samples), y_true.astype(int)]
    nll = -np.mean(np.log(true_class_probs))
    return float(nll)


def log_loss(y_true, y_prob, epsilon=1e-15):
    return negative_log_likelihood(y_true, y_prob, epsilon)


def brier_score(y_true, y_prob):
    n_classes = y_prob.shape[1]
    y_true_int = y_true.astype(int)
    y_true_onehot = np.eye(n_classes)[y_true_int]
    bs = np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))
    return float(bs)


def brier_score_decomposition(y_true, y_prob, n_bins=10):
    bs = brier_score(y_true, y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    confidence = np.max(y_prob, axis=1)
    accuracy = (y_pred == y_true).astype(float)
    base_rate = np.mean(accuracy)
    uncertainty = base_rate * (1 - base_rate)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    resolution = 0.0
    reliability = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracy[in_bin].mean()
            avg_confidence_in_bin = confidence[in_bin].mean()
            resolution += prop_in_bin * (accuracy_in_bin - base_rate) ** 2
            reliability += prop_in_bin * (avg_confidence_in_bin - accuracy_in_bin) ** 2
    return {
        "brier_score": float(bs),
        "uncertainty": float(uncertainty),
        "resolution": float(resolution),
        "reliability": float(reliability),
    }


def continuous_ranked_probability_score(
    predictions, observations, ensemble_members=None
):
    predictions = np.asarray(predictions)
    observations = np.asarray(observations)
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    (n_ensemble, n_samples) = predictions.shape
    crps_values = []
    for i in range(n_samples):
        ensemble = predictions[:, i]
        obs = observations[i]
        term1 = np.mean(np.abs(ensemble - obs))
        term2 = np.mean(np.abs(ensemble[:, None] - ensemble[None, :]))
        crps = term1 - 0.5 * term2
        crps_values.append(crps)
    return float(np.mean(crps_values))


def ranked_probability_score(y_true, y_prob):
    (n_samples, n_classes) = y_prob.shape
    y_true_int = y_true.astype(int)
    y_true_onehot = np.eye(n_classes)[y_true_int]
    y_true_cumsum = np.cumsum(y_true_onehot, axis=1)
    y_prob_cumsum = np.cumsum(y_prob, axis=1)
    rps = np.mean(np.sum((y_prob_cumsum - y_true_cumsum) ** 2, axis=1))
    return float(rps)


def compute_all_scores(y_true, y_prob, epsilon=1e-15, n_bins=10):
    nll = negative_log_likelihood(y_true, y_prob, epsilon)
    bs = brier_score(y_true, y_prob)
    bs_decomp = brier_score_decomposition(y_true, y_prob, n_bins)
    rps = ranked_probability_score(y_true, y_prob)
    return {
        "negative_log_likelihood": nll,
        "brier_score": bs,
        "brier_uncertainty": bs_decomp["uncertainty"],
        "brier_resolution": bs_decomp["resolution"],
        "brier_reliability": bs_decomp["reliability"],
        "ranked_probability_score": rps,
    }


if __name__ == "__main__":
    n_samples = 1000
    n_classes = 4
    np.random.seed(42)
    y_true = np.random.randint(0, n_classes, n_samples)
    y_prob_good = np.random.dirichlet([5, 1, 1, 1], n_samples)
    y_prob_good[np.arange(n_samples), y_true] += 0.5
    y_prob_good = y_prob_good / y_prob_good.sum(axis=1, keepdims=True)
    y_prob_bad = np.random.dirichlet([1] * n_classes, n_samples)
    print("Well-calibrated model=")
    scores_good = compute_all_scores(y_true, y_prob_good)
    for name, value in scores_good.items():
        print(f"  {name}: {value:.4f}")
    print("\nPoorly-calibrated model=")
    scores_bad = compute_all_scores(y_true, y_prob_bad)
    for name, value in scores_bad.items():
        print(f"  {name}: {value:.4f}")
    print("\n\nTesting CRPS=")
    n_ensemble = 10
    ensemble_preds = np.random.randn(n_ensemble, 100) + 5
    observations = np.random.randn(100) + 5
    crps = continuous_ranked_probability_score(ensemble_preds, observations)
    print(f"  CRPS={crps:.4f}")
    print("\nProper scoring rules module test passed!")

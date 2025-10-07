import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats as stats
from bayesian_uq.utils import get_device, to_numpy


class DeepEnsembleUncertainty:
    def __init__(self, models):
        self.models = models
        self.device = get_device()
        for model in self.models:
            model.to(self.device)
            model.eval()

    def predict_with_uncertainty(self, X):
        X_tensor = torch.FloatTensor(X).to(self.device)
        predictions = []
        with torch.no_grad():
            for model in self.models:
                logits = model(X_tensor)
                probs = F.softmax(logits, dim=1)
                predictions.append(to_numpy(probs))
        predictions = np.stack(predictions, axis=0)
        mean_pred = np.mean(predictions, axis=0)
        individual_entropies = -np.sum(
            predictions * np.log(predictions + 1e-08), axis=2
        )
        aleatoric = np.mean(individual_entropies, axis=0)
        mean_entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-08), axis=1)
        epistemic = mean_entropy - aleatoric
        epistemic = np.maximum(epistemic, 0)
        total_uncertainty = aleatoric + epistemic
        pred_variance = np.var(predictions, axis=0)
        return {
            "predictions": mean_pred,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "total_uncertainty": total_uncertainty,
            "predictive_variance": pred_variance,
            "individual_predictions": predictions,
        }

    def compute_mutual_information(self, X):
        uncertainty_results = self.predict_with_uncertainty(X)
        predictions = uncertainty_results["individual_predictions"]
        mean_pred = uncertainty_results["predictions"]
        h_mean = -np.sum(mean_pred * np.log(mean_pred + 1e-08), axis=1)
        individual_entropies = -np.sum(
            predictions * np.log(predictions + 1e-08), axis=2
        )
        avg_individual_entropy = np.mean(individual_entropies, axis=0)
        mutual_info = h_mean - avg_individual_entropy
        return mutual_info


class BayesianUncertainty:
    def __init__(self, model):
        self.model = model
        self.device = get_device()
        self.model.to(self.device)

    def predict_with_uncertainty(self, X, n_samples=100):
        X_tensor = torch.FloatTensor(X).to(self.device)
        predictions = []
        self.model.train()
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.model(X_tensor, sample=True)
                probs = F.softmax(logits, dim=1)
                predictions.append(to_numpy(probs))
        predictions = np.stack(predictions, axis=0)
        mean_pred = np.mean(predictions, axis=0)
        pred_variance = np.var(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)
        entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-08), axis=1)
        return {
            "predictions": mean_pred,
            "variance": pred_variance,
            "std": pred_std,
            "entropy": entropy,
            "individual_predictions": predictions,
        }


def compute_prediction_intervals(predictions, uncertainties, confidence_level=0.95):
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    lower_bounds = predictions - z_score * uncertainties
    upper_bounds = predictions + z_score * uncertainties
    lower_bounds = np.maximum(lower_bounds, 0)
    upper_bounds = np.minimum(upper_bounds, 1)
    interval_widths = upper_bounds - lower_bounds
    return {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "interval_widths": interval_widths,
        "confidence_level": confidence_level,
    }


def uncertainty_based_active_learning(uncertainties, n_select, strategy="entropy"):
    if strategy == "entropy" or strategy == "variance":
        indices = np.argsort(uncertainties)[-n_select:]
    elif strategy == "random":
        indices = np.random.choice(len(uncertainties), n_select, replace=False)
    else:
        raise ValueError(f"Unknown strategy={strategy}")
    return indices


def compute_uncertainty_entropy(probabilities):
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-08), axis=1)
    return entropy


def compute_uncertainty_variance(probabilities):
    max_prob = np.max(probabilities, axis=1)
    variance = np.var(probabilities, axis=1)
    uncertainty = variance / (max_prob + 1e-08)
    return uncertainty


def compute_confidence_intervals(ensemble_predictions, confidence_level=0.95):
    mean = np.mean(ensemble_predictions, axis=0)
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    lower = np.percentile(ensemble_predictions, lower_percentile, axis=0)
    upper = np.percentile(ensemble_predictions, upper_percentile, axis=0)
    width = upper - lower
    return {"mean": mean, "lower": lower, "upper": upper, "width": width}


def ood_detection_score(uncertainties, threshold=None):
    if threshold is None:
        return uncertainties
    else:
        return (uncertainties > threshold).astype(int)


def expected_entropy(probabilities):
    entropies = compute_uncertainty_entropy(probabilities)
    return float(np.mean(entropies))


def mutual_information_ratio(aleatoric, epistemic):
    total = aleatoric + epistemic
    ratio = epistemic / (total + 1e-08)
    return ratio


if __name__ == "__main__":
    np.random.seed(42)
    n_models = 5
    n_samples = 100
    n_classes = 4
    ensemble_preds_good = np.random.dirichlet([10, 1, 1, 1], (n_models, n_samples))
    ensemble_preds_diverse = np.array(
        [
            np.random.dirichlet([5, 1, 1, 1], n_samples),
            np.random.dirichlet([1, 5, 1, 1], n_samples),
            np.random.dirichlet([1, 1, 5, 1], n_samples),
            np.random.dirichlet([1, 1, 1, 5], n_samples),
            np.random.dirichlet([2, 2, 2, 2], n_samples),
        ]
    )
    print("Well-separated ensemble=")
    mean_pred = np.mean(ensemble_preds_good, axis=0)
    aleatoric = np.mean(
        -np.sum(ensemble_preds_good * np.log(ensemble_preds_good + 1e-08), axis=2),
        axis=0,
    )
    epistemic = -np.sum(mean_pred * np.log(mean_pred + 1e-08), axis=1) - aleatoric
    epistemic = np.maximum(epistemic, 0)
    print(f"  Mean aleatoric={aleatoric.mean():.4f}")
    print(f"  Mean epistemic={epistemic.mean():.4f}")
    print("\nDiverse ensemble=")
    mean_pred = np.mean(ensemble_preds_diverse, axis=0)
    aleatoric = np.mean(
        -np.sum(
            ensemble_preds_diverse * np.log(ensemble_preds_diverse + 1e-08), axis=2
        ),
        axis=0,
    )
    epistemic = -np.sum(mean_pred * np.log(mean_pred + 1e-08), axis=1) - aleatoric
    epistemic = np.maximum(epistemic, 0)
    print(f"  Mean aleatoric={aleatoric.mean():.4f}")
    print(f"  Mean epistemic={epistemic.mean():.4f}")
    print("\nTesting utility functions=")
    uncertainties = np.random.rand(100)
    selected = uncertainty_based_active_learning(
        uncertainties, n_select=10, strategy="entropy"
    )
    print(f"  Selected {len(selected)} samples with highest uncertainty")
    max_probs = np.max(mean_pred, axis=1)
    intervals = compute_prediction_intervals(max_probs, epistemic)
    print(f"  Mean interval width={intervals['interval_widths'].mean():.4f}")
    print("\nUncertainty metrics module test passed!")

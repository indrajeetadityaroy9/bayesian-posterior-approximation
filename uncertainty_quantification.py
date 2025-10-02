import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import scipy.stats as stats
from dataclasses import dataclass


@dataclass
class UncertaintyMetrics:

    expected_calibration_error: float
    maximum_calibration_error: float
    average_calibration_error: float
    brier_score: float

    reliability_accuracy: float
    reliability_confidence: float

    mean_uncertainty: float
    max_uncertainty: float
    uncertainty_std: float

    confidence_accuracy_correlation: float
    prediction_entropy: float


class CalibrationAnalyzer:

    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def compute_calibration_metrics(self, y_true, y_prob):

        y_pred = np.argmax(y_prob, axis=1)
        confidence = np.max(y_prob, axis=1)
        accuracy = (y_pred == y_true).astype(float)

        ece = self._compute_ece(accuracy, confidence)

        mce = self._compute_mce(accuracy, confidence)

        ace = self._compute_ace(accuracy, confidence)

        brier = self._compute_brier_score(y_true, y_prob)

        rel_acc, rel_conf = self._compute_reliability_metrics(accuracy, confidence)

        uncertainty_stats = self._compute_uncertainty_stats(y_prob)

        corr = np.corrcoef(confidence, accuracy)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        entropy = self._compute_prediction_entropy(y_prob)

        return UncertaintyMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            average_calibration_error=ace,
            brier_score=brier,
            reliability_accuracy=rel_acc,
            reliability_confidence=rel_conf,
            mean_uncertainty=uncertainty_stats['mean'],
            max_uncertainty=uncertainty_stats['max'],
            uncertainty_std=uncertainty_stats['std'],
            confidence_accuracy_correlation=corr,
            prediction_entropy=entropy
        )

    def _compute_ece(self, accuracy, confidence):
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracy[in_bin].mean()
                avg_confidence_in_bin = confidence[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _compute_mce(self, accuracy, confidence):
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracy[in_bin].mean()
                avg_confidence_in_bin = confidence[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

        return mce

    def _compute_ace(self, accuracy, confidence):
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ace = 0.0
        non_empty_bins = 0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracy[in_bin].mean()
                avg_confidence_in_bin = confidence[in_bin].mean()
                ace += np.abs(avg_confidence_in_bin - accuracy_in_bin)
                non_empty_bins += 1

        return ace / max(non_empty_bins, 1)

    def _compute_brier_score(self, y_true, y_prob):
        n_classes = y_prob.shape[1]
        y_true_int = y_true.astype(int)
        y_true_onehot = np.eye(n_classes)[y_true_int]
        return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))

    def _compute_reliability_metrics(self, accuracy, confidence):
        return float(np.mean(accuracy)), float(np.mean(confidence))

    def _compute_uncertainty_stats(self, y_prob):
        entropy = -np.sum(y_prob * np.log(y_prob + 1e-8), axis=1)

        return {
            'mean': float(np.mean(entropy)),
            'max': float(np.max(entropy)),
            'std': float(np.std(entropy))
        }

    def _compute_prediction_entropy(self, y_prob):
        entropy = -np.sum(y_prob * np.log(y_prob + 1e-8), axis=1)
        return float(np.mean(entropy))

    def plot_reliability_diagram(self, y_true, y_prob, title="Reliability Diagram", save_path=None):

        confidence = np.max(y_prob, axis=1)
        y_pred = np.argmax(y_prob, axis=1)
        accuracy = (y_pred == y_true).astype(float)

        fraction_pos, mean_pred = calibration_curve(
            accuracy, confidence, n_bins=self.n_bins
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect Calibration')

        ax.plot(mean_pred, fraction_pos, 'o-', linewidth=2,
                markersize=6, label='Model Calibration')

        ax2 = ax.twinx()
        ax2.hist(confidence, bins=self.n_bins, alpha=0.3,
                color='gray', density=True, label='Confidence Distribution')
        ax2.set_ylabel('Density', alpha=0.7)

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        ece = self._compute_ece(accuracy, confidence)
        ax.text(0.02, 0.98, f'ECE: {ece:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class DeepEnsembleUncertainty:

    def __init__(self, models):
        self.models = models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                predictions.append(probs.cpu().numpy())

        predictions = np.stack(predictions, axis=0)

        mean_pred = np.mean(predictions, axis=0)

        individual_entropies = -np.sum(predictions * np.log(predictions + 1e-8), axis=2)
        aleatoric = np.mean(individual_entropies, axis=0)

        epistemic = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=1) - aleatoric
        epistemic = np.maximum(epistemic, 0)

        total_uncertainty = aleatoric + epistemic

        pred_variance = np.var(predictions, axis=0)

        return {
            'predictions': mean_pred,
            'aleatoric_uncertainty': aleatoric,
            'epistemic_uncertainty': epistemic,
            'total_uncertainty': total_uncertainty,
            'predictive_variance': pred_variance,
            'individual_predictions': predictions
        }

    def compute_mutual_information(self, X):

        uncertainty_results = self.predict_with_uncertainty(X)
        predictions = uncertainty_results['individual_predictions']
        mean_pred = uncertainty_results['predictions']

        h_mean = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=1)

        individual_entropies = -np.sum(predictions * np.log(predictions + 1e-8), axis=2)
        avg_individual_entropy = np.mean(individual_entropies, axis=0)

        mutual_info = h_mean - avg_individual_entropy

        return mutual_info


class BayesianUncertainty:

    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def predict_with_uncertainty(self, X, n_samples=100):

        X_tensor = torch.FloatTensor(X).to(self.device)
        predictions = []

        self.model.train()

        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.model(X_tensor, sample=True)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())

        predictions = np.stack(predictions, axis=0)

        mean_pred = np.mean(predictions, axis=0)
        pred_variance = np.var(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)

        entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=1)

        return {
            'predictions': mean_pred,
            'variance': pred_variance,
            'std': pred_std,
            'entropy': entropy,
            'individual_predictions': predictions
        }


class TemperatureCalibrator:

    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False

    def fit(self, logits, y_true):

        logits_tensor = torch.FloatTensor(logits)
        y_tensor = torch.LongTensor(y_true)

        temperature = nn.Parameter(torch.ones(1))

        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

        def eval_loss():
            loss = F.cross_entropy(logits_tensor / temperature, y_tensor)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        self.temperature = float(temperature.item())
        self.is_fitted = True

        return self

    def transform(self, logits):
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")

        logits_tensor = torch.FloatTensor(logits)
        calibrated_logits = logits_tensor / self.temperature
        calibrated_probs = F.softmax(calibrated_logits, dim=1)

        return calibrated_probs.numpy()

    def fit_transform(self, logits, y_true):
        return self.fit(logits, y_true).transform(logits)


def compute_prediction_intervals(predictions, uncertainties, confidence_level=0.95):

    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    lower_bounds = predictions - z_score * uncertainties
    upper_bounds = predictions + z_score * uncertainties

    lower_bounds = np.maximum(lower_bounds, 0)
    upper_bounds = np.minimum(upper_bounds, 1)

    interval_widths = upper_bounds - lower_bounds

    return {
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'interval_widths': interval_widths,
        'confidence_level': confidence_level
    }


def uncertainty_based_active_learning(uncertainties, n_select, strategy='entropy'):

    if strategy == 'entropy':
        indices = np.argsort(uncertainties)[-n_select:]
    elif strategy == 'variance':
        indices = np.argsort(uncertainties)[-n_select:]
    elif strategy == 'random':
        indices = np.random.choice(len(uncertainties), n_select, replace=False)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return indices


if __name__ == "__main__":
    n_samples = 1000
    n_classes = 4

    y_true = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.dirichlet([1] * n_classes, n_samples)

    analyzer = CalibrationAnalyzer(n_bins=10)
    metrics = analyzer.compute_calibration_metrics(y_true, y_prob)

    print(f"ECE: {metrics.expected_calibration_error:.3f}")
    print(f"MCE: {metrics.maximum_calibration_error:.3f}")
    print(f"Brier Score: {metrics.brier_score:.3f}")
    print(f"Mean Uncertainty: {metrics.mean_uncertainty:.3f}")

    logits = np.random.randn(n_samples, n_classes)
    calibrator = TemperatureCalibrator()
    calibrated_probs = calibrator.fit_transform(logits, y_true)

    print(f"Temperature scaling factor: {calibrator.temperature:.3f}")
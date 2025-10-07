import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyMetrics:
    def __init__(
        self,
        expected_calibration_error=None,
        maximum_calibration_error=None,
        average_calibration_error=None,
        brier_score=None,
        reliability_accuracy=None,
        reliability_confidence=None,
        mean_uncertainty=None,
        max_uncertainty=None,
        uncertainty_std=None,
        confidence_accuracy_correlation=None,
        prediction_entropy=None,
    ):
        self.expected_calibration_error = expected_calibration_error
        self.maximum_calibration_error = maximum_calibration_error
        self.average_calibration_error = average_calibration_error
        self.brier_score = brier_score
        self.reliability_accuracy = reliability_accuracy
        self.reliability_confidence = reliability_confidence
        self.mean_uncertainty = mean_uncertainty
        self.max_uncertainty = max_uncertainty
        self.uncertainty_std = uncertainty_std
        self.confidence_accuracy_correlation = confidence_accuracy_correlation
        self.prediction_entropy = prediction_entropy


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
        (rel_acc, rel_conf) = self._compute_reliability_metrics(accuracy, confidence)
        uncertainty_stats = self._compute_uncertainty_stats(y_prob)
        with np.errstate(invalid="ignore"):
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
            mean_uncertainty=uncertainty_stats["mean"],
            max_uncertainty=uncertainty_stats["max"],
            uncertainty_std=uncertainty_stats["std"],
            confidence_accuracy_correlation=corr,
            prediction_entropy=entropy,
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
        return (float(np.mean(accuracy)), float(np.mean(confidence)))

    def _compute_uncertainty_stats(self, y_prob):
        entropy = -np.sum(y_prob * np.log(y_prob + 1e-08), axis=1)
        return {
            "mean": float(np.mean(entropy)),
            "max": float(np.max(entropy)),
            "std": float(np.std(entropy)),
        }

    def _compute_prediction_entropy(self, y_prob):
        entropy = -np.sum(y_prob * np.log(y_prob + 1e-08), axis=1)
        return float(np.mean(entropy))

    def plot_reliability_diagram(
        self, y_true, y_prob, title="Reliability Diagram", save_path=None
    ):
        confidence = np.max(y_prob, axis=1)
        y_pred = np.argmax(y_prob, axis=1)
        accuracy = (y_pred == y_true).astype(float)
        (fraction_pos, mean_pred) = calibration_curve(
            accuracy, confidence, n_bins=self.n_bins
        )
        (fig, ax) = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")
        ax.plot(
            mean_pred,
            fraction_pos,
            "o-",
            linewidth=2,
            markersize=6,
            label="Model Calibration",
        )
        ax2 = ax.twinx()
        ax2.hist(
            confidence,
            bins=self.n_bins,
            alpha=0.3,
            color="gray",
            density=True,
            label="Confidence Distribution",
        )
        ax2.set_ylabel("Density", alpha=0.7)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ece = self._compute_ece(accuracy, confidence)
        ax.text(
            0.02,
            0.98,
            f"ECE={ece:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig


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


def compute_ece(y_true, y_prob, n_bins=10):
    analyzer = CalibrationAnalyzer(n_bins=n_bins)
    y_pred = np.argmax(y_prob, axis=1)
    confidence = np.max(y_prob, axis=1)
    accuracy = (y_pred == y_true).astype(float)
    return analyzer._compute_ece(accuracy, confidence)


def compute_mce(y_true, y_prob, n_bins=10):
    analyzer = CalibrationAnalyzer(n_bins=n_bins)
    y_pred = np.argmax(y_prob, axis=1)
    confidence = np.max(y_prob, axis=1)
    accuracy = (y_pred == y_true).astype(float)
    return analyzer._compute_mce(accuracy, confidence)


def compute_ace(y_true, y_prob, n_bins=10):
    analyzer = CalibrationAnalyzer(n_bins=n_bins)
    y_pred = np.argmax(y_prob, axis=1)
    confidence = np.max(y_prob, axis=1)
    accuracy = (y_pred == y_true).astype(float)
    return analyzer._compute_ace(accuracy, confidence)


def compute_brier_score(y_true, y_prob):
    analyzer = CalibrationAnalyzer()
    return analyzer._compute_brier_score(y_true, y_prob)


def quick_evaluate(y_true, y_prob, n_bins=10):
    analyzer = CalibrationAnalyzer(n_bins=n_bins)
    return analyzer.compute_calibration_metrics(y_true, y_prob)


if __name__ == "__main__":
    n_samples = 1000
    n_classes = 4
    y_true = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.dirichlet([1] * n_classes, n_samples)
    analyzer = CalibrationAnalyzer(n_bins=10)
    metrics = analyzer.compute_calibration_metrics(y_true, y_prob)
    print("Calibration Metrics=")
    print(f"  ECE={metrics.expected_calibration_error:.3f}")
    print(f"  MCE={metrics.maximum_calibration_error:.3f}")
    print(f"  ACE={metrics.average_calibration_error:.3f}")
    print(f"  Brier Score={metrics.brier_score:.3f}")
    print(f"  Mean Uncertainty={metrics.mean_uncertainty:.3f}")
    logits = np.random.randn(n_samples, n_classes)
    calibrator = TemperatureCalibrator()
    calibrated_probs = calibrator.fit_transform(logits, y_true)
    print(f"\nTemperature scaling factor={calibrator.temperature:.3f}")
    print("Calibration module test passed!")

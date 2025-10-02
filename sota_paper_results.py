import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
from pathlib import Path
import warnings
from dataclasses import asdict

from utility import generate_data, compute_pos_probs, classify_with_gmm, estimate_error
from sota_mlp import AdvancedMLPConfig, create_model
from sota_trainer import AdvancedTrainer, train_ensemble, ModelEnsemble
from uncertainty_quantification import (
    CalibrationAnalyzer, DeepEnsembleUncertainty, TemperatureCalibrator,
    UncertaintyMetrics
)
from paper_utils import (
    create_results_summary_table, generate_latex_table,
    compute_statistical_analysis, format_console_summary
)

torch.manual_seed(42)
np.random.seed(42)

warnings.filterwarnings('ignore')


class SOTAPaperResultsGenerator:

    def __init__(self, results_dir="sota_paper_results", random_seed=42):

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.random_seed = random_seed

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.gmm_params = {
            'priors': np.array([0.25, 0.25, 0.25, 0.25]),
            'meanVectors': np.array([
                [0.0, 0.0, 0.0],
                [4.0, 4.0, 0.0],
                [0.0, 4.0, 4.0],
                [4.0, 0.0, 4.0]
            ]),
            'covMatrices': np.array([
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ])
        }

        self.training_sizes = [100, 500, 1000, 5000, 10000]
        self.test_size = 50000

        self.model_configs = self._create_model_configurations()

        self.calibration_analyzer = CalibrationAnalyzer(n_bins=10)

    def _create_model_configurations(self):

        configs = {}

        basic = AdvancedMLPConfig()
        basic.hidden_dims = [128, 256, 128]
        basic.activation = 'relu'
        basic.use_batch_norm = True
        basic.use_residual = False
        basic.uncertainty_method = 'none'
        basic.max_epochs = 200
        basic.learning_rate = 1e-3
        basic.batch_size = 64
        configs['sota_basic'] = basic

        advanced = AdvancedMLPConfig()
        advanced.hidden_dims = [128, 256, 256, 128]
        advanced.activation = 'swish'
        advanced.use_batch_norm = True
        advanced.use_residual = True
        advanced.use_mixup = True
        advanced.uncertainty_method = 'none'
        advanced.max_epochs = 300
        advanced.learning_rate = 1e-3
        advanced.batch_size = 64
        advanced.weight_decay = 1e-4
        advanced.label_smoothing = 0.1
        configs['sota_advanced'] = advanced

        mc_dropout = AdvancedMLPConfig()
        mc_dropout.hidden_dims = [128, 256, 128]
        mc_dropout.activation = 'swish'
        mc_dropout.use_batch_norm = True
        mc_dropout.dropout_rates = [0.1, 0.15, 0.1]
        mc_dropout.uncertainty_method = 'mc_dropout'
        mc_dropout.mc_samples = 100
        mc_dropout.max_epochs = 250
        mc_dropout.learning_rate = 1e-3
        mc_dropout.batch_size = 64
        mc_dropout.use_mixup = False
        mc_dropout.label_smoothing = 0.0
        mc_dropout.weight_decay = 1e-5
        configs['sota_mc_dropout'] = mc_dropout

        bayesian = AdvancedMLPConfig()
        bayesian.hidden_dims = [128, 256, 128]
        bayesian.activation = 'relu'
        bayesian.uncertainty_method = 'bayesian'
        bayesian.max_epochs = 300
        bayesian.learning_rate = 5e-4
        bayesian.batch_size = 32
        configs['sota_bayesian'] = bayesian

        ensemble = AdvancedMLPConfig()
        ensemble.hidden_dims = [128, 256, 128]
        ensemble.activation = 'swish'
        ensemble.use_batch_norm = True
        ensemble.uncertainty_method = 'ensemble'
        ensemble.num_ensemble_models = 5
        ensemble.max_epochs = 200
        ensemble.learning_rate = 1e-3
        ensemble.batch_size = 64
        configs['sota_ensemble'] = ensemble

        return configs

    def compute_bayes_error(self, n_samples=100000):
        print("Computing theoretical Bayes error")
        X_test, y_test = generate_data(n_samples, self.gmm_params)
        y_bayes = classify_with_gmm(X_test, self.gmm_params)
        bayes_error = estimate_error(y_bayes, y_test)
        print(f"Theoretical Bayes error: {bayes_error:.4f} (computed on {n_samples} samples)")

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_bayes)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        print(f"Per-class Bayes accuracies: {per_class_acc}")

        return bayes_error

    def run_single_experiment(self,
                            config_name,
                            config,
                            train_size,
                            X_test,
                            y_test,
                            run_id=0):

        print(f"Running {config_name} with {train_size} training samples (run {run_id})")

        torch.manual_seed(self.random_seed + run_id * 1000 + train_size)
        np.random.seed(self.random_seed + run_id * 1000 + train_size)

        start_time = time.time()

        X_train, y_train = generate_data(train_size, self.gmm_params)

        X_val, y_val = generate_data(min(2000, train_size // 2), self.gmm_params)

        results = {
            'config_name': config_name,
            'run_id': run_id,
            'train_size': train_size
        }

        if config.uncertainty_method == 'ensemble':
            ensemble = train_ensemble(X_train, y_train, X_val, y_val, config, config.num_ensemble_models)

            ensemble_results = ensemble.predict_with_decomposed_uncertainty(torch.FloatTensor(X_test))
            test_probs = ensemble_results['predictions'].numpy()
            test_predictions = np.argmax(test_probs, axis=1)
            test_logits = ensemble_results['logits'].cpu().numpy()

            aleatoric_uncertainty = ensemble_results['aleatoric_uncertainty'].numpy()
            epistemic_uncertainty = ensemble_results['epistemic_uncertainty'].numpy()
            total_uncertainty = ensemble_results['total_uncertainty'].numpy()

            results.update({
                'mean_aleatoric_uncertainty': float(np.mean(aleatoric_uncertainty)),
                'mean_epistemic_uncertainty': float(np.mean(epistemic_uncertainty)),
                'mean_total_uncertainty': float(np.mean(total_uncertainty))
            })

        else:
            trainer = AdvancedTrainer(config)
            metrics_history = trainer.fit(X_train, y_train, X_val, y_val)

            (
                test_predictions,
                test_probs,
                uncertainty_scores,
                test_logits,
            ) = trainer.predict(X_test)

            results['mean_uncertainty'] = float(np.mean(uncertainty_scores))
            results['training_history'] = [asdict(m) for m in metrics_history[-10:]]

        test_accuracy = np.mean(test_predictions == y_test)
        test_error = 1.0 - test_accuracy

        calibration_metrics = self.calibration_analyzer.compute_calibration_metrics(y_test, test_probs)

        temp_calibrator = TemperatureCalibrator()
        logits_input = (
            ensemble_results['logits'].cpu().numpy()
            if config.uncertainty_method == 'ensemble'
            else test_logits
        )
        calibrated_probs = temp_calibrator.fit_transform(logits_input, y_test)
        calibrated_metrics = self.calibration_analyzer.compute_calibration_metrics(y_test, calibrated_probs)
        results['temperature'] = temp_calibrator.temperature
        results['calibrated_ece'] = calibrated_metrics.expected_calibration_error

        training_time = time.time() - start_time

        logits_filename = f"logits_{config_name}_train{train_size}_run{run_id}.npy"
        np.save(self.results_dir / logits_filename, test_logits)

        results.update({
            'test_accuracy': test_accuracy,
            'test_error': test_error,
            'training_time': training_time,
            'ece': calibration_metrics.expected_calibration_error,
            'mce': calibration_metrics.maximum_calibration_error,
            'brier_score': calibration_metrics.brier_score,
            'confidence_accuracy_correlation': calibration_metrics.confidence_accuracy_correlation,
            'prediction_entropy': calibration_metrics.prediction_entropy,
            'logits_file': logits_filename
        })

        print(f"  Test accuracy: {test_accuracy:.4f}, ECE: {calibration_metrics.expected_calibration_error:.4f}")

        return results

    def run_comprehensive_comparison(self, n_runs=3):
        X_test, y_test = generate_data(self.test_size, self.gmm_params)

        all_results = []

        for config_name, config in self.model_configs.items():
            print(f"\n{'='*60}")
            print(f"TESTING CONFIGURATION: {config_name.upper()}")
            print(f"{'='*60}")

            for train_size in self.training_sizes:
                for run_id in range(n_runs):
                    result = self.run_single_experiment(
                        config_name, config, train_size, X_test, y_test, run_id
                    )
                    all_results.append(result)

        return all_results

    def analyze_results(self, results, bayes_error):

        df = pd.DataFrame(results)

        df = df[df['test_accuracy'] > 0]

        if len(df) == 0:
            print("No successful experiments found!")
            return df

        agg_results = []

        for config_name in df['config_name'].unique():
            config_df = df[df['config_name'] == config_name]

            for train_size in config_df['train_size'].unique():
                size_df = config_df[config_df['train_size'] == train_size]

                if len(size_df) == 0:
                    continue

                agg_result = {
                    'config_name': config_name,
                    'train_size': train_size,
                    'test_accuracy_mean': size_df['test_accuracy'].mean(),
                    'test_accuracy_std': size_df['test_accuracy'].std(),
                    'test_error_mean': size_df['test_error'].mean(),
                    'bayes_gap_mean': size_df['test_error'].mean() - bayes_error,
                    'ece_mean': size_df['ece'].mean(),
                    'ece_std': size_df['ece'].std(),
                    'brier_score_mean': size_df['brier_score'].mean(),
                    'training_time_mean': size_df['training_time'].mean(),
                    'n_runs': len(size_df)
                }

                if 'mean_uncertainty' in size_df.columns:
                    agg_result['uncertainty_mean'] = size_df['mean_uncertainty'].mean()

                if 'mean_total_uncertainty' in size_df.columns:
                    agg_result['total_uncertainty_mean'] = size_df['mean_total_uncertainty'].mean()
                    agg_result['epistemic_uncertainty_mean'] = size_df['mean_epistemic_uncertainty'].mean()
                    agg_result['aleatoric_uncertainty_mean'] = size_df['mean_aleatoric_uncertainty'].mean()

                agg_results.append(agg_result)

        return pd.DataFrame(agg_results)

    def create_comparison_plots(self, results_df, bayes_error):
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        for config_name in results_df['config_name'].unique():
            config_data = results_df[results_df['config_name'] == config_name]
            config_data = config_data.sort_values('train_size')

            ax1.errorbar(config_data['train_size'], config_data['test_accuracy_mean'],
                        yerr=config_data['test_accuracy_std'], marker='o',
                        label=config_name, linewidth=2, markersize=6)

        ax1.axhline(y=1-bayes_error, color='red', linestyle='--', linewidth=2, label='Bayes Optimal', alpha=0.8)
        ax1.set_xscale('log')
        ax1.set_xlabel('Training Set Size')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Model Performance Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for config_name in results_df['config_name'].unique():
            config_data = results_df[results_df['config_name'] == config_name]
            config_data = config_data.sort_values('train_size')

            ax2.errorbar(config_data['train_size'], config_data['ece_mean'],
                        yerr=config_data['ece_std'], marker='s',
                        label=config_name, linewidth=2, markersize=6)

        ax2.set_xscale('log')
        ax2.set_xlabel('Training Set Size')
        ax2.set_ylabel('Expected Calibration Error')
        ax2.set_title('Calibration Quality Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        for config_name in results_df['config_name'].unique():
            config_data = results_df[results_df['config_name'] == config_name]
            config_data = config_data.sort_values('train_size')

            ax3.plot(config_data['train_size'], config_data['bayes_gap_mean'],
                    marker='^', label=config_name, linewidth=2, markersize=6)

        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax3.set_xscale('log')
        ax3.set_xlabel('Training Set Size')
        ax3.set_ylabel('Performance Gap (vs Bayes Optimal)')
        ax3.set_title('Gap from Theoretical Optimum')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        for config_name in results_df['config_name'].unique():
            config_data = results_df[results_df['config_name'] == config_name]
            config_data = config_data.sort_values('train_size')

            ax4.plot(config_data['train_size'], config_data['training_time_mean'],
                    marker='d', label=config_name, linewidth=2, markersize=6)

        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.set_xlabel('Training Set Size')
        ax4.set_ylabel('Training Time (seconds)')
        ax4.set_title('Computational Efficiency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle('SOTA ML Techniques Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'sota_ml_comparison.pdf')
        plt.savefig(self.results_dir / 'sota_ml_comparison.png', dpi=300)
        plt.close()

        print(f"Comparison plots saved to {self.results_dir}")

    def save_comprehensive_results(self, results, results_df, bayes_error):
        with open(self.results_dir / 'raw_sota_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        results_df.to_csv(self.results_dir / 'sota_ml_comparison.csv', index=False)

        latex_table = generate_latex_table(
            results_df[['config_name', 'train_size', 'test_accuracy_mean', 'ece_mean', 'bayes_gap_mean']],
            caption="Comparison of SOTA ML Techniques vs Traditional MLP",
            label="tab:sota_ml_comparison"
        )

        with open(self.results_dir / 'sota_ml_table.tex', 'w') as f:
            f.write(latex_table)

        best_results = results_df.loc[results_df.groupby('train_size')['test_accuracy_mean'].idxmax()]

        report_lines = [
            "SOTA ML TECHNIQUES COMPARISON REPORT",
            "=" * 60,
            f"Theoretical Bayes error: {bayes_error:.4f}",
            "",
            "BEST PERFORMING METHODS BY TRAINING SIZE:",
        ]

        for _, row in best_results.iterrows():
            report_lines.append(
                f"  {row['train_size']:>5} samples: {row['config_name']:<15} "
                f"Acc: {row['test_accuracy_mean']:.4f} Gap: {row['bayes_gap_mean']:.4f}"
            )

        report_lines.extend([
            "",
            "OVERALL PERFORMANCE SUMMARY:",
            f"Best overall accuracy: {results_df['test_accuracy_mean'].max():.4f}",
            f"Best calibration (ECE): {results_df['ece_mean'].min():.4f}",
            f"Smallest Bayes gap: {results_df['bayes_gap_mean'].min():.4f}",
        ])

        report = "\n".join(report_lines)

        with open(self.results_dir / 'sota_ml_report.txt', 'w') as f:
            f.write(report)

        print(f"  Results saved to {self.results_dir}")

    def run_complete_analysis(self, n_runs=2):
        bayes_error = self.compute_bayes_error()
        results = self.run_comprehensive_comparison(n_runs)
        results_df = self.analyze_results(results, bayes_error)

        if len(results_df) == 0:
            print("No successful experiments to analyze!")
            return results

        self.create_comparison_plots(results_df, bayes_error)
        self.save_comprehensive_results(results, results_df, bayes_error)

        print("\n" + "=" * 80)
        print(f"Completed {len(results)} experiments across {len(results_df)} configurations")
        print(f"Best accuracy: {results_df['test_accuracy_mean'].max():.4f}")
        print(f"Best calibration: {results_df['ece_mean'].min():.4f}")
        print(f"Results saved to: {self.results_dir.absolute()}")

        return results


if __name__ == "__main__":
    import torch
    print(f"PyTorch available: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    generator = SOTAPaperResultsGenerator(random_seed=42)
    results = generator.run_complete_analysis(n_runs=2)

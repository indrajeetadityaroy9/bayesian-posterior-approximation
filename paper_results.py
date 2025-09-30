import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import time
import json
from pathlib import Path

from utility import (
    generate_data, compute_pos_probs, classify_with_gmm,
    runMultipleKFoldCV, train_mlp, estimate_error
)
from paper_utils import generate_latex_table

matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.0,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight'
})


class PaperResultsGenerator:

    def __init__(self, results_dir="paper_results", random_seed=42):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.gmm_params = {
            'priors': np.array([0.25, 0.25, 0.25, 0.25]),
            'meanVectors': np.array([
                [0.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [7.5, 0.0, 0.0]
            ]),
            'covMatrices': np.array([
                [[1.0, 0.3, 1.4], [0.3, 1.0, 0.3], [1.4, 0.3, 7.0]],
                [[1.0, -0.4, -0.7], [-0.4, 1.0, -0.4], [-0.7, -0.4, 3.0]],
                [[1.0, 0.4, 0.7], [0.4, 1.0, 0.4], [0.7, 0.4, 3.0]],
                [[1.0, -0.3, -1.4], [-0.3, 1.0, -0.3], [-1.4, -0.3, 7.0]]
            ])
        }

        self.training_sizes = [100, 500, 1000, 5000, 10000]
        self.test_size = 100000

    def compute_bayes_error(self, n_samples=100000):
        print("Computing theoretical Bayes error...")
        X_test, y_test = generate_data(n_samples, self.gmm_params)
        y_bayes = classify_with_gmm(X_test, self.gmm_params)
        bayes_error = estimate_error(y_bayes, y_test)
        print(f"Theoretical Bayes error: {bayes_error:.4f}")
        return bayes_error

    def run_single_experiment(self, train_size, run_id=0):
        print(f"Running experiment with {train_size} training samples (run {run_id})")

        np.random.seed(self.random_seed + run_id * 1000 + train_size)

        start_time = time.time()

        X_train, y_train = generate_data(train_size, self.gmm_params)

        cv_score, cv_error, optimal_p, cv_errors = runMultipleKFoldCV(
            X_train, y_train, K=5, num_runs=3
        )

        best_mlp, best_ll = train_mlp(X_train, y_train, optimal_p, num_initializations=5)

        X_test, y_test = generate_data(self.test_size, self.gmm_params)

        y_pred = best_mlp.predict(X_test)
        y_proba = best_mlp.predict_proba(X_test)
        test_accuracy = 1.0 - estimate_error(y_pred, y_test)
        test_error = estimate_error(y_pred, y_test)

        from sklearn.metrics import log_loss, precision_recall_fscore_support
        test_log_loss = log_loss(y_test, y_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        training_time = time.time() - start_time

        results = {
            'run_id': run_id,
            'train_size': train_size,
            'test_accuracy': test_accuracy,
            'test_error': test_error,
            'test_log_loss': test_log_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_score': cv_score,
            'cv_error': cv_error,
            'optimal_hidden_units': optimal_p,
            'training_time': training_time,
            'log_likelihood': best_ll
        }

        print(f"  Test accuracy: {test_accuracy:.4f}, Optimal units: {optimal_p}")
        return results

    def run_multiple_runs(self, n_runs=3):
        all_results = []

        for train_size in self.training_sizes:
            for run_id in range(n_runs):
                result = self.run_single_experiment(train_size, run_id)
                all_results.append(result)

        return all_results

    def create_publication_plots(self, results, bayes_error):
        print("Creating publication-quality plots...")

        size_to_results = {}
        for result in results:
            size = result['train_size']
            if size not in size_to_results:
                size_to_results[size] = []
            size_to_results[size].append(result)

        train_sizes = sorted(size_to_results.keys())
        test_errors_mean = []
        test_errors_std = []
        optimal_units_mean = []
        cv_errors_mean = []

        for size in train_sizes:
            size_results = size_to_results[size]
            test_errors = [r['test_error'] for r in size_results]
            units = [r['optimal_hidden_units'] for r in size_results]
            cv_errs = [r['cv_error'] for r in size_results]

            test_errors_mean.append(np.mean(test_errors))
            test_errors_std.append(np.std(test_errors))
            optimal_units_mean.append(np.mean(units))
            cv_errors_mean.append(np.mean(cv_errs))

        test_errors_mean = np.array(test_errors_mean)
        test_errors_std = np.array(test_errors_std)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.errorbar(train_sizes, test_errors_mean, yerr=test_errors_std,
                   marker='o', linestyle='-', linewidth=2, markersize=6,
                   label='MLP Test Error', color='blue', capsize=5)

        ax.axhline(y=bayes_error, color='red', linestyle='--', linewidth=2,
                  label='Bayes Optimal Error')

        ax.set_xscale('log')
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel('Error Rate')
        ax.set_title('Learning Curves: MLP vs Bayes Optimal Classifier')
        ax.legend()
        ax.grid(True, alpha=0.3)

        for i, size in enumerate(train_sizes):
            ax.annotate(f'{test_errors_mean[i]:.3f}',
                       (size, test_errors_mean[i]),
                       textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'learning_curves.pdf')
        plt.savefig(self.results_dir / 'learning_curves.png', dpi=150)
        print(f"  Learning curves saved to {self.results_dir}")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(train_sizes, optimal_units_mean, marker='s', linestyle='-',
               linewidth=2, markersize=8, color='green')

        ax.set_xscale('log')
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel('Optimal Number of Hidden Units')
        ax.set_title('Optimal Architecture vs Training Set Size')
        ax.grid(True, alpha=0.3)

        for i, (size, units) in enumerate(zip(train_sizes, optimal_units_mean)):
            ax.annotate(f'{int(units)}',
                       (size, units),
                       textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'architecture_selection.pdf')
        plt.savefig(self.results_dir / 'architecture_selection.png', dpi=150)
        print(f"  Architecture plot saved to {self.results_dir}")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(train_sizes, cv_errors_mean, marker='^', linestyle='-',
               linewidth=2, markersize=6, color='purple', label='CV Error')

        ax.axhline(y=bayes_error, color='red', linestyle='--', linewidth=2,
                  label='Bayes Optimal Error')

        ax.set_xscale('log')
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel('Cross-Validation Error Rate')
        ax.set_title('Cross-Validation Error vs Training Set Size')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'cv_analysis.pdf')
        plt.savefig(self.results_dir / 'cv_analysis.png', dpi=150)
        print(f"  CV analysis plot saved to {self.results_dir}")

        plt.close('all')

    def save_results_tables(self, results, bayes_error):
        print("Saving results tables...")

        size_to_results = {}
        for result in results:
            size = result['train_size']
            if size not in size_to_results:
                size_to_results[size] = []
            size_to_results[size].append(result)

        aggregated_results = []
        for size in sorted(size_to_results.keys()):
            size_results = size_to_results[size]

            mean_result = {
                'train_size': size,
                'test_accuracy': np.mean([r['test_accuracy'] for r in size_results]),
                'test_error': np.mean([r['test_error'] for r in size_results]),
                'test_accuracy_std': np.std([r['test_accuracy'] for r in size_results]),
                'test_log_loss': np.mean([r['test_log_loss'] for r in size_results]),
                'f1_score': np.mean([r['f1_score'] for r in size_results]),
                'optimal_hidden_units': int(np.mean([r['optimal_hidden_units'] for r in size_results])),
                'cv_error': np.mean([r['cv_error'] for r in size_results]),
                'training_time': np.mean([r['training_time'] for r in size_results]),
                'n_runs': len(size_results)
            }
            aggregated_results.append(mean_result)

        results_data = []
        for result in aggregated_results:
            results_data.append({
                'Training Size': result['train_size'],
                'Test Accuracy': f"{result['test_accuracy']:.4f} ± {result['test_accuracy_std']:.4f}",
                'Test Error Rate': f"{result['test_error']:.4f}",
                'Bayes Gap': f"{result['test_error'] - bayes_error:.4f}",
                'Optimal Hidden Units': result['optimal_hidden_units'],
                'CV Error': f"{result['cv_error']:.4f}",
                'Training Time (s)': f"{result['training_time']:.2f}",
                'Log Loss': f"{result['test_log_loss']:.4f}",
                'F1 Score': f"{result['f1_score']:.4f}"
            })

        df_results = pd.DataFrame(results_data)

        csv_path = self.results_dir / 'results_summary.csv'
        df_results.to_csv(csv_path, index=False)
        print(f"  Results CSV saved to {csv_path}")

        latex_table = generate_latex_table(
            df_results,
            caption="MLP Classification Results Compared to Bayes Optimal Classifier",
            label="tab:mlp_results"
        )

        latex_path = self.results_dir / 'results_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"  LaTeX table saved to {latex_path}")

        raw_results_path = self.results_dir / 'raw_results.json'
        with open(raw_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Raw results saved to {raw_results_path}")

    def generate_statistical_report(self, results, bayes_error):
        accuracies = [r['test_accuracy'] for r in results]
        errors = [r['test_error'] for r in results]
        gaps = [r['test_error'] - bayes_error for r in results]

        report_lines = [
            "STATISTICAL ANALYSIS REPORT",
            "=" * 50,
            f"Total experiments: {len(results)}",
            f"Training sizes: {sorted(set([r['train_size'] for r in results]))}",
            f"Theoretical Bayes error: {bayes_error:.4f}",
            "",
            "TEST ACCURACY STATISTICS:",
            f"  Mean: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
            f"  Range: [{np.min(accuracies):.4f}, {np.max(accuracies):.4f}]",
            f"  Best: {np.max(accuracies):.4f}",
            "",
            "PERFORMANCE GAP STATISTICS:",
            f"  Mean gap: {np.mean(gaps):.4f} ± {np.std(gaps):.4f}",
            f"  Best gap: {np.min(gaps):.4f}",
            f"  Relative error increase: {(np.mean(errors) - bayes_error) / bayes_error * 100:.1f}%"
        ]

        report_lines.extend(["", "TRAINING SIZE ANALYSIS:"])
        size_to_acc = {}
        for result in results:
            size = result['train_size']
            if size not in size_to_acc:
                size_to_acc[size] = []
            size_to_acc[size].append(result['test_accuracy'])

        for size in sorted(size_to_acc.keys()):
            accs = size_to_acc[size]
            report_lines.append(f"  {size:>5} samples: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

        report = "\n".join(report_lines)

        report_path = self.results_dir / 'statistical_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Statistical report saved to {report_path}")

        return report

    def run_complete_analysis(self, n_runs=3):
        print("=" * 60)
        print("RUNNING COMPLETE PAPER-READY ANALYSIS")
        print("=" * 60)

        bayes_error = self.compute_bayes_error()

        print(f"\nRunning {len(self.training_sizes)} × {n_runs} = {len(self.training_sizes) * n_runs} experiments...")
        results = self.run_multiple_runs(n_runs)

        self.create_publication_plots(results, bayes_error)

        self.save_results_tables(results, bayes_error)

        statistical_report = self.generate_statistical_report(results, bayes_error)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {self.results_dir.absolute()}")
        print(f"Generated {len(results)} experiment results")
        print(f"Best accuracy: {max([r['test_accuracy'] for r in results]):.4f}")
        print(f"Bayes error: {bayes_error:.4f}")
        print("\nFiles generated:")
        for file_path in sorted(self.results_dir.glob("*")):
            print(f"  - {file_path.name}")

        return results


if __name__ == "__main__":
    generator = PaperResultsGenerator(random_seed=42)
    results = generator.run_complete_analysis(n_runs=3)
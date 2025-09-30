import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json


def create_results_summary_table(experiments_data, save_path=None):
    summary_data = []

    for exp in experiments_data:
        config = exp['config']
        train_size = config['train_size']

        test_acc = exp['test_metrics'].get('accuracy', 0.0)
        test_error = 1.0 - test_acc
        train_acc = exp['train_metrics'].get('accuracy', 0.0)

        best_params = exp['best_params']
        hidden_units = best_params.get('mlp__hidden_layer_sizes', [0])[0]
        alpha = best_params.get('mlp__alpha', 0.0)
        activation = best_params.get('mlp__activation', 'unknown')

        training_time = exp.get('training_time', 0.0)

        uncertainty = exp.get('uncertainty_metrics', {})
        mean_uncertainty = uncertainty.get('mean_uncertainty', 0.0)

        summary_data.append({
            'Training_Size': train_size,
            'Test_Accuracy': test_acc,
            'Test_Error_Rate': test_error,
            'Train_Accuracy': train_acc,
            'Optimal_Hidden_Units': hidden_units,
            'Alpha_Regularization': alpha,
            'Activation_Function': activation,
            'Training_Time_Sec': training_time,
            'Mean_Uncertainty': mean_uncertainty,
            'Log_Loss': exp['test_metrics'].get('log_loss', 0.0),
            'F1_Score': exp['test_metrics'].get('f1_score', 0.0)
        })

    df = pd.DataFrame(summary_data)
    df = df.sort_values('Training_Size')

    if save_path:
        df.to_csv(save_path, index=False, float_format='%.4f')
        print(f"Results summary saved to: {save_path}")

    return df


def create_architecture_analysis(experiments_data, save_path=None):
    arch_data = []

    for exp in experiments_data:
        config = exp['config']
        train_size = config['train_size']
        best_params = exp['best_params']

        arch_data.append({
            'Training_Size': train_size,
            'Optimal_Hidden_Units': best_params.get('mlp__hidden_layer_sizes', [0])[0],
            'Alpha': best_params.get('mlp__alpha', 0.0),
            'Activation': best_params.get('mlp__activation', 'unknown'),
            'CV_Score': exp.get('cv_score', 0.0),
            'Test_Accuracy': exp['test_metrics'].get('accuracy', 0.0)
        })

    df = pd.DataFrame(arch_data)
    df = df.sort_values('Training_Size')

    if save_path:
        df.to_csv(save_path, index=False, float_format='%.4f')
        print(f"Architecture analysis saved to: {save_path}")

    return df


def generate_latex_table(df, caption="Experimental Results", label="tab:results", save_path=None):
    n_cols = len(df.columns)
    col_spec = "l" + "r" * (n_cols - 1)

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule"
    ]

    header = " & ".join([col.replace('_', '\\_') for col in df.columns]) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")

    for _, row in df.iterrows():
        formatted_row = []
        for i, val in enumerate(row):
            if isinstance(val, (int, float)):
                if isinstance(val, int) or val == int(val):
                    formatted_row.append(f"{int(val)}")
                else:
                    formatted_row.append(f"{val:.4f}")
            else:
                formatted_row.append(str(val))

        row_str = " & ".join(formatted_row) + " \\\\"
        latex_lines.append(row_str)

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    latex_table = "\n".join(latex_lines)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {save_path}")

    return latex_table


def compute_statistical_analysis(experiments_data, metric='test_accuracy', save_path=None):
    values = []
    training_sizes = []

    for exp in experiments_data:
        train_size = exp['config']['train_size']
        training_sizes.append(train_size)

        if metric == 'test_accuracy':
            values.append(exp['test_metrics'].get('accuracy', 0.0))
        elif metric == 'test_error_rate':
            values.append(1.0 - exp['test_metrics'].get('accuracy', 0.0))
        elif metric == 'training_time':
            values.append(exp.get('training_time', 0.0))
        else:
            values.append(exp['test_metrics'].get(metric, 0.0))

    values = np.array(values)

    stats_analysis = {
        'metric': metric,
        'n_experiments': len(values),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75))
    }

    if len(values) > 1:
        ci_95 = stats.t.interval(0.95, len(values)-1,
                               loc=np.mean(values),
                               scale=stats.sem(values))
        stats_analysis['ci_95_lower'] = float(ci_95[0])
        stats_analysis['ci_95_upper'] = float(ci_95[1])

    if len(values) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(values)
        stats_analysis['shapiro_stat'] = float(shapiro_stat)
        stats_analysis['shapiro_p_value'] = float(shapiro_p)
        stats_analysis['is_normal'] = bool(shapiro_p > 0.05)

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(stats_analysis, f, indent=2)
        print(f"Statistical analysis saved to: {save_path}")

    return stats_analysis


def create_performance_comparison(experiments_data, bayes_error=None, save_path=None):
    comparison_data = []

    for exp in experiments_data:
        train_size = exp['config']['train_size']
        test_acc = exp['test_metrics'].get('accuracy', 0.0)
        test_error = 1.0 - test_acc

        row = {
            'Training_Size': train_size,
            'MLP_Test_Error': test_error,
            'MLP_Test_Accuracy': test_acc
        }

        if bayes_error is not None:
            row['Bayes_Optimal_Error'] = bayes_error
            row['Bayes_Optimal_Accuracy'] = 1.0 - bayes_error
            row['Performance_Gap'] = test_error - bayes_error
            row['Relative_Error_Increase'] = (test_error - bayes_error) / bayes_error if bayes_error > 0 else np.inf

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Training_Size')

    if save_path:
        df.to_csv(save_path, index=False, float_format='%.4f')
        print(f"Performance comparison saved to: {save_path}")

    return df


def format_console_summary(experiments_data, bayes_error=None):
    lines = ["=" * 60]
    lines.append("PAPER RESULTS SUMMARY")
    lines.append("=" * 60)

    accuracies = [exp['test_metrics'].get('accuracy', 0.0) for exp in experiments_data]
    best_acc = max(accuracies)
    best_exp = experiments_data[np.argmax(accuracies)]

    lines.append(f"\nBest Test Accuracy: {best_acc:.4f}")
    lines.append(f"Training Size: {best_exp['config']['train_size']}")
    lines.append(f"Optimal Architecture: {best_exp['best_params'].get('mlp__hidden_layer_sizes', [0])[0]} hidden units")

    if bayes_error is not None:
        best_error = 1.0 - best_acc
        lines.append(f"\nBayes Optimal Error: {bayes_error:.4f}")
        lines.append(f"Best MLP Error: {best_error:.4f}")
        lines.append(f"Performance Gap: {best_error - bayes_error:.4f}")

    lines.append("\nPerformance by Training Size:")
    lines.append("-" * 40)
    lines.append(f"{'Size':<8} {'Accuracy':<10} {'Error':<10} {'Units':<8}")
    lines.append("-" * 40)

    for exp in sorted(experiments_data, key=lambda x: x['config']['train_size']):
        size = exp['config']['train_size']
        acc = exp['test_metrics'].get('accuracy', 0.0)
        error = 1.0 - acc
        units = exp['best_params'].get('mlp__hidden_layer_sizes', [0])[0]
        lines.append(f"{size:<8} {acc:<10.4f} {error:<10.4f} {units:<8}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    lines.append(f"\nStatistical Summary:")
    lines.append(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    lines.append(f"Training Sizes: {sorted([exp['config']['train_size'] for exp in experiments_data])}")

    lines.append("=" * 60)

    return "\n".join(lines)


def load_experiments_from_results_dir(results_dir):
    experiments = []

    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        exp_data = json.load(f)
                    experiments.append(exp_data)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not load {results_file}: {e}")

    return experiments


if __name__ == "__main__":
    results_dir = Path("results")
    if results_dir.exists():
        experiments = load_experiments_from_results_dir(results_dir)
        if experiments:
            print(f"Loaded {len(experiments)} experiments")

            summary_df = create_results_summary_table(experiments)
            print("\nResults Summary:")
            print(summary_df.to_string(index=False))

            console_summary = format_console_summary(experiments)
            print(console_summary)
        else:
            print("No experiment results found")
    else:
        print("Results directory not found")
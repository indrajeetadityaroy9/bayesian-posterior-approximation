import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def set_publication_style():
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Computer Modern Roman'],
        'text.usetex': False,
        'axes.linewidth': 1.2,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.transparent': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0,
        'lines.markersize': 6
    })


def plot_learning_curves_advanced(train_sizes, test_errors, test_errors_std, bayes_error, cv_errors=None, save_path=None, show_confidence=True):
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 7))

    train_sizes = np.array(train_sizes)
    test_errors = np.array(test_errors)
    test_errors_std = np.array(test_errors_std)

    if show_confidence:
        ax.fill_between(train_sizes,
                       test_errors - test_errors_std,
                       test_errors + test_errors_std,
                       alpha=0.2, color='blue', label='_nolegend_')

    ax.plot(train_sizes, test_errors, 'o-', color='blue', linewidth=2.5,
           markersize=8, label='MLP Test Error', markerfacecolor='white',
           markeredgewidth=2, markeredgecolor='blue')

    if cv_errors is not None:
        ax.plot(train_sizes, cv_errors, 's--', color='green', linewidth=2,
               markersize=6, label='Cross-Validation Error', alpha=0.8)

    ax.axhline(y=bayes_error, color='red', linestyle='--', linewidth=2.5,
              label='Bayes Optimal Error', alpha=0.9)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Training Samples', fontweight='bold')
    ax.set_ylabel('Error Rate', fontweight='bold')
    ax.set_title('Learning Curves: MLP vs Bayes Optimal Classifier',
                fontweight='bold', pad=20)

    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    for i, (size, error) in enumerate(zip(train_sizes, test_errors)):
        gap = error - bayes_error
        ax.annotate(f'{error:.3f}\n(+{gap:.3f})',
                   (size, error),
                   textcoords="offset points",
                   xytext=(0, 15), ha='center', va='bottom',
                   fontsize=9, color='blue',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                           edgecolor='blue', alpha=0.8))

    ax.set_ylim(bottom=0, top=max(test_errors + test_errors_std) * 1.1)
    ax.set_xlim(left=min(train_sizes) * 0.8, right=max(train_sizes) * 1.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to {save_path}")

    return fig


def plot_architecture_selection(train_sizes, optimal_units, units_std=None, save_path=None):
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    train_sizes = np.array(train_sizes)
    optimal_units = np.array(optimal_units)

    ax.plot(train_sizes, optimal_units, 'o-', color='darkgreen', linewidth=3,
           markersize=10, label='Optimal Hidden Units', markerfacecolor='lightgreen',
           markeredgewidth=2, markeredgecolor='darkgreen')

    if units_std is not None:
        ax.errorbar(train_sizes, optimal_units, yerr=units_std,
                   color='darkgreen', capsize=5, capthick=2, alpha=0.7)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Training Samples', fontweight='bold')
    ax.set_ylabel('Optimal Number of Hidden Units', fontweight='bold')
    ax.set_title('Model Complexity vs Training Set Size', fontweight='bold', pad=20)

    for i, (size, units) in enumerate(zip(train_sizes, optimal_units)):
        ax.annotate(f'{int(units)}',
                   (size, units),
                   textcoords="offset points",
                   xytext=(0, 15), ha='center', va='bottom',
                   fontsize=11, fontweight='bold', color='darkgreen',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen',
                           edgecolor='darkgreen', alpha=0.8))

    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    ax.set_ylim(bottom=0, top=max(optimal_units) * 1.15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Architecture plot saved to {save_path}")

    return fig


def plot_performance_comparison(train_sizes, mlp_errors, bayes_error, gaussian_nb_errors=None, save_path=None):
    set_publication_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    train_sizes = np.array(train_sizes)
    mlp_errors = np.array(mlp_errors)

    ax1.plot(train_sizes, mlp_errors, 'o-', color='blue', linewidth=2.5,
            markersize=8, label='MLP', markerfacecolor='white',
            markeredgewidth=2, markeredgecolor='blue')

    if gaussian_nb_errors is not None:
        ax1.plot(train_sizes, gaussian_nb_errors, 's--', color='purple',
                linewidth=2, markersize=6, label='Gaussian Naive Bayes', alpha=0.8)

    ax1.axhline(y=bayes_error, color='red', linestyle='--', linewidth=2.5,
               label='Bayes Optimal', alpha=0.9)

    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Training Samples', fontweight='bold')
    ax1.set_ylabel('Error Rate', fontweight='bold')
    ax1.set_title('Performance Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    performance_gaps = mlp_errors - bayes_error
    ax2.plot(train_sizes, performance_gaps, 'o-', color='orange', linewidth=2.5,
            markersize=8, label='MLP - Bayes Gap', markerfacecolor='white',
            markeredgewidth=2, markeredgecolor='orange')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Training Samples', fontweight='bold')
    ax2.set_ylabel('Performance Gap', fontweight='bold')
    ax2.set_title('Performance Gap vs Bayes Optimal', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    for i, (size, gap) in enumerate(zip(train_sizes, performance_gaps)):
        ax2.annotate(f'{gap:.3f}',
                    (size, gap),
                    textcoords="offset points",
                    xytext=(0, 10), ha='center', va='bottom',
                    fontsize=9, color='orange')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved to {save_path}")

    return fig


def plot_convergence_analysis(train_sizes, training_times, test_errors, save_path=None):
    set_publication_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(train_sizes, training_times, 'o-', color='red', linewidth=2.5,
            markersize=8, label='Training Time')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Training Samples', fontweight='bold')
    ax1.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax1.set_title('Training Time Scaling', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    scatter = ax2.scatter(training_times, test_errors, s=100,
                         c=train_sizes, cmap='viridis', alpha=0.7, edgecolors='black')
    ax2.set_xlabel('Training Time (seconds)', fontweight='bold')
    ax2.set_ylabel('Test Error Rate', fontweight='bold')
    ax2.set_title('Time-Accuracy Tradeoff', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Training Set Size', fontweight='bold')

    for i, (time, error, size) in enumerate(zip(training_times, test_errors, train_sizes)):
        ax2.annotate(f'{size}',
                    (time, error),
                    textcoords="offset points",
                    xytext=(5, 5), ha='left', va='bottom',
                    fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Convergence analysis saved to {save_path}")

    return fig


def plot_statistical_summary(results_df, save_path=None):
    set_publication_style()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.hist(results_df['test_accuracy'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(results_df['test_accuracy'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {results_df["test_accuracy"].mean():.3f}')
    ax1.set_xlabel('Test Accuracy', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Distribution of Test Accuracies', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    train_size_groups = results_df.groupby('train_size')['test_accuracy'].apply(list)
    ax2.boxplot([train_size_groups[size] for size in sorted(train_size_groups.keys())],
               labels=sorted(train_size_groups.keys()))
    ax2.set_xlabel('Training Set Size', fontweight='bold')
    ax2.set_ylabel('Test Accuracy', fontweight='bold')
    ax2.set_title('Accuracy Distribution by Training Size', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax3.scatter(results_df['train_size'], results_df['training_time'],
               alpha=0.6, color='green', s=50)
    ax3.set_xscale('log')
    ax3.set_xlabel('Training Set Size', fontweight='bold')
    ax3.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax3.set_title('Training Time vs Dataset Size', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    architecture_counts = results_df['optimal_hidden_units'].value_counts().sort_index()
    ax4.bar(architecture_counts.index, architecture_counts.values,
           alpha=0.7, color='purple', edgecolor='black')
    ax4.set_xlabel('Optimal Hidden Units', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Distribution of Optimal Architectures', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Statistical Summary of Experiments', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Statistical summary saved to {save_path}")

    return fig


def create_publication_figure_set(results, bayes_error, save_dir):
    save_dir.mkdir(exist_ok=True)

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
    training_times_mean = []
    cv_errors_mean = []

    for size in train_sizes:
        size_results = size_to_results[size]
        test_errors = [r['test_error'] for r in size_results]
        units = [r['optimal_hidden_units'] for r in size_results]
        times = [r['training_time'] for r in size_results]
        cv_errs = [r['cv_error'] for r in size_results]

        test_errors_mean.append(np.mean(test_errors))
        test_errors_std.append(np.std(test_errors))
        optimal_units_mean.append(np.mean(units))
        training_times_mean.append(np.mean(times))
        cv_errors_mean.append(np.mean(cv_errs))

    figures = {}

    figures['learning_curves'] = plot_learning_curves_advanced(
        train_sizes, test_errors_mean, test_errors_std, bayes_error,
        cv_errors=cv_errors_mean, save_path=save_dir / 'figure1_learning_curves'
    )

    figures['architecture'] = plot_architecture_selection(
        train_sizes, optimal_units_mean, save_path=save_dir / 'figure2_architecture'
    )

    figures['performance'] = plot_performance_comparison(
        train_sizes, test_errors_mean, bayes_error,
        save_path=save_dir / 'figure3_performance'
    )

    figures['convergence'] = plot_convergence_analysis(
        train_sizes, training_times_mean, test_errors_mean,
        save_path=save_dir / 'figure4_convergence'
    )

    results_df = pd.DataFrame(results)
    figures['statistics'] = plot_statistical_summary(
        results_df, save_path=save_dir / 'figure5_statistics'
    )

    print(f"\nComplete figure set saved to {save_dir}")
    print("Generated figures:")
    for name in figures.keys():
        print(f"  - {name}")

    return figures


if __name__ == "__main__":
    set_publication_style()
    print("Publication plotting module loaded successfully!")
    print("Available functions:")
    print("  - plot_learning_curves_advanced()")
    print("  - plot_architecture_selection()")
    print("  - plot_performance_comparison()")
    print("  - plot_convergence_analysis()")
    print("  - plot_statistical_summary()")
    print("  - create_publication_figure_set()")